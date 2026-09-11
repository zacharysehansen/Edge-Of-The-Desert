"""Guard the one piece of logic this project implements twice.

`frontend/catalog.js` turns a scenario (policy deltas + month + duration) into the
88 raw model inputs the browser feeds to ONNX. `slider_sensitivity.py` has to do
exactly the same thing to run the headless acceptance sweep. Two implementations
of the same arithmetic drift, and when they drift the sweep quietly starts
measuring a different application from the one that ships — which is the class of
failure PHASE3_PLAN.md exists to document, so it should not be reintroduced by the
fix.

This runs `node frontend/catalog.js --dump` over the fixed scenario set declared in
that file — each entry carries its own deltas/month/duration, so the scenario list
lives in exactly one place — and compares every feature value against the Python
mirror.

    python scripts/phase3/check_catalog_parity.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slider_sensitivity import Catalog, load_stats  # noqa: E402
from structural import Structural  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CATALOG_JS = ROOT / "frontend" / "catalog.js"
STRUCTURAL_JS = ROOT / "frontend" / "structural.js"

# Float32 round-trips through the ONNX tensor anyway, so agreement to ~1e-9
# relative is far tighter than anything that could matter downstream.
TOLERANCE = 1e-9


def dump_js(script: Path = None) -> list[dict]:
    result = subprocess.run(
        ["node", str(script or CATALOG_JS), "--dump"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    return json.loads(result.stdout)


def close(a: float, b: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=1e-12)


def main() -> None:
    slider_stats, _ = load_stats()
    catalog = Catalog(slider_stats)

    failures = 0
    for entry in dump_js():
        name = entry["name"]
        deltas = dict(catalog.zero) | {
            k: float(v) for k, v in entry["deltas"].items()
        }
        py = catalog.build(deltas, entry["month"], entry["duration"])
        js = entry["catalog"]

        # The response-variable seeds are anchors runPipeline() overwrites, not
        # scenario reconstruction; the Python mirror sets them in
        # fill_response_features() instead. Compare only the driver features.
        shared = sorted(set(js) & set(py))
        js_only = sorted(set(js) - set(py) - {"month"})

        mismatched = [k for k in shared if not close(js[k], py[k])]
        if mismatched or not shared:
            failures += 1
            print(f"FAIL {name}: {len(mismatched)} of {len(shared)} features differ")
            for key in mismatched[:8]:
                print(f"       {key:48s} js={js[key]!r:<24} py={py[key]!r}")
        else:
            print(f"ok   {name:20s} {len(shared)} features agree")
        if js_only:
            print(f"       (js-only, not mirrored: {len(js_only)} anchors/seeds)")

    failures += check_structural()

    if failures:
        raise SystemExit(f"\n{failures} scenario(s) disagree between JS and Python.")
    print("\nJS and Python agree on every scenario, for both the feature catalog and")
    print("the structural layer.")


def check_structural() -> int:
    """Same guard for Layer 2 / Layer 3 (frontend/structural.js)."""
    structural = Structural()
    print("\nstructural layer (Layer 2 + Layer 3):")

    failures = 0
    for entry in dump_js(STRUCTURAL_JS):
        name = entry["name"]
        deltas = {k: float(v) for k, v in entry["deltas"].items()}
        py = structural.response(deltas, entry["month"], entry["duration"])
        js = entry["response"]

        mismatched = []
        for output, bucket in js.items():
            if not close(bucket["total"], py[output]["total"]):
                mismatched.append((output, bucket["total"], py[output]["total"]))
            for lever_id, detail in bucket["byLever"].items():
                mirror = py[output]["byLever"].get(lever_id)
                if mirror is None or not close(
                    detail["displacement"], mirror["displacement"]
                ):
                    mismatched.append((lever_id, detail["displacement"], mirror))

        # The declared sign in structural_params.json is documentation; this makes it
        # load-bearing, so a formula that contradicts its own stated physics fails here
        # rather than shipping.
        by_id = {l["id"]: l for l in structural.params["levers"]}
        for output, bucket in js.items():
            for lever_id, detail in bucket["byLever"].items():
                lever = by_id.get(lever_id)
                if lever is None:
                    # A transfer edge: its sign is relative to the source output's
                    # displacement, not to any one slider.
                    transfer = next(
                        t for t in structural.params["transfers"] if t["id"] == lever_id
                    )
                    source_total = js[transfer["from"]]["total"]
                    expected = transfer["sign"] * source_total
                else:
                    expected = lever["sign"] * deltas[detail["slider"]]
                if expected != 0 and detail["displacement"] != 0:
                    if (detail["displacement"] > 0) != (expected > 0):
                        mismatched.append(
                            (f"{lever_id} SIGN", detail["displacement"], expected)
                        )

        if mismatched:
            failures += 1
            print(f"  FAIL {name}: {len(mismatched)} disagreements")
            for key, a, b in mismatched[:6]:
                print(f"         {key:44s} js={a!r:<22} py/expected={b!r}")
        else:
            active = sum(len(b["byLever"]) for b in js.values())
            print(f"  ok   {name:20s} {active} lever contributions agree")
    return failures


if __name__ == "__main__":
    main()

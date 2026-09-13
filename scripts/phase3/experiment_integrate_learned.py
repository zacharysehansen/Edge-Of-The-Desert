"""
experiment_integrate_learned.py
-------------------------------
PHASE3_PLAN.md §13 revisited (§32): should Layer 3 integrate the LEARNED climate
residual over the scenario duration, the way it already integrates the structural
forcing?

§13 measured this once and said no: with human deltas still inside Layer 1,
integrating the learned residual multiplied D2's wrong-signed human coefficients
(Mead -> NDVI -3.27 -> -14.01) and sent precipitation -> streamflow past the top of
the scale (+103.9). Two things have changed since. Layer 1 is climate-only, so no
human coefficient is inside the term any more (§13 itself), and two of the four
residual models are now linear (§27, §29). §31 then showed what the current
one-step display costs: for a sustained scenario a fast-memory model's one-step
change is near zero with a sign set by the lag structure, so the streamflow card
shows "less flow" under a wet April.

THIS SCRIPT CHANGES NOTHING THAT SHIPS. It loads the shipped stack, flips
`integrate_learned_residual` IN MEMORY, and re-runs the checks that decide whether
the flip is an improvement. The closed form is §4b's, already in structural.js and
its mirror: z_n = (s/λ)(1 − (1−λ)^n), with each output's fitted λ.

THE CRITERIA, FIXED BEFORE THE FIRST RUN
----------------------------------------
The flip SHIPS only if ALL of these hold at the app's default 12-month duration:
  1. The climate-sign gate (sustained, gated) has NO MORE failing pairs than today's
     five, and rain -> streamflow — the pair §31 diagnosed as a display limit — PASSES.
  2. No climate slider's full swing moves any output by more than 100 score points
     (the whole card). §13's +103.9 is the failure this rules out.
  3. The human levers are untouched: the acceptance gate and the no-double-count
     gate still pass. (They should by construction — only the learned residual is
     integrated and it carries no human input — and that is asserted, not assumed.)
If any fails, the switch stays off and the run is recorded.

Run:
    python scripts/phase3/experiment_integrate_learned.py
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "phase3"))

from slider_sensitivity import (  # noqa: E402
    CLIMATE_SLIDERS,
    HUMAN_SLIDERS,
    MODELS,
    Runner,
    acceptance,
    climate_signs,
    no_double_count,
)

OUTPUT = ROOT / "model" / "experiment_integrate_learned.json"
MAX_POINTS = 100.0
MONTHS = 12


def sweep(runner: Runner, months: int) -> dict[str, dict[str, float]]:
    table = {}
    for slider in HUMAN_SLIDERS + CLIMATE_SLIDERS:
        pol = runner.slider_stats[slider]["policy"]
        lo, hi = dict(runner.default), dict(runner.default)
        lo[slider], hi[slider] = pol["min"], pol["max"]
        a, b = runner.one_step(lo, 7, months), runner.one_step(hi, 7, months)
        table[slider] = {k: runner.score(k, b[k]) - runner.score(k, a[k]) for k in MODELS}
    return table


def gate_counts(runner: Runner, months: int) -> tuple[int, dict]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        failures = climate_signs(runner, months, gated=True)
    rows = {}
    for line in buf.getvalue().splitlines():
        if "->" in line and ("PASS" in line or "FAIL" in line):
            parts = line.split()
            rows[f"{parts[0]} -> {parts[2]}"] = parts[-1]
    return failures, rows


def main() -> None:
    runner = Runner()
    print("=== integrate the learned climate residual over the scenario? ===\n")
    lam = {k: v["value"] for k, v in runner.structural.params["reversion_per_month"].items()}
    print("  λ per month:", {k: round(v, 4) for k, v in lam.items()})
    print("  integration multiplier at 12 months:", {k: round((1 - (1 - v) ** 12) / v, 2) for k, v in lam.items()})

    before = {"sweep": sweep(runner, MONTHS)}
    before["sign_failures"], before["sign_rows"] = gate_counts(runner, MONTHS)

    runner.structural.params["integrate_learned_residual"]["value"] = True
    after = {"sweep": sweep(runner, MONTHS)}
    after["sign_failures"], after["sign_rows"] = gate_counts(runner, MONTHS)
    with redirect_stdout(io.StringIO()):
        acc = acceptance(runner, MONTHS)
        ndc = no_double_count(runner, MONTHS)

    print(f"\n  climate sweep at {MONTHS} months (score points, min -> max), off -> ON:")
    print(f"  {'slider':24s}" + "".join(f"{k[:11]:>22s}" for k in MODELS))
    for slider in CLIMATE_SLIDERS:
        print(f"  {slider[:24]:24s}" + "".join(
            f"{before['sweep'][slider][k]:+9.2f} -> {after['sweep'][slider][k]:+8.2f}" for k in MODELS))
    print(f"\n  human sweep unchanged: {all(abs(before['sweep'][s][k] - after['sweep'][s][k]) < 1e-9 for s in HUMAN_SLIDERS for k in MODELS)}")

    print("\n  climate-sign gate (sustained 12 mo), off -> ON:")
    changed = {k: (before["sign_rows"].get(k), after["sign_rows"].get(k)) for k in after["sign_rows"] if before["sign_rows"].get(k) != after["sign_rows"].get(k)}
    for k, (a, b) in changed.items():
        print(f"    {k:44s} {a} -> {b}")
    print(f"    failing pairs {before['sign_failures']} -> {after['sign_failures']}")

    max_pts = max(abs(v) for s in CLIMATE_SLIDERS for v in after["sweep"][s].values())
    rain_sw = after["sign_rows"].get("precipitation_mm_day -> surface_water")
    checks = [
        ("no new sign failures", after["sign_failures"] <= before["sign_failures"], f"{after['sign_failures']} vs {before['sign_failures']}"),
        ("rain -> streamflow passes", rain_sw == "PASS", str(rain_sw)),
        (f"largest climate swing <= {MAX_POINTS:.0f} pts", max_pts <= MAX_POINTS, f"{max_pts:.1f}"),
        ("acceptance gate passes", acc == 0, f"{acc} failures"),
        ("no-double-count passes", ndc == 0, f"{ndc} deviations"),
    ]
    print("\n=== verdict, against the criteria fixed in the docstring ===")
    for label, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label:36s} {detail}")
    ships = all(ok for _, ok, _ in checks)
    print(f"\n  verdict: {'SHIPS — flip the switch in structural_params.py' if ships else 'DOES NOT SHIP — the switch stays off'}")

    OUTPUT.write_text(json.dumps({
        "criteria": {"max_points": MAX_POINTS, "months": MONTHS},
        "lambda": lam,
        "before": before, "after": after,
        "acceptance_failures_on": acc, "no_double_count_deviations_on": ndc,
        "checks": [{"label": l, "ok": bool(ok), "detail": d} for l, ok, d in checks],
        "verdict": "SHIPS" if ships else "DOES NOT SHIP",
    }, indent=2))
    print(f"\n  wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

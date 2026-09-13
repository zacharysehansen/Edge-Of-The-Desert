"""Generate the frontend's "responds mainly to" strings from the exported models.

PHASE3_PLAN.md D4: `TOP_INPUTS` was a hand-written table in `state.js`, and it had
drifted into telling users the opposite of the truth. It still said wildfire
"responds mainly to: population, urbanization (lag), temperature (lag)" after the
MTBS target fix removed every human feature from that model, and it still credited
NDVI to public supply groundwater, which NDVI has never carried since it went
`extended`. The app was advertising human drivers for outputs that cannot see them.

The fix is to stop writing the strings by hand. This reads `model/*_feature_
importance.json` — the artifacts Phase 2 exports — and writes
`frontend/top_inputs.json`, so the claim on each output card is derived from the
model that is actually deployed and cannot drift from it again.

It also records, per model, how many of its features are human levers at all. That
number is the D1 measurement (four of six models have zero), and putting it on the
card is the difference between an interface that overstates its reach and one that
says what it can and cannot answer.

Written to its own file rather than into `computed_stats.json` so that
`generate_stats.py` stays the sole writer of that one.

    python scripts/phase3/top_inputs.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model"
OUTPUT_FILE = ROOT / "frontend" / "top_inputs.json"

# UI key -> exported filename stem. Mirrors MODEL_FILENAMES in models.js.
MODELS = {
    "grace": "grace",
    "ndvi": "ndvi",
    "groundwater": "groundwater",
    "surface_water": "surface_water",
    "wildfire": "wildfire_monthly",
    "wildlife": "wildlife",
}

HOW_MANY_TO_SHOW = 3

# Base variable -> (display label, family). The nClimDiv series are a second
# encoding of the same physical quantity as the MERRA-2 ones, so they share a
# label; that is also why one slider drives both in models.js.
BASES = {
    "population": ("population", "human"),
    "irrigation_total_withdrawal_mgd": ("irrigation withdrawal", "human"),
    "public_supply_groundwater_mgd": ("public supply groundwater", "human"),
    "impervious_pct": ("urbanization", "human"),
    "mead_pool_elevation": ("Lake Mead level", "human"),
    # Human, but held fixed in the frontend — there is no slider for releases.
    "mead_total_release": ("Lake Mead releases", "human-fixed"),
    "precip_x_impervious": ("precipitation x urbanization", "human"),
    "precipitation_mm_day": ("precipitation", "climate"),
    # GRACE's storage-change input (PHASE3_PLAN.md §26). Driven from the rain and
    # temperature sliders in the frontend; it is climate, not a lever of its own.
    "gldas_tws_proxy_delta": ("land-surface storage change (GLDAS)", "climate"),
    "nclimdiv_precipitation_mm_day": ("precipitation", "climate"),
    "temperature_2m_c": ("temperature", "climate"),
    "nclimdiv_temperature_c": ("temperature", "climate"),
    "usdm_dsci": ("drought (USDM)", "climate"),
    "nclimdiv_pdsi": ("drought (PDSI)", "climate"),
    "precip_x_temperature": ("precipitation x temperature", "climate"),
    "nclimdiv_precip_x_temperature": ("precipitation x temperature", "climate"),
    "nclimdiv_log_precip_annual": ("precipitation", "climate"),
    "month_sin": ("season", "season"),
    "month_cos": ("season", "season"),
    "year_linear": ("year", "trend"),
    "grace_groundwater_anomaly": ("GRACE storage", "response"),
    "grace_available": ("GRACE coverage", "response"),
    "ndvi": ("NDVI", "response"),
    "depth_to_water_anomaly_ft": ("well depth", "response"),
    "discharge_log_anomaly": ("streamflow", "response"),
    "bbs_abundance_anomaly": ("bird abundance", "response"),
}

# Suffixes features.py / merge.py append, innermost last.
SUFFIX = re.compile(
    r"_(lag\d+|roll\d+|anomaly|annual_mean|annual_sum|jja_mean|june|year_end)$"
)


def split_feature(name: str) -> tuple[str, list[str]]:
    """Peel the engineered suffixes off a feature name, outermost first.

    Stops as soon as what is left is a known base, because several bases end in a
    suffix-shaped token themselves — `grace_groundwater_anomaly` and
    `discharge_log_anomaly` are variable names, not engineered anomalies.
    """
    qualifiers: list[str] = []
    while name not in BASES and (match := SUFFIX.search(name)) is not None:
        qualifiers.append(match.group(1))
        name = name[: match.start()]
    return name, list(reversed(qualifiers))


def describe_qualifier(qualifier: str) -> str:
    if qualifier.startswith("lag"):
        return f"{qualifier[3:]}-mo lag"
    if qualifier.startswith("roll"):
        return f"{qualifier[4:]}-mo mean"
    return {
        "anomaly": "anomaly",
        "annual_mean": "annual mean",
        "annual_sum": "annual total",
        "jja_mean": "summer",
        "june": "June",
        "year_end": "year end",
    }[qualifier]


def describe(name: str) -> tuple[str, str, str]:
    """Feature name -> (base label, full label, family)."""
    base, qualifiers = split_feature(name)
    if base not in BASES:
        raise KeyError(
            f"No label for base variable '{base}' (from feature '{name}'). "
            f"Add it to BASES in {Path(__file__).name}."
        )
    label, family = BASES[base]
    if not qualifiers:
        return label, label, family
    detail = ", ".join(describe_qualifier(q) for q in qualifiers)
    return label, f"{label} ({detail})", family


def summarize(stem: str) -> dict[str, object]:
    importance = json.loads(
        (MODEL_DIR / f"{stem}_feature_importance.json").read_text()
    )
    names = json.loads((MODEL_DIR / f"{stem}_feature_names.json").read_text())

    # Rank by BASE variable, not by raw feature. A top-3 of "temperature",
    # "temperature (1-mo lag)" and "temperature (anomaly)" is one driver listed
    # three times, which tells a reader nothing.
    by_base: dict[str, float] = defaultdict(float)
    family_of: dict[str, str] = {}
    for feature, weight in importance.items():
        label, _, family = describe(feature)
        by_base[label] += float(weight)
        family_of[label] = family

    ranked = sorted(by_base.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(by_base.values()) or 1.0
    top = [
        {
            "label": label,
            "family": family_of[label],
            "share": round(weight / total, 4),
        }
        for label, weight in ranked[:HOW_MANY_TO_SHOW]
    ]

    families = [describe(name)[2] for name in names]
    human = [n for n, f in zip(names, families) if f == "human"]

    return {
        "summary": ", ".join(entry["label"] for entry in top),
        "top": top,
        "feature_count": len(names),
        # The D1 measurement, carried onto the card so the interface cannot
        # advertise a reach it does not have.
        "human_feature_count": len(human),
        "human_features": sorted(human),
    }


def main() -> None:
    result = {key: summarize(stem) for key, stem in MODELS.items()}

    with OUTPUT_FILE.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"Top inputs written to {OUTPUT_FILE}\n")
    for key, entry in result.items():
        human = entry["human_feature_count"]
        note = f"{human} human" if human else "NO human features"
        print(f"  {key:14s} {entry['summary']:<52s} ({note} of {entry['feature_count']})")


if __name__ == "__main__":
    main()

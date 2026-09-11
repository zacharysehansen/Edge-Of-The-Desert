"""Estimate the Layer 2 transfer coefficients between outputs.

PHASE3_PLAN.md §4's Layer 2 table has no row for surface water, wildfire or wildlife,
and §10 confirmed the learned layer cannot reach them either — so as built, three of
six outputs do not respond to a human lever at all. The fix is not more direct levers
but TRANSFER edges: the outputs form a chain, and `groundwater` and `ndvi` already
carry working structural levers, so one coefficient per edge propagates all of them.

    lever → pumping        → stream capture   → surface water
    lever → NDVI           → forage/cover     → wildlife
    lever → surface water  → riparian habitat → wildlife

The first is water balance and is handled in structural_params.py. The other two are
ecological transfers with no closed-form arithmetic, so they are estimated here from
the project's own annual panel, controlling for climate and trend so the coefficient
is not just the climate-mediated correlation between two climate-driven series.

    d(bbs_abundance_anomaly) = a + b·X + climate + trend + e

n is small — the overlap of BBS with MODIS NDVI is about two decades — so these are
reported with their t-statistics and shipped with wide bands. The alternative is
assuming a number, which is worse.

    python scripts/phase3/transfer_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "annual_panel.csv"
OUTPUT_FILE = ROOT / "model" / "transfer_calibration.json"

TARGET = "bbs_abundance_anomaly"

# Kept deliberately short. With ~20 usable years,each extra control costs a degree of
# freedom that the estimate cannot spare; PDSI is the single climate variable the
# wildlife model itself leans on hardest.
CONTROLS = ["nclimdiv_pdsi_annual_mean"]


def load_annual() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH)
    if "year" not in panel.columns:
        panel = panel.rename(columns={panel.columns[0]: "year"})
    panel = panel.set_index("year")

    # merge.py never aggregates surface water to annual grain — discharge is a
    # monthly-only column — so the riparian edge has to build its own predictor.
    monthly = pd.read_csv(ROOT / "data" / "Final" / "water_surface_monthly.csv")
    year = pd.PeriodIndex(monthly["year_month"], freq="M").year
    annual_discharge = monthly.groupby(year)["discharge_log_anomaly"].mean()
    annual_discharge.index.name = "year"
    panel["discharge_log_anomaly_annual_mean"] = annual_discharge
    return panel


def fit_transfer(
    panel: pd.DataFrame, predictor: str, *, trend: bool
) -> dict | None:
    columns = [predictor, *CONTROLS, TARGET]
    missing = [c for c in columns if c not in panel.columns]
    if missing:
        return {"error": f"missing columns: {missing}"}

    frame = panel[columns].dropna()
    if len(frame) < 12:  # noqa: PLR2004
        return {"error": f"only {len(frame)} usable years"}

    design = frame[[predictor, *CONTROLS]].copy()
    if trend:
        design["trend"] = frame.index - frame.index.min()

    model = sm.OLS(frame[TARGET], sm.add_constant(design)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 2}
    )
    return {
        "predictor": predictor,
        "n": int(model.nobs),
        "years": f"{int(frame.index.min())}-{int(frame.index.max())}",
        "beta": float(model.params[predictor]),
        "se": float(model.bse[predictor]),
        "t": float(model.tvalues[predictor]),
        "p": float(model.pvalues[predictor]),
        "r2": float(model.rsquared),
        "predictor_sd": float(frame[predictor].std()),
        "target_sd": float(frame[TARGET].std()),
    }


def main() -> None:
    panel = load_annual()

    edges = [
        ("ndvi_to_wildlife", "ndvi_annual_mean", +1),
        ("surface_water_to_wildlife", "discharge_log_anomaly_annual_mean", +1),
    ]

    print("=" * 78)
    print("TRANSFER COEFFICIENTS  (PHASE3_PLAN.md §14)")
    print("=" * 78)
    print(f"\nTarget: {TARGET}   controls: {CONTROLS}   HAC(2) standard errors\n")

    results = {}
    for name, predictor, expected in edges:
        results[name] = {"expected_sign": expected}
        print(f"{name}   ({predictor})")
        print(f"  {'specification':22s}{'beta':>12s}{'t':>8s}{'p':>8s}{'n':>5s}{'R2':>7s}")
        for label, trend in [("+ climate", False), ("+ climate + trend", True)]:
            entry = fit_transfer(panel, predictor, trend=trend)
            results[name][label] = entry
            if entry is None or "error" in entry:
                print(f"  {label:22s}{entry['error'] if entry else 'no fit':>40s}")
                continue
            print(
                f"  {label:22s}{entry['beta']:>12.4f}{entry['t']:>8.2f}"
                f"{entry['p']:>8.3f}{entry['n']:>5d}{entry['r2']:>7.3f}"
            )
        entry = results[name].get("+ climate + trend")
        if entry and "error" not in entry:
            # Effect of a one-sd move in the predictor, in target sd — the scale that
            # says whether this edge is worth shipping at all.
            in_sd = entry["beta"] * entry["predictor_sd"] / entry["target_sd"]
            results[name]["effect_in_sd_per_predictor_sd"] = in_sd
            agrees = np.sign(entry["beta"]) == expected
            print(
                f"  -> 1 sd of {predictor.split('_annual')[0]} moves wildlife "
                f"{in_sd:+.3f} sd   sign {'agrees' if agrees else 'DISAGREES'} with expectation"
            )
            print(f"     years {entry['years']}")
        print()

    with OUTPUT_FILE.open("w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

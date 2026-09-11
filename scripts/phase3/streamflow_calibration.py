"""
streamflow_calibration.py
-------------------------
Try to corroborate the four UNTESTED constants on the streamflow card.

PHASE3_PLAN.md §16 item 3. `aquifer_calibration.py` established the pattern — regress
the lever against the project's own panel with climate and trend controls and
Newey-West errors — and it is what earned irrigation its `corroborated` badge. These
four are the widest remaining bands in Layer 2:

    stream_capture_fraction        0.10   band 0.05-0.25   four negative streamflow paths
    effluent_return_fraction       0.55   band 0.45-0.70   population -> streamflow
    runoff_coefficient_impervious  0.85   band 0.75-0.95   urbanization -> streamflow
    runoff_coefficient_natural     0.15   band 0.05-0.25   (only their DIFFERENCE matters)

WHAT IS BEING INVERTED
----------------------
Layer 2 converts an added flow into the target's units as
`ln(1 + Δcfs / regional_baseline_cfs)`, which for small changes is
`Δcfs / regional_baseline_cfs`. So a regression coefficient on a physically-scaled
regressor inverts straight back to the constant:

    capture      X = pumping anomaly, AF/month
                 β = −capture × cfs_per_af_month / baseline_cfs
    effluent     X = population anomaly
                 β = +return × (gpcd/1e6) × cfs_per_mgd / baseline_cfs
    runoff       X = (impervious/100) × precip_depth × region_area × CFS_PER_CMS, in cfs
                 β = +(c_impervious − c_natural) / baseline_cfs

Only the runoff *contrast* is identifiable, never the two coefficients separately —
they enter the physics solely as a difference. That is a property of the model, not a
limitation of the data, and is why this script reports one number for the pair.

WHAT THIS SCRIPT EXPECTS TO FAIL, AND WHY IT RUNS ANYWAY
--------------------------------------------------------
Decomposing each candidate regressor's variance on the panel, before fitting anything:

    irrigation withdrawal      5.4% of variance survives deseasonalising
    population                99.8%   -- i.e. a pure trend, no seasonal cycle at all
    impervious cover          99.9%   -- likewise
    precipitation             58.3%

A regressor that is a pure trend cannot be separated from a trend control, and the
trend control is not optional here: the gage network itself moved from 73 to 110
gages over the record, so anything identified off slow secular change is identified
off a changing denominator as much as off the lever. **Population and impervious
cover are therefore expected to return nulls, and a null is the useful answer** — it
says these bands cannot be narrowed from this panel at any effort, and need either a
different design or different data. Recording that is worth more than not looking.

Run:
    python scripts/phase3/streamflow_calibration.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.csv"
PARAMS_PATH = ROOT / "frontend" / "structural_params.json"
OUTPUT_FILE = ROOT / "model" / "streamflow_calibration.json"

TARGET = "discharge_log_anomaly"

# Same distributed lag as aquifer_calibration: an effect on a gage is not
# instantaneous, and a sustained change shows up as the SUM of these.
LAGS = (0, 1, 2, 3)

CONTROLS = ["nclimdiv_pdsi", "precipitation_mm_day", "temperature_2m_c"]
CONTROL_LAGS = (0, 1, 3, 6)

SQ_M_PER_ACRE = 4046.8564224
CFS_PER_CMS = 35.3147
SECONDS_PER_DAY = 86400.0
MM_PER_M = 1000.0

# Below this share of non-seasonal variance a regressor is a trend, and a
# trend-controlled fit cannot identify it. Declared before the fits are run.
TREND_LIKE_THRESHOLD = 0.90

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("streamflow_calibration")


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH)
    period = pd.PeriodIndex(panel["year_month"], freq="M")
    return panel.assign(month=period.month).set_index(period)


def deseasonalize(series: pd.Series, month: pd.Series) -> pd.Series:
    return series - series.groupby(month).transform("mean")


def nonseasonal_share(series: pd.Series, month: pd.Series) -> float:
    """Share of variance surviving removal of the month-of-year mean.

    Near 1.0 means the series has no seasonal cycle — for these inputs that means it
    is a secular trend, which a trend control will absorb entirely.
    """
    total = float(series.var())
    if not total:
        return float("nan")
    return float(deseasonalize(series, month).var() / total)


def fit(
    panel: pd.DataFrame,
    regressor: pd.Series,
    *,
    trend: bool,
    control_gages: bool,
) -> tuple[float, float, float, int]:
    """Summed distributed-lag coefficient, its Newey-West SE, R², n."""
    frame = pd.DataFrame(index=panel.index)
    for lag in LAGS:
        frame[f"lever_lag{lag}"] = regressor.shift(lag)
    for control in CONTROLS:
        centred = deseasonalize(panel[control], panel["month"])
        for lag in CONTROL_LAGS:
            frame[f"{control}_lag{lag}"] = centred.shift(lag)
    if trend:
        frame["time_trend"] = np.arange(len(panel), dtype=float) / 12.0
    if control_gages:
        # The target is a mean over gages and the network grew from 73 to 110. Any
        # effect identified off slow secular change is otherwise partly identified off
        # the denominator changing underneath it.
        frame["n_gages"] = panel["n_gages"].astype(float)
    angle = 2 * np.pi * (panel["month"] - 1) / 12
    frame["month_sin"], frame["month_cos"] = np.sin(angle), np.cos(angle)

    target = panel[TARGET]
    mask = frame.notna().all(axis=1) & target.notna()
    design, y = frame[mask], target[mask]
    model = sm.OLS(y, sm.add_constant(design)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}
    )
    terms = [c for c in design.columns if c.startswith("lever_lag")]
    beta = float(model.params[terms].sum())
    weights = np.array([1.0 if c in terms else 0.0 for c in model.params.index])
    variance = float(weights @ model.cov_params().to_numpy() @ weights)
    se = float(np.sqrt(variance)) if variance > 0 else float("nan")
    return beta, se, float(model.rsquared), int(model.nobs)


def main() -> None:
    panel = load_panel()
    constants = {
        k: v["value"] for k, v in json.loads(PARAMS_PATH.read_text())["constants"].items()
    }
    baseline_cfs = constants["regional_baseline_cfs"]

    # ── the three physically-scaled regressors ───────────────────────────────
    pumping_af = (
        deseasonalize(panel["irrigation_total_withdrawal_mgd"], panel["month"])
        * constants["af_per_mgd_month"]
    )
    population = deseasonalize(panel["population"], panel["month"])
    # Runoff enters as an interaction: impervious AREA times the depth of rain that
    # falls on it. Precipitation is NOT deseasonalized here — the physical quantity is
    # the actual rain, and the climate controls carry the deseasonalized version.
    runoff_cfs_per_unit_contrast = (
        (panel["impervious_pct"] / 100.0)
        * (panel["precipitation_mm_day"] / MM_PER_M / SECONDS_PER_DAY)
        * (constants["region_acres"] * SQ_M_PER_ACRE)
        * CFS_PER_CMS
    )

    specs = [
        ("stream_capture_fraction", pumping_af,
         lambda b: -b * baseline_cfs / constants["cfs_per_af_month"],
         "irrigation_total_withdrawal_mgd", -1),
        ("effluent_return_fraction", population,
         lambda b: b * baseline_cfs / (constants["gpcd_groundwater"] / 1e6 * constants["cfs_per_mgd"]),
         "population", +1),
        ("runoff_contrast", deseasonalize(runoff_cfs_per_unit_contrast, panel["month"]),
         lambda b: b * baseline_cfs,
         "impervious_pct", +1),
    ]

    print("=" * 78)
    print("STREAMFLOW CALIBRATION  (PHASE3_PLAN.md §16 item 3)")
    print("=" * 78)
    print(f"\nTarget: {TARGET}   |   distributed lag {LAGS}")
    print(f"Controls: {CONTROLS} at lags {CONTROL_LAGS}, + trend, + n_gages")
    print("Standard errors: Newey-West, 6 lags\n")

    print("Identifiability, decided before any fit (share of variance surviving")
    print("deseasonalising; above %.0f%% the series is a trend and a trend control absorbs it):"
          % (TREND_LIKE_THRESHOLD * 100))
    shares = {}
    for _, _, _, column, _ in specs:
        share = nonseasonal_share(panel[column].dropna(), panel.loc[panel[column].notna(), "month"])
        shares[column] = share
        note = "TREND-LIKE — expect a null" if share > TREND_LIKE_THRESHOLD else "has usable anomaly variance"
        print(f"  {column:34s} {share * 100:5.1f}%   {note}")
    print(f"\n  gage network over the record: {panel['n_gages'].min():.0f} -> {panel['n_gages'].max():.0f} gages")

    results = {}
    for name, regressor, invert, column, expected_sign in specs:
        print(f"\n{name}")
        print(f"  {'specification':26s}{'beta':>12s}{'t':>8s}{'implied':>12s}{'R2':>7s}")
        entry = {"regressor": column, "expected_sign": expected_sign,
                 "nonseasonal_share": shares[column], "specifications": {}}
        for label, kwargs in [
            ("+ climate", dict(trend=False, control_gages=False)),
            ("+ climate + trend", dict(trend=True, control_gages=False)),
            ("+ climate + trend + gages", dict(trend=True, control_gages=True)),
        ]:
            beta, se, r2, n = fit(panel, regressor, **kwargs)
            implied = invert(beta)
            t_stat = beta / se if se and np.isfinite(se) else float("nan")
            entry["specifications"][label] = {
                "beta": beta, "se": se, "t": t_stat, "implied": implied, "r2": r2, "n": n
            }
            print(f"  {label:26s}{beta:>12.3e}{t_stat:>8.2f}{implied:>12.4f}{r2:>7.3f}")
        results[name] = entry

    verdict(results, constants)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWritten to {OUTPUT_FILE.relative_to(ROOT)}")


def verdict(results: dict, constants: dict) -> None:
    print("\n" + "-" * 78)
    print("VERDICT — preferred specification is the fully controlled one")
    print("-" * 78)
    shipped = {
        "stream_capture_fraction": constants["stream_capture_fraction"],
        "effluent_return_fraction": constants["effluent_return_fraction"],
        "runoff_contrast": constants["runoff_coefficient_impervious"]
                           - constants["runoff_coefficient_natural"],
    }
    for name, entry in results.items():
        best = entry["specifications"]["+ climate + trend + gages"]
        t_stat, implied = best["t"], best["implied"]
        sign_ok = np.sign(implied) == entry["expected_sign"] if implied else False
        plausible = 0.0 < implied < 1.0
        strong = abs(t_stat) >= 2.0  # noqa: PLR2004
        if entry["nonseasonal_share"] > TREND_LIKE_THRESHOLD:
            call = "NULL — regressor is a trend, not identifiable from this panel"
        elif strong and sign_ok and plausible:
            call = f"IDENTIFIED — implied {implied:.4f} vs shipped {shipped[name]:.4f}"
        elif strong and not (sign_ok and plausible):
            call = f"REJECT — significant but implies {implied:.4f}, outside [0,1] or wrong sign"
        else:
            call = f"NULL — |t| = {abs(t_stat):.2f}, below 2.0"
        entry["verdict"] = call
        entry["shipped"] = shipped[name]
        print(f"  {name:28s} {call}")
    print(
        "\nOnly the runoff CONTRAST is ever identifiable, never the two coefficients\n"
        "separately: they enter the physics only as a difference."
    )


if __name__ == "__main__":
    main()

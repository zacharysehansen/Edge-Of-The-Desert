"""Reconcile the two estimates of the aquifer storage coefficient S_y * A.

This is the first Layer 2 task (PHASE3_PLAN.md §11.2). Every groundwater lever in
Layer 2 runs through one number:

    ΔDepth (ft) = ΔPumping (AF) / (S_y * A_eff)

and there are now two estimates of that denominator that disagree by more than an
order of magnitude:

  PHYSICAL   S_y * A ~ 1.87 M AF/ft.  PHASE3_PARAMS.md §2: S_y = 0.15 (ADWR basin-fill
             modelling range midpoint) times 45% of the eight-county area as alluvial
             basin. The document flags A_eff as "the least defensible number in the
             whole plan".

  EMPIRICAL  S_y * A ~ 120 k AF/ft, from the XGBoost partial-dependence slope measured
             in the §5 cross-check. That slope absorbs everything correlated with a
             pumping anomaly -- drought above all -- so it is a LOWER bound on the
             denominator and an upper bound on the lever.

This script produces the third, cleanest estimate: a distributed-lag regression of the
monthly change in well depth on the DESEASONALIZED irrigation anomaly, with climate
controls, so the coefficient is identified off excess pumping rather than off the
calendar or off drought.

    ΔDepth_t = a + Σ_j b_j · Q_anom_{t-j} + climate + season + e

Under sustained excess pumping Q, the per-month depth response is Σ_j b_j, so

    S_y * A = 1 / Σ_j b_j        (AF per foot, when Q is in AF/month)

The same regression is run for Lake Mead elevation, the other lever §10 corroborated,
because the DCP tier chain terminates in the same storage balance.

    python scripts/phase3/aquifer_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
PANEL_PATH = ROOT / "data" / "processed" / "monthly_panel.csv"
OUTPUT_FILE = ROOT / "model" / "aquifer_calibration.json"

TARGET = "depth_to_water_anomaly_ft"

# The other target §10 flagged as corroborated. GRACE is an equivalent-water-height
# anomaly over a much larger footprint than the monitored well network, so it yields
# no comparable storage coefficient (PHASE3_PARAMS.md §2 dead end 2 measured the two
# disagreeing outright, r = -0.752 with the wrong sign). It is checked here only for
# the SIGN, to see whether §10's irrigation -> GRACE finding survives the same
# controls that dissolved the Mead one.
SECOND_TARGET = "grace_groundwater_anomaly"

# 1 Mgal = 3.06889 acre-ft, so 1 MGD sustained for a year is 1,120 AF/yr.
AF_PER_MGD_YEAR = 1120.0
AF_PER_MGD_MONTH = AF_PER_MGD_YEAR / 12.0

# Distributed lag. Pumping does not reach a monitoring well the same month, and a
# sustained change shows up as the SUM of these, which is the quantity the storage
# balance predicts.
LAGS = (0, 1, 2, 3)

# Everything the pumping anomaly could otherwise be standing in for. PDSI and
# precipitation drive recharge directly; temperature drives demand, which is what
# makes pumping and drought collinear in the first place.
CONTROLS = [
    "nclimdiv_pdsi",
    "precipitation_mm_day",
    "temperature_2m_c",
]
CONTROL_LAGS = (0, 1, 3, 6)

# PHASE3_PARAMS.md §2 / §4a, for comparison.
PHYSICAL_SY_A = 0.15 * 12_500_000.0        # AF/ft
# The deseasonalized 2020 baseline the policy slider scales, from computed_stats.json.
IRRIGATION_BASELINE_MGD = 2744.5
# depth_to_water_anomaly_ft p5..p95 = -8.872..10.361, i.e. 100 score points.
DEPTH_RANGE_FT = 19.233
REGION_ACRES = 27_779_840.0
IRRIGATED_ACRES = 681_143.0
SPECIFIC_YIELD = 0.15


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL_PATH)
    period = pd.PeriodIndex(panel["year_month"], freq="M")
    return panel.assign(month=period.month, year=period.year).set_index(period)


def deseasonalize(series: pd.Series, month: pd.Series) -> pd.Series:
    """Departure from the month-of-year mean — the excess, not the calendar."""
    return series - series.groupby(month).transform("mean")


def build_design(
    panel: pd.DataFrame,
    lever: str,
    to_af_per_month: float,
    *,
    controls: bool = True,
    trend: bool = False,
    target_column: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Design matrix for one lever.

    `trend` adds a linear time index. It matters more than it looks: a lever with
    little within-year variance is mostly a proxy for the calendar year, and
    TimeSeriesSplit folds all share the same secular trend, so five folds "agreeing"
    on its sign is one piece of evidence, not five. Mead elevation is 7.7%
    within-year (PHASE3_PLAN.md D5); irrigation is 99.3%.
    """
    frame = pd.DataFrame(index=panel.index)
    target = panel[target_column].diff()

    anomaly = deseasonalize(panel[lever], panel["month"]) * to_af_per_month
    for lag in LAGS:
        frame[f"lever_lag{lag}"] = anomaly.shift(lag)

    if controls:
        for control in CONTROLS:
            centred = deseasonalize(panel[control], panel["month"])
            for lag in CONTROL_LAGS:
                frame[f"{control}_lag{lag}"] = centred.shift(lag)

    if trend:
        frame["time_trend"] = np.arange(len(panel), dtype=float) / 12.0

    angle = 2 * np.pi * (panel["month"] - 1) / 12
    frame["month_sin"] = np.sin(angle)
    frame["month_cos"] = np.cos(angle)

    mask = frame.notna().all(axis=1) & target.notna()
    return frame[mask], target[mask]


def fit(design: pd.DataFrame, target: pd.Series) -> sm.regression.linear_model.RegressionResults:
    # Newey-West: the residuals of a monthly water-level series are autocorrelated,
    # and an OLS standard error would overstate the significance of everything here.
    return sm.OLS(target, sm.add_constant(design)).fit(
        cov_type="HAC", cov_kwds={"maxlags": 6}
    )


def summarize(name: str, model, design: pd.DataFrame) -> dict:
    lever_terms = [c for c in design.columns if c.startswith("lever_lag")]
    beta = float(model.params[lever_terms].sum())

    # Variance of a sum of correlated coefficients, so the lag structure does not
    # get to look more precise than it is.
    weights = np.array([1.0 if c in lever_terms else 0.0 for c in model.params.index])
    variance = float(weights @ model.cov_params().to_numpy() @ weights)
    se = float(np.sqrt(variance)) if variance > 0 else float("nan")
    t_stat = beta / se if se and np.isfinite(se) else float("nan")

    storage = 1.0 / beta if beta != 0 else float("inf")
    return {
        "lever": name,
        "n": int(model.nobs),
        "beta_ft_per_af_month": beta,
        "beta_se": se,
        "t": t_stat,
        "storage_af_per_ft": storage,
        "implied_area_acres": storage / SPECIFIC_YIELD if np.isfinite(storage) else None,
        "r2": float(model.rsquared),
        "per_lag": {c: float(model.params[c]) for c in lever_terms},
    }


def main() -> None:
    panel = load_panel()

    levers = [
        ("irrigation", "irrigation_total_withdrawal_mgd", AF_PER_MGD_MONTH, "AF/month"),
        ("public_supply", "public_supply_groundwater_mgd", AF_PER_MGD_MONTH, "AF/month"),
        # Mead is an elevation, not a volume: ft of depth per ft of elevation, and
        # the DCP tier table (PHASE3_PARAMS.md §1) supplies the volume conversion.
        ("mead_elevation", "mead_pool_elevation", 1.0, "ft elevation"),
    ]
    specifications = [
        ("raw", dict(controls=False, trend=False)),
        ("+ climate", dict(controls=True, trend=False)),
        ("+ climate + trend", dict(controls=True, trend=True)),
    ]

    print("=" * 78)
    print("AQUIFER STORAGE CALIBRATION  (PHASE3_PLAN.md §11.2)")
    print("=" * 78)
    print(
        f"\nTarget: monthly change in {TARGET}\n"
        f"Lever:  deseasonalized anomaly, distributed lag {LAGS}\n"
        f"Climate controls: {CONTROLS} at lags {CONTROL_LAGS}\n"
        f"Standard errors: Newey-West, 6 lags\n"
    )

    results: dict[str, dict] = {}
    for name, column, scale, unit in levers:
        results[name] = {"unit": f"ft per {unit}"}
        print(f"{name}")
        print(f"  {'specification':22s}{'beta':>12s}{'t':>8s}{'S_y*A (AF/ft)':>16s}{'R2':>7s}")
        for label, kwargs in specifications:
            design, target = build_design(panel, column, scale, **kwargs)
            entry = summarize(name, fit(design, target), design)
            results[name][label] = entry
            storage = (
                f"{entry['storage_af_per_ft']:>16,.0f}"
                if entry["beta_ft_per_af_month"] > 0
                else f"{'wrong sign':>16s}"
            )
            print(
                f"  {label:22s}{entry['beta_ft_per_af_month']:>12.3e}"
                f"{entry['t']:>8.2f}{storage}{entry['r2']:>7.3f}"
            )
        print()

    # Does §10's irrigation -> GRACE corroboration survive the same controls?
    print(f"sign check: irrigation -> {SECOND_TARGET} (expect NEGATIVE, pumping depletes)")
    print(f"  {'specification':22s}{'beta':>12s}{'t':>8s}{'R2':>7s}")
    results["irrigation_grace"] = {"unit": "cm per AF/month", "expected_sign": -1}
    for label, kwargs in specifications:
        design, target = build_design(
            panel,
            "irrigation_total_withdrawal_mgd",
            AF_PER_MGD_MONTH,
            target_column=SECOND_TARGET,
            **kwargs,
        )
        entry = summarize("irrigation_grace", fit(design, target), design)
        results["irrigation_grace"][label] = entry
        print(
            f"  {label:22s}{entry['beta_ft_per_af_month']:>12.3e}"
            f"{entry['t']:>8.2f}{entry['r2']:>7.3f}"
        )
    print()

    print("-" * 78)
    print("RECONCILIATION — S_y * A_eff for the groundwater storage balance")
    print("-" * 78)
    preferred = results["irrigation"]["+ climate + trend"]
    print(f"  physical (PHASE3_PARAMS §2, 0.15 x 12.5M acres)  {PHYSICAL_SY_A:>12,.0f} AF/ft")
    print(f"  XGBoost partial dependence (PHASE3_PLAN §10)     {120_500:>12,.0f} AF/ft")
    for label, _ in specifications:
        entry = results["irrigation"][label]
        if entry["beta_ft_per_af_month"] > 0:
            print(
                f"  regression, irrigation, {label:20s} {entry['storage_af_per_ft']:>12,.0f}"
                f" AF/ft   (t = {entry['t']:+.2f})"
            )
    if preferred["beta_ft_per_af_month"] > 0:
        area = preferred["implied_area_acres"]
        print(
            f"\n  preferred: {preferred['storage_af_per_ft']:,.0f} AF/ft"
            f"  ->  A_eff = {area:,.0f} acres"
            f"  = {area / IRRIGATED_ACRES:.2f}x the irrigated area,"
            f" {area / REGION_ACRES * 100:.1f}% of the region"
        )

    # What the two coefficients imply for the lever the UI will actually render.
    # PHASE3_PARAMS.md §2 warned against tuning A_eff to make the §7 test pass; the
    # point of printing both is that neither can be chosen on convenience.
    print()
    print("-" * 78)
    print("WHAT EACH COEFFICIENT IMPLIES FOR THE SLIDER")
    print("-" * 78)
    swing_mgd = IRRIGATION_BASELINE_MGD * 0.60          # the policy slider's -60% end
    swing_af_month = swing_mgd * AF_PER_MGD_MONTH
    print(
        f"  a sustained -60% irrigation scenario = -{swing_mgd:,.0f} MGD"
        f" = -{swing_af_month:,.0f} AF/month"
    )
    for label, storage in [
        ("measured (short run)", preferred["storage_af_per_ft"]),
        ("physical (PHASE3_PARAMS §2)", PHYSICAL_SY_A),
    ]:
        per_month = swing_af_month / storage
        year = per_month * 12
        print(
            f"    {label:30s} {per_month:6.3f} ft/month"
            f"  = {year:7.2f} ft/yr  = {abs(year) / DEPTH_RANGE_FT * 100:6.1f} score points at 12 mo"
        )
    print(
        "\n  These are not competing estimates of one quantity. A drawdown cone spreads\n"
        "  with time, so the area it draws from grows: the measured value is the SHORT-RUN\n"
        "  limit (a monthly pumping anomaly, which is what the regression identifies) and\n"
        "  the physical value is the LONG-RUN limit (the whole alluvial basin). Their ratio\n"
        f"  is {PHYSICAL_SY_A / preferred['storage_af_per_ft']:.1f}x, which is exactly the ratio of the two areas.\n"
        "\n  Applying the short-run coefficient to a 12-month scenario is the same linear-ramp\n"
        "  error PHASE3_PLAN.md §4a already measured and rejected. Layer 2 must pair it with\n"
        "  the mean-reverting integration of §4b, not multiply it by the duration."
    )

    with OUTPUT_FILE.open("w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nWritten to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

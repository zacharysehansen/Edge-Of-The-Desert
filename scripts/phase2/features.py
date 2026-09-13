"""
features.py
-----------
Take the merged panels from merge.py and produce fully-engineered
(X, y) pairs, one per model.

Returns
-------
dict mapping model_id -> (X: pd.DataFrame, y: pd.Series)

  "ndvi"          - Model 1: NDVI (vegetation health), monthly
  "grace"         - Model 2: GRACE groundwater anomaly, monthly
  "groundwater"   - Model 3: Groundwater well levels, monthly
  "surface_water" - Model 4: Surface water discharge, monthly
  "wildfire"      - Model 5: Wildfire risk index, annual
  "wildlife"      - Model 6: Wildlife abundance index, annual

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.phase2.merge import (
    ANNUAL_WINDOW_END,
    ANNUAL_WINDOW_START,
    MONTHLY_CAPPED_2020_BASES,
    MONTHLY_WINDOW_END,
    MONTHLY_WINDOW_END_EXTENDED,
    MONTHLY_WINDOW_START,
    build_annual_panel,
    build_monthly_panel,
)

_MONTHLY_BASE_INPUTS = [
    "population",
    "irrigation_total_withdrawal_mgd",
    "public_supply_groundwater_mgd",
    "mead_pool_elevation",
    "mead_total_release",
    "usdm_dsci",
    "temperature_2m_c",
    "precipitation_mm_day",
    "impervious_pct",
]

# Columns that get lag + rolling treatment
# nClimDiv drought/temperature/precipitation — the only climate inputs that exist before
# 2000. Any monthly model reaching back past the satellite era is built from these.
_NCLIMDIV_INPUTS = [
    "nclimdiv_pdsi",
    "nclimdiv_temperature_c",
    "nclimdiv_precipitation_mm_day",
]

_LAG_ROLL_COLS = _MONTHLY_BASE_INPUTS + _NCLIMDIV_INPUTS + [
    "grace_groundwater_anomaly",
    "ndvi",
    "wildfire_risk_index",
    "fire_count",
    "log_acres",
    # Per-station anomaly indices — the actual targets for the groundwater and
    # surface-water models. The raw *_mean columns beside them are a mean over
    # whichever stations happened to report that month, so they move with the
    # roster as much as with the water. See scripts/phase1/groundwater_levels.py.
    "depth_to_water_anomaly_ft",
    "discharge_log_anomaly",
    "depth_to_water_ft_mean",
    "discharge_cfs_mean",
]


def _add_monthly_lag_roll(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features for all numeric input columns."""
    df = df.copy()
    for col in _LAG_ROLL_COLS:
        if col not in df.columns:
            continue
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag3"] = df[col].shift(3)
        df[f"{col}_lag6"] = df[col].shift(6)
        df[f"{col}_roll3"] = df[col].shift(1).rolling(3, min_periods=2).mean()
        df[f"{col}_roll6"] = df[col].shift(1).rolling(6, min_periods=4).mean()
        df[f"{col}_roll12"] = df[col].shift(1).rolling(12, min_periods=8).mean()
    return df


def _add_anomaly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standardised departure from 12-month trailing mean for temperature and precip."""
    df = df.copy()
    for col in (
        "temperature_2m_c",
        "precipitation_mm_day",
        "nclimdiv_temperature_c",
        "nclimdiv_precipitation_mm_day",
    ):
        if col not in df.columns:
            continue
        roll12_col = f"{col}_roll12"
        if roll12_col in df.columns:
            df[f"{col}_anomaly"] = df[col] - df[roll12_col]
            df[f"{col}_anomaly_lag1"] = df[f"{col}_anomaly"].shift(1)
            df[f"{col}_anomaly_roll3"] = (
                df[f"{col}_anomaly"].shift(1).rolling(3, min_periods=2).mean()
            )
    return df


def _add_seasonal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Sine/cosine month encoding — no look-ahead, deterministic."""
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physically motivated interaction terms."""
    df = df.copy()
    # Precipitation x impervious = runoff proxy (dominant discharge in urban desert)
    if "precipitation_mm_day" in df.columns and "impervious_pct" in df.columns:
        df["precip_x_impervious"] = df["precipitation_mm_day"] * df["impervious_pct"]
    # Precipitation x temperature = evapotranspiration proxy
    if "precipitation_mm_day" in df.columns and "temperature_2m_c" in df.columns:
        df["precip_x_temperature"] = df["precipitation_mm_day"] * df["temperature_2m_c"]
    # Same, on the long-record series, so pre-2000 rows have one too.
    if (
        "nclimdiv_precipitation_mm_day" in df.columns
        and "nclimdiv_temperature_c" in df.columns
    ):
        df["nclimdiv_precip_x_temperature"] = (
            df["nclimdiv_precipitation_mm_day"] * df["nclimdiv_temperature_c"]
        )
    return df


def _add_gldas_features(df: pd.DataFrame) -> pd.DataFrame:
    """The month-to-month change in GLDAS's land-surface storage (soil 0-200 cm +
    snow + canopy, mm). GRACE's target is a month-to-month change in total storage,
    and this is the part of it the land-surface model observes (PHASE3_PLAN.md §25:
    r = +0.62, slope +0.8 m/m). Contemporaneous with the target's month, exactly as
    precipitation and temperature are."""
    df = df.copy()
    if "gldas_tws_proxy_mm" in df.columns:
        df["gldas_tws_proxy_delta"] = df["gldas_tws_proxy_mm"].diff()
    return df


def _engineer_monthly(monthly_raw: pd.DataFrame) -> pd.DataFrame:
    """Full monthly feature engineering pipeline."""
    df = monthly_raw.copy()
    df = _add_monthly_lag_roll(df)
    df = _add_anomaly_features(df)
    df = _add_seasonal_encoding(df)
    df = _add_interaction_features(df)
    df = _add_gldas_features(df)
    return df


def _monthly_feature_cols(exclude_target_base: list[str]) -> list[str]:
    """
    Build the ordered feature column list for a monthly model.
    exclude_target_base: base column name(s) whose derived features must be
    removed (e.g. ['grace_groundwater_anomaly'] for Model 2).
    """
    base_features = [
        # Base inputs
        "population",
        "irrigation_total_withdrawal_mgd",
        "public_supply_groundwater_mgd",
        "mead_pool_elevation",
        "mead_total_release",
        "usdm_dsci",
        "temperature_2m_c",
        "precipitation_mm_day",
        "impervious_pct",
        # Lags and rolls for base inputs
        "population_lag1",
        "population_roll12",
        "irrigation_total_withdrawal_mgd_lag1",
        "irrigation_total_withdrawal_mgd_roll3",
        "irrigation_total_withdrawal_mgd_roll12",
        "public_supply_groundwater_mgd_lag1",
        "public_supply_groundwater_mgd_roll3",
        "mead_pool_elevation_lag1",
        "mead_pool_elevation_lag3",
        "mead_pool_elevation_roll3",
        "mead_pool_elevation_roll6",
        "mead_total_release_lag1",
        "mead_total_release_roll3",
        "usdm_dsci_lag1",
        "usdm_dsci_lag3",
        "usdm_dsci_roll3",
        "usdm_dsci_roll6",
        "usdm_dsci_roll12",
        "temperature_2m_c_lag1",
        "temperature_2m_c_roll3",
        "temperature_2m_c_roll6",
        "temperature_2m_c_roll12",
        "temperature_2m_c_anomaly",
        "temperature_2m_c_anomaly_lag1",
        "temperature_2m_c_anomaly_roll3",
        "precipitation_mm_day_lag1",
        "precipitation_mm_day_roll3",
        "precipitation_mm_day_roll6",
        "precipitation_mm_day_roll12",
        "precipitation_mm_day_anomaly",
        "precipitation_mm_day_anomaly_lag1",
        "precipitation_mm_day_anomaly_roll3",
        "impervious_pct_lag1",
        "impervious_pct_roll12",
        # GRACE features
        "grace_groundwater_anomaly",
        "grace_groundwater_anomaly_lag1",
        "grace_groundwater_anomaly_lag3",
        "grace_groundwater_anomaly_roll3",
        "grace_groundwater_anomaly_roll6",
        "grace_available",
        # NDVI features
        "ndvi",
        "ndvi_lag1",
        "ndvi_lag3",
        "ndvi_roll3",
        "ndvi_roll6",
        # Groundwater well level (per-well anomaly; positive = deeper than normal)
        "depth_to_water_anomaly_ft",
        "depth_to_water_anomaly_ft_lag1",
        "depth_to_water_anomaly_ft_lag3",
        "depth_to_water_anomaly_ft_roll3",
        "depth_to_water_anomaly_ft_roll6",
        # Surface water discharge (per-gage log anomaly)
        "discharge_log_anomaly",
        "discharge_log_anomaly_lag1",
        "discharge_log_anomaly_lag3",
        "discharge_log_anomaly_roll3",
        "discharge_log_anomaly_roll6",
        # Interaction features
        "precip_x_impervious",
        "precip_x_temperature",
        # Seasonal
        "month_sin",
        "month_cos",
    ]

    # Remove any feature whose name starts with an excluded base column
    filtered = []
    for feat in base_features:
        exclude = False
        for exc in exclude_target_base:
            if feat == exc or feat.startswith(exc + "_"):
                exclude = True
                break
        if not exclude:
            filtered.append(feat)

    return filtered


def _add_annual_lag_roll(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add 1-year and 2-year lags and 3-year rolling mean for annual columns."""
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
        df[f"{col}_roll3"] = df[col].shift(1).rolling(3, min_periods=2).mean()
    return df


def _engineer_annual(annual_raw: pd.DataFrame) -> pd.DataFrame:
    """Full annual feature engineering pipeline."""
    df = annual_raw.copy()

    annual_lag_roll_cols = [
        "nclimdiv_pdsi_annual_mean",
        "nclimdiv_temperature_c_annual_mean",
        "nclimdiv_precipitation_mm_day_annual_sum",
        "population_annual_mean",
        "irrigation_total_withdrawal_mgd_annual_sum",
        "public_supply_groundwater_mgd_annual_sum",
        "mead_pool_elevation_annual_mean",
        "mead_total_release_annual_sum",
        "usdm_dsci_annual_mean",
        "usdm_dsci_jja_mean",
        "temperature_2m_c_annual_mean",
        "temperature_2m_c_jja_mean",
        "precipitation_mm_day_annual_sum",
        "log_precip_annual",
        "grace_groundwater_anomaly_annual_mean",
        "ndvi_annual_mean",
        "ndvi_jja_mean",
        "impervious_pct_annual_mean",
        "wildfire_risk_index",
        "bbs_abundance_anomaly",
    ]
    df = _add_annual_lag_roll(df, annual_lag_roll_cols)

    # --- Fire-ecology features ---
    df = _add_fire_ecology_features(df)

    return df


def _add_fire_ecology_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fire-ecology-specific features motivated by Southwest fire science:
    - Wet winter → dry summer mechanism (fuel growth then ignition)
    - Consecutive dry year count (cumulative fuel stress)
    - Prior 2-year precipitation total (fuel load accumulation time)
    - Wet-then-dry interaction term
    """
    df = df.copy()

    # Prior year precipitation
    if "precipitation_mm_day_annual_sum" in df.columns:
        precip = df["precipitation_mm_day_annual_sum"]
        df["precip_prior_2yr_sum"] = precip.shift(1) + precip.shift(2)
        df["precip_prior_1yr"] = precip.shift(1)

    if (
        "precipitation_mm_day_annual_sum" in df.columns
        and "temperature_2m_c_jja_mean" in df.columns
    ):
        df["wet_then_dry"] = df["precip_prior_1yr"] * df["temperature_2m_c_jja_mean"]

    if "precipitation_mm_day_annual_sum" in df.columns:
        precip = df["precipitation_mm_day_annual_sum"]
        # Expanding, not full-series: a full-series median would let a given year's
        # "is this a dry year" flag depend on precipitation that had not happened yet.
        median_precip = precip.expanding(min_periods=3).median()
        is_dry = (precip < median_precip).astype(int)
        dry_count = []
        count = 0
        for val in is_dry:
            if val == 1:
                count += 1
            else:
                count = 0
            dry_count.append(count)
        df["consecutive_dry_years"] = dry_count

    # Temperature anomaly relative to period mean (is this year unusually hot?)
    if "temperature_2m_c_jja_mean" in df.columns:
        jja = df["temperature_2m_c_jja_mean"]
        df["jja_temp_anomaly"] = jja - jja.expanding(min_periods=3).mean()

    # USDM drought persistence (max drought severity in prior 2 years)
    if "usdm_dsci_jja_mean" in df.columns:
        dsci = df["usdm_dsci_jja_mean"]
        df["dsci_max_prior_2yr"] = pd.concat(
            [dsci.shift(1), dsci.shift(2)], axis=1
        ).max(axis=1)

    return df


# ---------------------------------------------------------------------------
# Feature columns for annual models
# ---------------------------------------------------------------------------

# Wildlife runs on the BBS record: 1968-2024. Every input except nClimDiv floors at
# 2000, so
# reaching back means using nClimDiv (drought/temperature/precipitation, 1895+) and
# nothing
# else. NDVI, GRACE, USDM, Lake Mead, irrigation, public supply, population and
# impervious %
# are all satellite-or-2000-era series; a single one of them in X would NaN-mask the
# panel
# straight back to 20 rows via the not-null filter.
#
# The old list was 22 features on 20 rows. This is 12 on 56 — the p >> n problem that
# PHASE2_REPORT.md called unfixable is simply gone, and it cost no new downloads beyond
# a
# free NOAA text file.
#
# What is given up: NDVI as a food-availability proxy, which is genuinely the mechanism
# you
# would want for birds. That is a real loss, and it is the trade for 36 extra years. If
# the
# NDVI link matters more than sample size, the 2000-2023 feature set is still buildable
# —
# but at n=20 it was not learnable, which is the whole reason for this change.
_WILDLIFE_FEATURES = [
    "nclimdiv_pdsi_annual_mean",
    "nclimdiv_pdsi_jja_mean",
    "nclimdiv_temperature_c_annual_mean",
    "nclimdiv_temperature_c_jja_mean",
    "nclimdiv_precipitation_mm_day_annual_sum",
    "nclimdiv_log_precip_annual",
    "year_linear",
    "bbs_abundance_anomaly_lag1",
    "bbs_abundance_anomaly_lag2",
    "bbs_abundance_anomaly_roll3",
    "nclimdiv_pdsi_annual_mean_lag1",
    "nclimdiv_precipitation_mm_day_annual_sum_lag1",
]

# ---------------------------------------------------------------------------
# Monthly model specs
# ---------------------------------------------------------------------------
# Each model declares its own target, its own end date, and its own feature set.
#
# `extended` models run to 2023-12. The price is dropping every feature derived
# from MONTHLY_CAPPED_2020_BASES — irrigation, public supply, well depth and
# discharge all stop at 2020-12, and one NaN column would mask the extra rows
# straight back off again. That price is low: those inputs are near-pure
# month-of-year templates already carried by month_sin/month_cos.
#
# `groundwater` and `surface_water` are not extended because their *targets*
# end 2020-12. There are no extra rows for them to gain, so they keep the full
# feature set.

# Features available before 2000. Everything else in the panel — USDM, MERRA-2
# temperature/precipitation, NDVI, GRACE, Lake Mead, population, irrigation, public
# supply, impervious % — floors at 2000, and a single one of them in X would NaN-mask
# the model straight back to the short window.  The price is real and worth stating:
# this drops `usdm_dsci`, which PHASE2_REPORT.md's variance decomposition names the
# single highest-signal input in the entire panel, and it drops `precip_x_impervious`,
# which was a top-5 feature for surface water. A model on this list is buying rows
# with signal. Whether that is a good trade is an empirical question, and the answer
# is not obviously yes — see PROBLEMS.md Part 4, where extending the window to rescue
# GRACE and wildfire made both *worse*.
_LONG_RECORD_MONTHLY_FEATURES = [
    "nclimdiv_pdsi",
    "nclimdiv_pdsi_lag1",
    "nclimdiv_pdsi_lag3",
    "nclimdiv_pdsi_roll3",
    "nclimdiv_pdsi_roll6",
    "nclimdiv_pdsi_roll12",
    "nclimdiv_temperature_c",
    "nclimdiv_temperature_c_lag1",
    "nclimdiv_temperature_c_roll3",
    "nclimdiv_temperature_c_roll6",
    "nclimdiv_temperature_c_roll12",
    "nclimdiv_temperature_c_anomaly",
    "nclimdiv_temperature_c_anomaly_lag1",
    "nclimdiv_temperature_c_anomaly_roll3",
    "nclimdiv_precipitation_mm_day",
    "nclimdiv_precipitation_mm_day_lag1",
    "nclimdiv_precipitation_mm_day_lag3",
    "nclimdiv_precipitation_mm_day_roll3",
    "nclimdiv_precipitation_mm_day_roll6",
    "nclimdiv_precipitation_mm_day_roll12",
    "nclimdiv_precipitation_mm_day_anomaly",
    "nclimdiv_precipitation_mm_day_anomaly_lag1",
    "nclimdiv_precipitation_mm_day_anomaly_roll3",
    "nclimdiv_precip_x_temperature",
    "month_sin",
    "month_cos",
]


# Discharge is a *response* variable, not a human or climate pressure. Until now the
# extended models never saw it — not by design, but as an accident of the data:
# discharge ended in 2020 and so was dropped by MONTHLY_CAPPED_2020_BASES. Extending
# NWIS to 2025 removed that accident and silently handed these models a new feature
# block, which cost
# wildfire 0.034 skill (+0.5615 -> +0.5271).
#  So the exclusion is now explicit and stated, rather than implied by a coverage gap.
# If feeding discharge into NDVI is worth trying on the merits, that is its own
# experiment — it should not arrive as a side effect of a Phase 1 window change.
_RESPONSE_FEATURE_BASES = ["discharge_log_anomaly", "discharge_cfs_mean", "n_gages"]

_MONTHLY_MODEL_SPECS: dict[str, dict] = {
    "ndvi": {
        "target": "ndvi",
        "extended": True,
        "also_exclude": [*_RESPONSE_FEATURE_BASES],
    },
    "grace": {
        "target": "grace_groundwater_anomaly",
        "extended": True,
        "also_exclude": [*_RESPONSE_FEATURE_BASES],
        # PHASE3_PLAN.md §26: GRACE is a ridge on four physical inputs, not the
        # 45-column catalogue. The list was fixed before the experiment that chose
        # the model class, and it is not to be edited by adding or removing columns
        # afterwards; a change here is a new pre-declared experiment.
        "fixed_features": [
            "gldas_tws_proxy_delta",
            "precipitation_mm_day",
            "precipitation_mm_day_lag1",
            "temperature_2m_c_anomaly",
        ],
        # GRACE has no instrument for 2000-01..2002-03 (zero-filled) or across
        # the GRACE→GRACE-FO gap (interpolated). Those months are fabrication,
        # not measurement, and must never appear in the target or in the lag1
        # anchor the residual is built on.
        "require_real_target": "grace_available",
    },
    "groundwater": {
        # Per-well anomaly, not the raw roster mean. The raw mean moves when the set of
        # reporting wells changes (turnover correlates +0.75 with its month-to-month
        # jump), which is not a thing any model can learn.
        "target": "depth_to_water_anomaly_ft",
        "extended": False,
        # depth_to_water_ft_mean is the *same measurement* as the target, just averaged
        # without centering. Leaving it in X would hand the model the answer.
        "also_exclude": ["depth_to_water_ft_mean", "n_wells"],
    },
    "surface_water": {
        "target": "discharge_log_anomaly",
        "extended": False,
        # NWIS now runs 1980-2025, so this is the longest monthly series in the project.
        # Reaching back past 2000 means the feature set collapses to nClimDiv alone —
        # no usdm_dsci, no impervious interaction, 63 features down to 26.
        #
        # That trade was measured rather than assumed, because it is exactly the trade
        # that failed for GRACE and wildfire (PROBLEMS.md Part 4). Holding the test rows
        # fixed at the common 2002-10..2020-12 period and varying only the training
        # data:
        #
        #     SHORT  2002-2020, 63 features   R² = +0.6777   skill = +0.6344
        #     LONG   1980-2025, 26 features   R² = +0.7955   skill = +0.7522
        #
        # Same 180 test months, same persistence baseline (+0.0433). The 332 extra
        # months buy more than the lost features cost. n: 219 -> 551.
        "long_record": True,
        "window_start": "1980-01",
        "window_end": "2025-12",
        # discharge_cfs_mean is the same measurement as the target; n_gages is the
        # roster size, which is exactly the artifact the anomaly index removes.
        "also_exclude": [
            "gage_height_ft_mean",
            "discharge_cfs_mean",
            "n_gages",
        ],
    },
    "wildfire_monthly": {
        "target": "wildfire_risk_index",
        "extended": False,
        # MTBS ignition dates run back to 1984, so this is a long-record model like
        # surface_water. Reaching before 2000 collapses the feature set to nClimDiv
        # alone — no usdm_dsci, no MERRA-2, no NDVI/GRACE cross-features — because
        # every one of those floors at 2000 and a single NaN column would mask the
        # pre-2000 rows straight back off again. That is an acceptable trade here
        # precisely because fire is driven by drought/heat/precipitation, which is
        # exactly what nClimDiv (PDSI/temp/precip, 1895+) carries. The old target
        # extension made wildfire *worse* (PROBLEMS.md Part 4), but that was a
        # fabricated DATE_CUR time axis; the MTBS target is real, so the extra ~192
        # months are real signal. Measured before/after in PHASE2_REPORT.md.
        "long_record": True,
        "window_start": "1984-01",
        "window_end": "2023-12",
        "also_exclude": [
            "fire_count",
            "log_acres",
            "total_acres",
            *_RESPONSE_FEATURE_BASES,
        ],
    },
}


def _monthly_dataset(
    monthly: pd.DataFrame, spec: dict
) -> tuple[pd.DataFrame, pd.Series]:
    """Slice the engineered monthly panel into one model's (X, y)."""
    target = spec["target"]
    extended = spec["extended"]

    # A model may override either bookend. The global MONTHLY_WINDOW_START (2002-10)
    # exists
    # only because GRACE is zero-filled before it; a model carrying no GRACE features
    # is not
    # bound by it and can run back to whatever its own target and features support.
    window_start = spec.get("window_start", MONTHLY_WINDOW_START)
    window_end = spec.get(
        "window_end", MONTHLY_WINDOW_END_EXTENDED if extended else MONTHLY_WINDOW_END
    )
    window = monthly.loc[
        pd.Period(window_start, freq="M") : pd.Period(window_end, freq="M")
    ].copy()

    if spec.get("long_record"):
        # Pre-2000 rows exist only for nClimDiv, so the feature set is restricted to it.
        exclude = [target, *spec.get("also_exclude", [])]
        features = [
            f
            for f in _LONG_RECORD_MONTHLY_FEATURES
            if not any(f == e or f.startswith(e + "_") for e in exclude)
        ]
        features = [f for f in features if f in window.columns]
        lag1_col = f"{target}_lag1"
        x = window[features + [lag1_col]].copy()
        y = window[target].copy()
        mask = x.notna().all(axis=1) & y.notna()
        return x[mask], y[mask]

    exclude = [target, *spec.get("also_exclude", [])]
    features = spec.get("fixed_features") or _monthly_feature_cols(exclude_target_base=exclude)

    if extended and not spec.get("fixed_features"):
        features = [
            f
            for f in features
            if not any(
                f == base or f.startswith(base + "_")
                for base in MONTHLY_CAPPED_2020_BASES
            )
        ]

    features = [f for f in features if f in window.columns]

    lag1_col = f"{target}_lag1"
    x = window[features + [lag1_col]].copy()  # lag1 kept for the residual anchor
    y = window[target].copy()
    mask = x.notna().all(axis=1) & y.notna()

    flag = spec.get("require_real_target")
    if flag is not None:
        # Both the target and the lag1 anchor must be real measurements —
        # otherwise the residual is a change measured against a filled value.
        # Shifted on the full panel so the first window row sees its true
        # predecessor rather than a NaN introduced by the slice.
        real = monthly[flag] == 1
        real_and_lag1 = (real & real.shift(1, fill_value=False)).loc[window.index]
        mask &= real_and_lag1
        # Constant once filtered, and it would leak the fill pattern anyway.
        x = x.drop(columns=[flag], errors="ignore")

    return x[mask], y[mask]


# ---------------------------------------------------------------------------
# Build all datasets
# ---------------------------------------------------------------------------


def build_all(  # noqa: PLR0915
    monthly_raw: pd.DataFrame | None = None,
    annual_raw: pd.DataFrame | None = None,
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Build (X, y) for all models.

    Parameters
    ----------
    monthly_raw, annual_raw : optional
        Pre-built panels from merge.py. If None, panels are built fresh.

    Returns
    -------
    dict with keys "ndvi", "grace", "wildfire", "wildlife"
    Each value is (X: pd.DataFrame, y: pd.Series), index-aligned, no NaNs.
    """
    if monthly_raw is None:
        monthly_raw = build_monthly_panel()
    if annual_raw is None:
        annual_raw = build_annual_panel(monthly_raw)

    datasets = {}

    # -----------------------------------------------------------------------
    # Monthly models
    # -----------------------------------------------------------------------
    monthly = _engineer_monthly(monthly_raw)

    for model_id, spec in _MONTHLY_MODEL_SPECS.items():
        if spec["target"] not in monthly.columns:
            continue
        datasets[model_id] = _monthly_dataset(monthly, spec)

    # Model 6 - Wildlife
    annual = _engineer_annual(annual_raw)
    annual_w = annual.loc[ANNUAL_WINDOW_START:ANNUAL_WINDOW_END].copy()

    wildlife_w = annual_w.drop(index=2020, errors="ignore")
    wl_features = [f for f in _WILDLIFE_FEATURES if f in wildlife_w.columns]
    x_wildlife = wildlife_w[wl_features].copy()
    y_wildlife = wildlife_w["bbs_abundance_anomaly"].copy()
    _base_cols_wl = [c for c in wl_features if "_lag" not in c and "_roll" not in c]
    mask = x_wildlife[_base_cols_wl].notna().all(axis=1) & y_wildlife.notna()
    datasets["wildlife"] = (x_wildlife[mask], y_wildlife[mask])

    return datasets


def main() -> None:
    datasets = build_all()
    print("\nFeature engineering complete.\n")
    for name, (x, y) in datasets.items():
        print(
            f"  {name:<10}  X: {x.shape}  y: {y.shape}  "
            f"index: {y.index[0]} → {y.index[-1]}"
        )
        null_counts = x.isna().sum()
        if null_counts.any():
            print(f"    WARNING: NaN columns: {null_counts[null_counts > 0].to_dict()}")


if __name__ == "__main__":
    main()

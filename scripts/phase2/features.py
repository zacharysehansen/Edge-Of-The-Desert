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
    MONTHLY_WINDOW_END,
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
_LAG_ROLL_COLS = _MONTHLY_BASE_INPUTS + [
    "grace_groundwater_anomaly",
    "ndvi",
    "wildfire_risk_index",
    "fire_count",
    "log_acres",
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
    for col in ("temperature_2m_c", "precipitation_mm_day"):
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
    return df


def _engineer_monthly(monthly_raw: pd.DataFrame) -> pd.DataFrame:
    """Full monthly feature engineering pipeline."""
    df = monthly_raw.copy()
    df = _add_monthly_lag_roll(df)
    df = _add_anomaly_features(df)
    df = _add_seasonal_encoding(df)
    df = _add_interaction_features(df)
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
        # Groundwater well level
        "depth_to_water_ft_mean",
        "depth_to_water_ft_mean_lag1",
        "depth_to_water_ft_mean_lag3",
        "depth_to_water_ft_mean_roll3",
        "depth_to_water_ft_mean_roll6",
        # Surface water discharge
        "discharge_cfs_mean",
        "discharge_cfs_mean_lag1",
        "discharge_cfs_mean_lag3",
        "discharge_cfs_mean_roll3",
        "discharge_cfs_mean_roll6",
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
        "bbs_abundance_index",
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
        median_precip = precip.median()
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

_WILDLIFE_FEATURES = [
    "population_annual_mean",
    "irrigation_total_withdrawal_mgd_annual_sum",
    "public_supply_groundwater_mgd_annual_sum",
    "mead_pool_elevation_annual_mean",
    "mead_pool_elevation_june",
    "mead_total_release_annual_sum",
    "usdm_dsci_annual_mean",
    "temperature_2m_c_annual_mean",
    "temperature_2m_c_jja_mean",
    "precipitation_mm_day_annual_sum",
    "log_precip_annual",
    "ndvi_annual_mean",
    "ndvi_jja_mean",
    "grace_groundwater_anomaly_annual_mean",
    "impervious_pct_annual_mean",
    "year_linear",
    "bbs_abundance_index_lag1",
    "bbs_abundance_index_lag2",
    "bbs_abundance_index_roll3",
    "ndvi_annual_mean_lag1",
    "precipitation_mm_day_annual_sum_lag1",
    "usdm_dsci_annual_mean_lag1",
]

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
    # Monthly models (Model 1: NDVI, Model 2: GRACE)
    # -----------------------------------------------------------------------
    monthly = _engineer_monthly(monthly_raw)

    window_start = pd.Period(MONTHLY_WINDOW_START, freq="M")
    window_end = pd.Period(MONTHLY_WINDOW_END, freq="M")
    monthly_w = monthly.loc[window_start:window_end].copy()

    # Model 1 - NDVI
    ndvi_features = _monthly_feature_cols(exclude_target_base=["ndvi"])
    ndvi_features = [f for f in ndvi_features if f in monthly_w.columns]
    x_ndvi = monthly_w[ndvi_features + ["ndvi_lag1"]].copy()  # lag1 kept for residual
    y_ndvi = monthly_w["ndvi"].copy()
    mask = x_ndvi.notna().all(axis=1) & y_ndvi.notna()
    datasets["ndvi"] = (x_ndvi[mask], y_ndvi[mask])

    # Model 2 - GRACE
    grace_features = _monthly_feature_cols(
        exclude_target_base=["grace_groundwater_anomaly"]
    )
    grace_features = [f for f in grace_features if f in monthly_w.columns]
    x_grace = monthly_w[grace_features + ["grace_groundwater_anomaly_lag1"]].copy()
    y_grace = monthly_w["grace_groundwater_anomaly"].copy()
    mask = x_grace.notna().all(axis=1) & y_grace.notna()
    datasets["grace"] = (x_grace[mask], y_grace[mask])

    # Model 3 - Groundwater Well Levels
    if "depth_to_water_ft_mean" in monthly_w.columns:
        gw_features = _monthly_feature_cols(
            exclude_target_base=["depth_to_water_ft_mean"]
        )
        gw_features = [f for f in gw_features if f in monthly_w.columns]
        x_gw = monthly_w[gw_features + ["depth_to_water_ft_mean_lag1"]].copy()
        y_gw = monthly_w["depth_to_water_ft_mean"].copy()
        mask = x_gw.notna().all(axis=1) & y_gw.notna()
        datasets["groundwater"] = (x_gw[mask], y_gw[mask])

    # Model 4 - Surface Water Conditions (discharge only; gage_height has gaps)
    if "discharge_cfs_mean" in monthly_w.columns:
        sw_features = _monthly_feature_cols(
            exclude_target_base=["discharge_cfs_mean", "gage_height_ft_mean"]
        )
        sw_features = [f for f in sw_features if f in monthly_w.columns]
        x_sw = monthly_w[sw_features + ["discharge_cfs_mean_lag1"]].copy()
        y_sw = monthly_w["discharge_cfs_mean"].copy()
        mask = x_sw.notna().all(axis=1) & y_sw.notna()
        datasets["surface_water"] = (x_sw[mask], y_sw[mask])

    # Model 5 - Wildfire Risk Index
    if "wildfire_risk_index" in monthly_w.columns:
        wf_monthly_features = _monthly_feature_cols(
            exclude_target_base=[
                "wildfire_risk_index",
                "fire_count",
                "log_acres",
                "total_acres",
            ]
        )
        wf_monthly_features = [f for f in wf_monthly_features if f in monthly_w.columns]
        x_wf_monthly = monthly_w[
            wf_monthly_features + ["wildfire_risk_index_lag1"]
        ].copy()
        y_wf_monthly = monthly_w["wildfire_risk_index"].copy()
        mask = x_wf_monthly.notna().all(axis=1) & y_wf_monthly.notna()
        datasets["wildfire_monthly"] = (x_wf_monthly[mask], y_wf_monthly[mask])

    # Model 6 - Wildlife
    annual = _engineer_annual(annual_raw)
    annual_w = annual.loc[ANNUAL_WINDOW_START:ANNUAL_WINDOW_END].copy()

    wildlife_w = annual_w.drop(index=2020, errors="ignore")
    wl_features = [f for f in _WILDLIFE_FEATURES if f in wildlife_w.columns]
    x_wildlife = wildlife_w[wl_features].copy()
    y_wildlife = wildlife_w["bbs_abundance_index"].copy()
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

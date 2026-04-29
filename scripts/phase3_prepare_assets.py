import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "Final"
MODEL_DIR = REPO_ROOT / "model"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "frontend" / "public" / "model"

SOURCES = {
    "snotel": "snotel_swe.csv",
    "irrigation": "irrigation_huc12_monthly_az_2000_2020.csv",
    "public": "nwaa_public_supply_az_monthly.csv",
    "powell": "powell_combined.csv",
    "population": "azpop_monthly.csv",
    "ndvi": "modis_ndvi.csv",
    "precip": "merra_precipitation.csv",
    "temperature": "merra_temperature_2m.csv",
    "grace": "grace_groundwater_anomaly.csv",
    "streamflow": "usgs_streamflow.csv",
    "usdm": "usdm_sustainability.csv",
}

DROP_COLS = {
    "irrigation": [
        "irrigation_total_withdrawal_gallons_per_day",
        "irrigation_total_withdrawal_acre_feet_month",
        "huc12_count",
    ],
    "public": [
        "public_supply_groundwater_gallons_per_day",
        "public_supply_groundwater_acre_feet_month",
        "huc12_count",
    ],
}

CONTROLLED_MODEL_FEATURES = [
    "powell_pool_elevation",
    "snow_water_equivalent_in",
    "precipitation_mm_day",
    "temperature_2m_c",
    "irrigation_total_withdrawal_mgd",
    "public_supply_groundwater_mgd",
    "grace_groundwater_anomaly",
]

CONTROL_DEFAULT_OVERRIDES = {
    "grace_groundwater_anomaly": -0.077,
}

CONTROL_KNOBS = [
    {
        "id": "powell_pool_elevation",
        "label": "Lake Powell Pool Elevation",
        "unit": "ft",
        "decimals": 1,
        "source_column": "powell_pool_elevation",
        "source_file": SOURCES["powell"],
        "model_feature": "powell_pool_elevation",
        "mapping_type": "direct_feature",
    },
    {
        "id": "snow_water_equivalent_in",
        "label": "Snowpack",
        "unit": "in",
        "decimals": 2,
        "source_column": "snow_water_equivalent_in",
        "source_file": SOURCES["snotel"],
        "model_feature": "snow_water_equivalent_in",
        "mapping_type": "direct_feature",
    },
    {
        "id": "precipitation_mm_day",
        "label": "Precipitation",
        "unit": "mm/day",
        "decimals": 2,
        "source_column": "precipitation_mm_day",
        "source_file": SOURCES["precip"],
        "model_feature": "precipitation_mm_day",
        "mapping_type": "direct_feature",
    },
    {
        "id": "temperature_2m_c",
        "label": "Temperature",
        "unit": "C",
        "decimals": 1,
        "source_column": "temperature_2m_c",
        "source_file": SOURCES["temperature"],
        "model_feature": "temperature_2m_c",
        "mapping_type": "direct_feature",
    },
    {
        "id": "irrigation_total_withdrawal_mgd",
        "label": "Irrigation Withdrawal",
        "unit": "mgd",
        "decimals": 0,
        "source_column": "irrigation_total_withdrawal_mgd",
        "source_file": SOURCES["irrigation"],
        "model_feature": "irrigation_total_withdrawal_mgd",
        "mapping_type": "direct_feature",
    },
    {
        "id": "public_supply_groundwater_mgd",
        "label": "Public Supply Groundwater",
        "unit": "mgd",
        "decimals": 0,
        "source_column": "public_supply_groundwater_mgd",
        "source_file": SOURCES["public"],
        "model_feature": "public_supply_groundwater_mgd",
        "mapping_type": "direct_feature",
    },
    {
        "id": "grace_groundwater_anomaly",
        "label": "GRACE Groundwater Anomaly",
        "unit": "anomaly",
        "decimals": 3,
        "source_column": "grace_groundwater_anomaly",
        "source_file": SOURCES["grace"],
        "model_feature": "grace_groundwater_anomaly",
        "mapping_type": "direct_feature",
    },
]


def knob_default(name: str, stats: dict) -> float:
    return float(CONTROL_DEFAULT_OVERRIDES.get(name, stats["median"]))

CALENDAR_FEATURES = [
    "month_sin",
    "month_cos",
]

LAGGED_OR_ROLLING_SUFFIXES = (
    "_lag1",
    "_lag3",
    "_roll3",
    "_roll6",
)

BASE_DISPLAY_SPECS = {
    "snow_water_equivalent_in": {
        "label": "Snow Water Equivalent",
        "unit": "in",
        "decimals": 2,
        "control_enabled": True,
    },
    "irrigation_total_withdrawal_mgd": {
        "label": "Irrigation Withdrawal",
        "unit": "mgd",
        "decimals": 0,
        "control_enabled": True,
    },
    "public_supply_groundwater_mgd": {
        "label": "Public Supply Groundwater",
        "unit": "mgd",
        "decimals": 0,
        "control_enabled": True,
    },
    "powell_evaporation": {
        "label": "Lake Powell Evaporation",
        "unit": "acre-feet/month",
        "decimals": 0,
        "control_enabled": False,
    },
    "powell_total_release": {
        "label": "Lake Powell Total Release",
        "unit": "cfs",
        "decimals": 0,
        "control_enabled": False,
    },
    "powell_inflow": {
        "label": "Lake Powell Inflow",
        "unit": "cfs",
        "decimals": 0,
        "control_enabled": False,
    },
    "powell_pool_elevation": {
        "label": "Lake Powell Pool Elevation",
        "unit": "ft",
        "decimals": 1,
        "control_enabled": True,
    },
    "ndvi": {
        "label": "NDVI",
        "unit": "index",
        "decimals": 3,
        "control_enabled": False,
    },
    "precipitation_mm_day": {
        "label": "Precipitation",
        "unit": "mm/day",
        "decimals": 2,
        "control_enabled": True,
    },
    "temperature_2m_c": {
        "label": "Temperature",
        "unit": "C",
        "decimals": 1,
        "control_enabled": True,
    },
    "temperature_2m_c_anomaly": {
        "label": "Temperature Anomaly",
        "unit": "C",
        "decimals": 1,
        "control_enabled": False,
    },
    "grace_groundwater_anomaly": {
        "label": "GRACE Groundwater Anomaly",
        "unit": "anomaly",
        "decimals": 3,
        "control_enabled": True,
    },
    "streamflow_cfs": {
        "label": "Streamflow",
        "unit": "cfs",
        "decimals": 0,
        "control_enabled": False,
    },
    "grace_available": {
        "label": "GRACE Availability",
        "unit": "flag",
        "decimals": 0,
        "control_enabled": False,
    },
    "month_sin": {
        "label": "Seasonal Sine",
        "unit": "index",
        "decimals": 3,
        "control_enabled": False,
    },
    "month_cos": {
        "label": "Seasonal Cosine",
        "unit": "index",
        "decimals": 3,
        "control_enabled": False,
    },
    "usdm_sustainability": {
        "label": "Sustainability Score",
        "unit": "score",
        "decimals": 1,
        "control_enabled": False,
    },
    "AZPOP_pct_change": {
        "label": "Population Change",
        "unit": "fraction",
        "decimals": 4,
        "control_enabled": False,
    },
    "AZPOP": {
        "label": "Population",
        "unit": "people",
        "decimals": 0,
        "control_enabled": True,
    },
}


def log(message: str) -> None:
    print(f"[phase3_prepare_assets] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare browser-ready model and data assets for the Phase 3 frontend."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to receive the browser-ready assets.",
    )
    return parser.parse_args()


def load_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for key, filename in SOURCES.items():
        frame = pd.read_csv(DATA_DIR / filename)
        frame["year_month"] = pd.to_datetime(frame["year_month"]).dt.to_period("M")
        frame = frame.set_index("year_month").sort_index()
        if key in DROP_COLS:
            frame = frame.drop(columns=DROP_COLS[key], errors="ignore")
        frames[key] = frame
    return frames


def build_phase2_feature_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base_index = pd.period_range(start="2000-10", end="2020-12", freq="M")
    merged = pd.DataFrame(index=base_index)
    merged.index.name = "year_month"

    for frame in frames.values():
        merged = merged.join(frame, how="left")

    snotel_col = "snow_water_equivalent_in"
    snotel_observed = merged[snotel_col].dropna()
    monthly_climatology = (
        snotel_observed.groupby(snotel_observed.index.month).median().clip(lower=0)
    )
    gap_mask = merged[snotel_col].isnull()
    merged.loc[gap_mask, snotel_col] = merged.loc[gap_mask].index.month.map(
        monthly_climatology
    )

    merged = merged.interpolate(method="linear", limit_direction="both")
    merged = merged[merged["usdm_sustainability"].notna()]
    merged[snotel_col] = merged[snotel_col].clip(lower=0)

    grace_start = pd.Period("2002-04", freq="M")
    merged["grace_available"] = (merged.index >= grace_start).astype(float)
    merged.loc[merged.index < grace_start, "grace_groundwater_anomaly"] = 0.0

    months = merged.index.month
    merged["month_sin"] = np.sin(2 * np.pi * months / 12)
    merged["month_cos"] = np.cos(2 * np.pi * months / 12)

    temperature_monthly_median = (
        merged.groupby(merged.index.month)["temperature_2m_c"].transform("median")
    )
    merged["temperature_2m_c_anomaly"] = (
        merged["temperature_2m_c"] - temperature_monthly_median
    )

    lag_features = {
        "usdm_sustainability_lag1": ("usdm_sustainability", 1),
        "usdm_sustainability_lag3": ("usdm_sustainability", 3),
        "streamflow_cfs_lag1": ("streamflow_cfs", 1),
        "snow_water_equivalent_in_lag1": ("snow_water_equivalent_in", 1),
        "powell_pool_elevation_lag1": ("powell_pool_elevation", 1),
        "precipitation_mm_day_lag1": ("precipitation_mm_day", 1),
        "temperature_2m_c_lag1": ("temperature_2m_c", 1),
        "temperature_2m_c_anomaly_lag1": ("temperature_2m_c_anomaly", 1),
        "grace_groundwater_anomaly_lag1": ("grace_groundwater_anomaly", 1),
        "ndvi_lag1": ("ndvi", 1),
    }
    for name, (column, periods) in lag_features.items():
        merged[name] = merged[column].shift(periods)

    merged["usdm_sustainability_roll3"] = (
        merged["usdm_sustainability"].shift(1).rolling(3).mean()
    )
    merged["usdm_sustainability_roll6"] = (
        merged["usdm_sustainability"].shift(1).rolling(6).mean()
    )
    merged["streamflow_cfs_roll3"] = (
        merged["streamflow_cfs"].shift(1).rolling(3).mean()
    )
    merged["precipitation_mm_day_roll3"] = (
        merged["precipitation_mm_day"].shift(1).rolling(3).mean()
    )
    merged["precipitation_mm_day_roll6"] = (
        merged["precipitation_mm_day"].shift(1).rolling(6).mean()
    )
    merged["temperature_2m_c_roll3"] = (
        merged["temperature_2m_c"].shift(1).rolling(3).mean()
    )
    merged["temperature_2m_c_roll6"] = (
        merged["temperature_2m_c"].shift(1).rolling(6).mean()
    )
    merged["temperature_2m_c_anomaly_roll3"] = (
        merged["temperature_2m_c_anomaly"].shift(1).rolling(3).mean()
    )
    merged["grace_groundwater_anomaly_roll3"] = (
        merged["grace_groundwater_anomaly"].shift(1).rolling(3).mean()
    )
    merged["grace_groundwater_anomaly_roll6"] = (
        merged["grace_groundwater_anomaly"].shift(1).rolling(6).mean()
    )

    merged["AZPOP_pct_change"] = merged["AZPOP"].pct_change()
    merged = merged.drop(columns=["AZPOP", "powell_storage"])
    merged = merged.dropna()
    return merged


def load_model_artifacts() -> tuple[list[str], dict, dict, pd.DataFrame]:
    feature_names = json.loads((MODEL_DIR / "feature_names.json").read_text())
    feature_stats = json.loads((MODEL_DIR / "feature_stats.json").read_text())
    cv_results = json.loads((MODEL_DIR / "cv_results.json").read_text())
    historical = pd.read_csv(MODEL_DIR / "historical_sustainability.csv")
    historical["year_month"] = pd.to_datetime(historical["year_month"]).dt.to_period("M")
    historical = historical.set_index("year_month").sort_index()
    return feature_names, feature_stats, cv_results, historical


def build_raw_support_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    usdm = frames["usdm"].copy()
    base_index = usdm.index
    support = pd.DataFrame(index=base_index)

    raw_columns = [
        "AZPOP",
        "snow_water_equivalent_in",
        "irrigation_total_withdrawal_mgd",
        "public_supply_groundwater_mgd",
        "powell_evaporation",
        "powell_total_release",
        "powell_inflow",
        "powell_pool_elevation",
        "ndvi",
        "precipitation_mm_day",
        "temperature_2m_c",
        "grace_groundwater_anomaly",
        "streamflow_cfs",
    ]
    for column in raw_columns:
        for frame in frames.values():
            if column in frame.columns:
                support[column] = frame.reindex(base_index)[column]
                break

    support["grace_available"] = (support.index >= pd.Period("2002-04", freq="M")).astype(float)
    temperature_monthly_medians = support.groupby(support.index.month)["temperature_2m_c"].median()
    support["temperature_2m_c_anomaly"] = (
        support["temperature_2m_c"]
        - support.index.month.map(temperature_monthly_medians.to_dict())
    )
    support["AZPOP_pct_change"] = (
        frames["population"].reindex(base_index)["AZPOP"].pct_change(fill_method=None)
    )
    return support


def build_historical_feature_vectors(
    full_history: pd.DataFrame,
    raw_support: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    historical_predictions: pd.DataFrame,
) -> pd.DataFrame:
    historical_vectors = pd.DataFrame(index=full_history.index)
    historical_vectors["usdm_dsci"] = full_history["usdm_dsci"]
    historical_vectors["usdm_sustainability"] = full_history["usdm_sustainability"]
    historical_vectors["predicted_usdm_sustainability"] = historical_predictions.reindex(
        historical_vectors.index
    )["usdm_sustainability"]
    historical_vectors["prediction_residual"] = historical_predictions.reindex(
        historical_vectors.index
    )["residual"]

    export_features = feature_matrix.reindex(historical_vectors.index)
    for column in raw_support.columns:
        if column in export_features.columns:
            export_features[column] = export_features[column].combine_first(
                raw_support[column]
            )
        else:
            export_features[column] = raw_support[column]

    historical_vectors = historical_vectors.join(export_features)
    historical_vectors["model_complete"] = historical_vectors.index.isin(feature_matrix.index)
    historical_vectors["vector_status"] = np.where(
        historical_vectors["model_complete"],
        "model_complete",
        "historical_only",
    )
    historical_vectors = historical_vectors.reset_index()
    historical_vectors["year_month"] = historical_vectors["year_month"].astype(str)
    return historical_vectors


def suffix_metadata(feature_name: str) -> tuple[str, str | None]:
    for suffix in LAGGED_OR_ROLLING_SUFFIXES:
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)], suffix
    return feature_name, None


def feature_label(base_name: str, suffix: str | None) -> str:
    spec = BASE_DISPLAY_SPECS.get(base_name)
    base_label = spec["label"] if spec else base_name.replace("_", " ").title()
    if suffix == "_lag1":
        return f"{base_label} Lag 1"
    if suffix == "_lag3":
        return f"{base_label} Lag 3"
    if suffix == "_roll3":
        return f"{base_label} 3-Month Rolling Mean"
    if suffix == "_roll6":
        return f"{base_label} 6-Month Rolling Mean"
    return base_label


def feature_unit(base_name: str) -> str:
    spec = BASE_DISPLAY_SPECS.get(base_name)
    return spec["unit"] if spec else "value"


def feature_decimals(base_name: str) -> int:
    spec = BASE_DISPLAY_SPECS.get(base_name)
    return spec["decimals"] if spec else 3


def numeric_summary(series: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if clean.empty:
        raise ValueError("Cannot build summary statistics from an empty series.")
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "median": float(clean.median()),
        "p5": float(clean.quantile(0.05)),
        "p95": float(clean.quantile(0.95)),
    }


def build_control_metadata(
    feature_stats: dict,
    control_support_frame: pd.DataFrame,
) -> dict[str, dict]:
    controls: dict[str, dict] = {}
    for spec in CONTROL_KNOBS:
        if spec["mapping_type"] == "direct_feature":
            domain = {
                "min": float(feature_stats[spec["model_feature"]]["min"]),
                "max": float(feature_stats[spec["model_feature"]]["max"]),
                "median": float(feature_stats[spec["model_feature"]]["median"]),
                "p5": float(feature_stats[spec["model_feature"]]["p5"]),
                "p95": float(feature_stats[spec["model_feature"]]["p95"]),
            }
        else:
            domain = numeric_summary(control_support_frame[spec["source_column"]])

        control_payload = {
            "label": spec["label"],
            "unit": spec["unit"],
            "decimals": spec["decimals"],
            "source_column": spec["source_column"],
            "source_file": spec["source_file"],
            "domain": domain,
            "knob": {
                "default": knob_default(spec["id"], domain),
                "min": float(domain["p5"]),
                "max": float(domain["p95"]),
            },
            "model_mapping": {
                "type": spec["mapping_type"],
                "output_feature": spec["model_feature"],
            },
        }

        if spec["mapping_type"] == "direct_feature":
            control_payload["model_mapping"]["input_feature"] = spec["model_feature"]
        else:
            control_payload["model_mapping"]["input_feature"] = spec["source_column"]
            control_payload["model_mapping"]["notes"] = [
                "The UI stores an absolute monthly population scenario.",
                "The ONNX model consumes AZPOP_pct_change, so the browser must derive percentage change from the projected population path.",
                "If population is held constant after the first projected month, AZPOP_pct_change becomes 0.0 on later projected steps.",
            ]

        controls[spec["id"]] = control_payload

    return controls


def build_display_metadata(
    feature_names: list[str],
    feature_stats: dict,
    cv_results: dict,
    feature_frame: pd.DataFrame,
    control_support_frame: pd.DataFrame,
) -> dict:
    temperature_monthly_climatology = (
        feature_frame.groupby(feature_frame.index.month)["temperature_2m_c"].median().to_dict()
    )
    controls = build_control_metadata(
        feature_stats=feature_stats,
        control_support_frame=control_support_frame,
    )

    features = {}
    for name in feature_names:
        base_name, suffix = suffix_metadata(name)
        knob_enabled = (
            suffix is None
            and base_name in CONTROLLED_MODEL_FEATURES
        )
        stats = feature_stats[name]
        features[name] = {
            "label": feature_label(base_name, suffix),
            "unit": feature_unit(base_name),
            "decimals": feature_decimals(base_name),
            "knob_enabled": knob_enabled,
            "slider_enabled": knob_enabled,
            "domain": {
                "min": float(stats["min"]),
                "max": float(stats["max"]),
                "median": float(stats["median"]),
                "p5": float(stats["p5"]),
                "p95": float(stats["p95"]),
            },
        }
        if knob_enabled:
            knob_payload = {
                "default": knob_default(name, stats),
                "min": float(stats["p5"]),
                "max": float(stats["p95"]),
            }
            features[name]["knob"] = knob_payload
            features[name]["slider"] = knob_payload

    derived_features = [
        name
        for name in feature_names
        if name not in CONTROLLED_MODEL_FEATURES + CALENDAR_FEATURES
        and (
            name.startswith("usdm_sustainability_")
            or name.endswith(LAGGED_OR_ROLLING_SUFFIXES)
            or name == "temperature_2m_c_anomaly"
        )
    ]
    fixed_raw_features = [
        name
        for name in feature_names
        if name not in CONTROLLED_MODEL_FEATURES + CALENDAR_FEATURES + derived_features
    ]

    return {
        "selected_model_name": cv_results["selected_model_name"],
        "selected_model_label": cv_results["selected_model_label"],
        "selected_feature_set": cv_results["selected_feature_set"],
        "selected_feature_count": len(feature_names),
        "score_summary": {
            "mean_r2": float(cv_results["mean_r2"]),
            "mean_mae": float(cv_results["mean_mae"]),
            "std_r2": float(cv_results["std_r2"]),
            "std_mae": float(cv_results["std_mae"]),
        },
        "timeline": {
            "model_complete_start": str(feature_frame.index.min()),
            "model_complete_end": str(feature_frame.index.max()),
        },
        "knob_controls": [spec["id"] for spec in CONTROL_KNOBS],
        "controlled_model_features": CONTROLLED_MODEL_FEATURES,
        "slider_features": CONTROLLED_MODEL_FEATURES,
        "controls": controls,
        "projection_policy": {
            "knob_controls": [spec["id"] for spec in CONTROL_KNOBS],
            "controlled_model_features": CONTROLLED_MODEL_FEATURES,
            "slider_features": CONTROLLED_MODEL_FEATURES,
            "calendar_features": CALENDAR_FEATURES,
            "derived_features": derived_features,
            "fixed_raw_features": fixed_raw_features,
            "notes": [
                "The seven UI knobs are user-controlled scenario inputs.",
                "Month sin/cos are recomputed from the projection date.",
                "Lagged and rolling features are updated by the projection engine.",
                "Streamflow remains model-facing but is held fixed by default under the seven-knob plan.",
                "All other exogenous raw features are seeded from the latest complete observed row and held fixed by default.",
            ],
            "temperature_monthly_climatology_c": {
                str(month): float(value)
                for month, value in temperature_monthly_climatology.items()
            },
        },
        "target_definition": {
            "source_feature": "usdm_dsci",
            "target_feature": "usdm_sustainability",
            "formula": "100 - (usdm_dsci / 5)",
            "inverse_formula": "usdm_dsci = 5 * (100 - usdm_sustainability)",
            "notes": [
                "Higher DSCI means worse drought severity.",
                "The exported target is an inverted drought-severity proxy on a 0-100 scale.",
                "The model predicts usdm_sustainability directly; same-month usdm_dsci is not part of the input feature row.",
            ],
        },
        "score_encoding": {
            "domain": [0, 100],
            "threshold_bands": [
                {"label": "Severe", "start": 0, "end": 33, "color": "#c65135"},
                {"label": "Moderate", "start": 33, "end": 66, "color": "#ddb44e"},
                {"label": "Healthy", "start": 66, "end": 100, "color": "#3e7ea6"},
            ],
            "color_ramp": ["#c65135", "#ddb44e", "#3e7ea6"],
        },
        "aquifer_encoding": {
            "dry_fill": "#b48248",
            "wet_fill": "#607989",
            "water_table_line": "#245f8f",
            "ground_line": "#60472f",
        },
        "features": features,
    }


def build_projection_seed(
    feature_frame: pd.DataFrame,
    control_support_frame: pd.DataFrame,
    historical_predictions: pd.DataFrame,
    feature_names: list[str],
    display_metadata: dict,
) -> dict:
    last_month = feature_frame.index.max()
    next_month = last_month + 1
    last_row = feature_frame.loc[last_month]
    last_control_row = control_support_frame.loc[last_month]
    actual_tail = feature_frame["usdm_sustainability"].loc[last_month - 5 : last_month]

    def tail_history(frame: pd.DataFrame, column: str, count: int) -> list[dict]:
        series = frame[column].loc[last_month - (count - 1) : last_month]
        return [
            {"year_month": str(index), "value": float(value)}
            for index, value in series.items()
        ]

    control_seed_values = {
        name: display_metadata["controls"][name]["knob"]["default"]
        for name in display_metadata["knob_controls"]
    }
    fixed_raw_feature_names = display_metadata["projection_policy"]["fixed_raw_features"]
    fixed_raw_feature_values = {
        name: float(last_row[name])
        for name in fixed_raw_feature_names
    }

    return {
        "source_last_observed_month": str(last_month),
        "projection_start_month": str(next_month),
        "latest_observed_score": float(feature_frame.loc[last_month, "usdm_sustainability"]),
        "latest_predicted_score": float(
            historical_predictions.loc[last_month, "usdm_sustainability"]
        ),
        "feature_row_last_observed": {
            name: float(last_row[name])
            for name in feature_names
        },
        "latest_observed_control_values": {
            "powell_pool_elevation": float(last_row["powell_pool_elevation"]),
            "snow_water_equivalent_in": float(last_row["snow_water_equivalent_in"]),
            "precipitation_mm_day": float(last_row["precipitation_mm_day"]),
            "temperature_2m_c": float(last_row["temperature_2m_c"]),
            "irrigation_total_withdrawal_mgd": float(last_row["irrigation_total_withdrawal_mgd"]),
            "public_supply_groundwater_mgd": float(last_row["public_supply_groundwater_mgd"]),
            "grace_groundwater_anomaly": float(last_row["grace_groundwater_anomaly"]),
        },
        "control_seed_values": control_seed_values,
        "slider_seed_values": {
            name: control_seed_values[name]
            for name in CONTROLLED_MODEL_FEATURES
        },
        "fixed_raw_feature_values": fixed_raw_feature_values,
        "target_history_last6": [
            {"year_month": str(index), "value": float(value)}
            for index, value in actual_tail.items()
        ],
        "exogenous_history": {
            "streamflow_cfs_last3": tail_history(feature_frame, "streamflow_cfs", 3),
            "snow_water_equivalent_in_last1": tail_history(feature_frame, "snow_water_equivalent_in", 1),
            "powell_pool_elevation_last1": tail_history(feature_frame, "powell_pool_elevation", 1),
            "precipitation_mm_day_last6": tail_history(feature_frame, "precipitation_mm_day", 6),
            "temperature_2m_c_last6": tail_history(feature_frame, "temperature_2m_c", 6),
            "grace_groundwater_anomaly_last6": tail_history(feature_frame, "grace_groundwater_anomaly", 6),
            "irrigation_total_withdrawal_mgd_last1": tail_history(
                feature_frame, "irrigation_total_withdrawal_mgd", 1
            ),
            "public_supply_groundwater_mgd_last1": tail_history(
                feature_frame, "public_supply_groundwater_mgd", 1
            ),
            "population_last2": tail_history(control_support_frame, "AZPOP", 2),
            "population_change_last2": tail_history(feature_frame, "AZPOP_pct_change", 2),
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def copy_static_assets(output_dir: Path) -> None:
    copy_map = {
        MODEL_DIR / "water_sustainability.onnx": output_dir / "water_sustainability.onnx",
        MODEL_DIR / "feature_names.json": output_dir / "feature_names.json",
        MODEL_DIR / "feature_stats.json": output_dir / "feature_stats.json",
        MODEL_DIR / "historical_sustainability.csv": output_dir / "historical_sustainability.csv",
        DATA_DIR / "grace_groundwater_anomaly.csv": output_dir / "grace_groundwater_anomaly.csv",
        DATA_DIR / "merra_precipitation.csv": output_dir / "merra_precipitation.csv",
        DATA_DIR / "merra_temperature_2m.csv": output_dir / "merra_temperature_2m.csv",
        DATA_DIR / "snotel_swe.csv": output_dir / "snotel_swe.csv",
        DATA_DIR / "powell_combined.csv": output_dir / "powell_combined.csv",
        DATA_DIR / "irrigation_huc12_monthly_az_2000_2020.csv": output_dir / "irrigation_huc12_monthly_az_2000_2020.csv",
        DATA_DIR / "nwaa_public_supply_az_monthly.csv": output_dir / "nwaa_public_supply_az_monthly.csv",
        DATA_DIR / "azpop_monthly.csv": output_dir / "azpop_monthly.csv",
    }
    for source, destination in copy_map.items():
        shutil.copy2(source, destination)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log("loading phase 2 source frames")
    frames = load_frames()
    feature_frame = build_phase2_feature_frame(frames)

    log("loading model artifacts")
    feature_names, feature_stats, cv_results, historical_predictions = load_model_artifacts()
    model_complete_frame = feature_frame.loc[historical_predictions.index, feature_names].copy()
    if list(model_complete_frame.index.astype(str)) != list(historical_predictions.index.astype(str)):
        raise RuntimeError("The rebuilt feature frame does not match historical_sustainability.csv")

    full_history = frames["usdm"][["usdm_dsci", "usdm_sustainability"]].copy()
    raw_support = build_raw_support_frame(frames)
    control_support = raw_support.loc[historical_predictions.index].copy()

    log("building historical_feature_vectors.csv")
    historical_feature_vectors = build_historical_feature_vectors(
        full_history=full_history,
        raw_support=raw_support,
        feature_matrix=model_complete_frame,
        historical_predictions=historical_predictions,
    )
    historical_feature_vectors.to_csv(output_dir / "historical_feature_vectors.csv", index=False)

    log("building display_metadata.json")
    display_metadata = build_display_metadata(
        feature_names=feature_names,
        feature_stats=feature_stats,
        cv_results=cv_results,
        feature_frame=feature_frame.loc[historical_predictions.index],
        control_support_frame=control_support,
    )
    write_json(output_dir / "display_metadata.json", display_metadata)

    log("building projection_seed.json")
    projection_seed = build_projection_seed(
        feature_frame=feature_frame.loc[historical_predictions.index],
        control_support_frame=control_support,
        historical_predictions=historical_predictions,
        feature_names=feature_names,
        display_metadata=display_metadata,
    )
    write_json(output_dir / "projection_seed.json", projection_seed)

    log("copying static model and endpoint assets")
    copy_static_assets(output_dir)

    log(f"asset bundle ready in {output_dir.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

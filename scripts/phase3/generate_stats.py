import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "Final"
OUTPUT_FILE = REPO_ROOT / "frontend" / "computed_stats.json"

FILE_TO_KEY = {
    # Sliders
    "azpop_monthly.csv": {
        "json_key": "population",
        "column": "population",
    },
    "irrigation_monthly.csv": {
        "json_key": "irrigation_total_withdrawal_mgd",
        "column": "irrigation_total_withdrawal_mgd",
    },
    "public_supply_monthly.csv": {
        "json_key": "public_supply_groundwater_mgd",
        "column": "public_supply_groundwater_mgd",
    },
    "urbanization_monthly.csv": {
        "json_key": "impervious_pct",
        "column": "impervious_pct",
    },
    "lake_mead_monthly.csv": {
        "json_key": "mead_pool_elevation",
        "column": "mead_pool_elevation",
    },
    "precipitation_monthly.csv": {
        "json_key": "precipitation_mm_day",
        "column": "precipitation_mm_day",
    },
    "water_stress_monthly.csv": {
        "json_key": "usdm_dsci",
        "column": "usdm_dsci",
    },
    "nclimdiv_monthly.csv": {
        # THE drought slider. PDSI is signed (- dry, + wet), runs back to 1895, and is
        # the
        # exact input for the two strongest models (surface water, wildlife).
        # usdm_dsci is
        # still a slider stat because NDVI/GRACE/wildfire use it, but the frontend no
        # longer
        # exposes it directly — it derives it from PDSI. See DROUGHT_PDSI_TO_DSCI in
        # frontend/models.js. Zero is "normal", a real reading, so it must not be
        # dropped.
        "json_key": "nclimdiv_pdsi",
        "column": "nclimdiv_pdsi",
        "zero_is_data": True,
    },
    "temperature_monthly.csv": {
        "json_key": "temperature_2m_c",
        "column": "temperature_2m_c",
    },
    # Outputs
    "grace_monthly.csv": {
        "json_key": "grace",
        "column": "grace_groundwater_anomaly",
    },
    "ndvi_monthly.csv": {
        "json_key": "ndvi",
        "column": "ndvi",
    },
    "groundwater_levels_monthly.csv": {
        "json_key": "groundwater",
        # The per-well anomaly, not the raw roster mean — this is what the model
        # predicts. Positive = deeper than normal = less groundwater. A zero here is
        # "exactly normal", a real and meaningful reading, so it must not be dropped.
        "column": "depth_to_water_anomaly_ft",
        "zero_is_data": True,
    },
    "water_surface_monthly.csv": {
        "json_key": "surface_water",
        # Per-gage log anomaly. Zero = normal flow, a real reading.
        "column": "discharge_log_anomaly",
        "zero_is_data": True,
    },
    "wildfire_monthly.csv": {
        "json_key": "wildfire",
        "column": "wildfire_risk_index",
        # A zero here is a real observation — 61% of months have no large fire.
        # Dropping zeros would put the frontend's resting wildfire risk at 0.41
        # (the median of fire months only) instead of 0.0.
        "zero_is_data": True,
    },
    "wildlife_annual.csv": {
        "json_key": "wildlife",
        # The per-route anomaly, not the old min-max index. The old one was a sum over
        # whichever routes were surveyed, so it tracked survey effort (r = +0.94 with
        # route_count) rather than birds. Zero here means "an average year", a real and
        # meaningful reading, so it must not be dropped.
        "column": "abundance_anomaly",
        "zero_is_data": True,
    },
}

SLIDER_KEYS = {
    "population",
    "irrigation_total_withdrawal_mgd",
    "public_supply_groundwater_mgd",
    "impervious_pct",
    "mead_pool_elevation",
    "precipitation_mm_day",
    "temperature_2m_c",
    "usdm_dsci",
    "nclimdiv_pdsi",
}

OUTPUT_KEYS = {
    "grace",
    "ndvi",
    "groundwater",
    "surface_water",
    "wildfire",
    "wildlife",
}


def compute_stats(values: pd.DataFrame, zero_is_data: bool = False) -> dict[str, float]:
    """
    Return 5th, 50th, and 95th percentiles after removing NaNs.

    Zeros are dropped by default: for most variables here (population, discharge,
    withdrawal) a zero is a missing reading rather than a measurement. Set
    `zero_is_data` for series where zero is a real observation — wildfire, where
    a month with no large fire is the single most common outcome.
    """
    values = values.dropna()
    if not zero_is_data:
        values = values[values != 0]

    if values.empty:
        raise ValueError("No values left after filtering.")

    return {
        "min": round(float(np.percentile(values, 5)), 4),
        "median": round(float(np.percentile(values, 50)), 4),
        "max": round(float(np.percentile(values, 95)), 4),
    }


def main() -> None:
    slider_stats = {}
    output_stats = {}

    for filename, config in FILE_TO_KEY.items():
        filepath = DATA_PATH / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Missing file: {filepath}")

        df = pd.read_csv(filepath)

        key = config["json_key"]
        column = config["column"]

        if column not in df.columns:
            raise ValueError(
                f"{filename} does not contain expected column '{column}'. "
                f"Available columns: {list(df.columns)}"
            )

        stats = compute_stats(df[column], config.get("zero_is_data", False))

        if key in SLIDER_KEYS:
            slider_stats[key] = {
                "min": stats["min"],
                "max": stats["max"],
                "default": stats["median"],
            }
        elif key in OUTPUT_KEYS:
            output_stats[key] = {
                "min": stats["min"],
                "max": stats["max"],
                "baseline": stats["median"],
            }
        else:
            raise KeyError(f"Unknown key '{key}' in FILE_TO_KEY.")

    missing_sliders = SLIDER_KEYS - slider_stats.keys()
    missing_outputs = OUTPUT_KEYS - output_stats.keys()

    if missing_sliders:
        raise RuntimeError(f"Missing slider stats for: {sorted(missing_sliders)}")

    if missing_outputs:
        raise RuntimeError(f"Missing output stats for: {sorted(missing_outputs)}")

    result = {
        "SLIDER_STATS": slider_stats,
        "OUTPUT_STATS": output_stats,
    }

    with OUTPUT_FILE.open("w") as f:
        json.dump(result, f, indent=2)

    print(f"Stats written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

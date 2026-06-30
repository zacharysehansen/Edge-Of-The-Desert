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
        "column": "depth_to_water_ft_mean",
    },
    "water_surface_monthly.csv": {
        "json_key": "surface_water",
        "column": "discharge_cfs_mean",
    },
    "wildfire_monthly.csv": {
        "json_key": "wildfire",
        "column": "wildfire_risk_index",
    },
    "wildlife_annual.csv": {
        "json_key": "wildlife",
        "column": "abundance_index",
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
}

OUTPUT_KEYS = {
    "grace",
    "ndvi",
    "groundwater",
    "surface_water",
    "wildfire",
    "wildlife",
}


def compute_stats(values: pd.DataFrame) -> dict[str, float]:
    """Return 5th, 50th, and 95th percentiles after removing NaNs and zeros."""
    values = values.dropna()
    values = values[values != 0]

    if values.empty:
        raise ValueError("No non-zero values found.")

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

        stats = compute_stats(df[column])

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

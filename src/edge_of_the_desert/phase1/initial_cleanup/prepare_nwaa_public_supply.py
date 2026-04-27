from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FINAL_DIR = REPO_ROOT / "data" / "Final"
LONG_OUTPUT = PROCESSED_DIR / "nwaa_public_supply_az_huc12_monthly.csv"
MONTHLY_OUTPUT = FINAL_DIR / "nwaa_public_supply_az_monthly.csv"
SUMMARY_OUTPUT = PROCESSED_DIR / "nwaa_public_supply_az_summary.json"
EXPECTED_COLUMNS = ["huc12_id", "year_month", "pswdgw_mgd"]


def prepare_nwaa_public_supply() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(RAW_DIR.glob("nwaa_data_*_of_6.csv"))
    if not shard_paths:
        raise SystemExit("No NWAA shard files were found in data/raw.")

    frames = []
    for path in shard_paths:
        frame = pd.read_csv(path)
        columns = list(frame.columns)
        if columns != EXPECTED_COLUMNS:
            raise SystemExit(
                f"{path.name} has columns {columns}, expected {EXPECTED_COLUMNS}"
            )
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["huc12_id"] = combined["huc12_id"].astype(str)
    combined["year_month"] = pd.to_datetime(combined["year_month"] + "-01").dt.strftime("%Y-%m")
    combined["pswdgw_mgd"] = pd.to_numeric(combined["pswdgw_mgd"], errors="raise")

    duplicate_count = int(combined.duplicated(["huc12_id", "year_month"]).sum())
    if duplicate_count:
        raise SystemExit(f"Found {duplicate_count} duplicate huc12_id/year_month pairs.")

    combined = combined.sort_values(["huc12_id", "year_month"]).reset_index(drop=True)
    combined.to_csv(LONG_OUTPUT, index=False)

    monthly = (
        combined.groupby("year_month", as_index=False)
        .agg(
            public_supply_groundwater_mgd=("pswdgw_mgd", "sum"),
            huc12_count=("huc12_id", "nunique"),
        )
        .sort_values("year_month")
        .reset_index(drop=True)
    )
    monthly_dates = pd.to_datetime(monthly["year_month"] + "-01")
    days_in_month = monthly_dates.dt.days_in_month
    monthly["public_supply_groundwater_gallons_per_day"] = (
        monthly["public_supply_groundwater_mgd"] * 1_000_000.0
    )
    monthly["public_supply_groundwater_acre_feet_month"] = (
        monthly["public_supply_groundwater_gallons_per_day"] * days_in_month / 325851.429
    )
    monthly = monthly[
        [
            "year_month",
            "public_supply_groundwater_mgd",
            "public_supply_groundwater_gallons_per_day",
            "public_supply_groundwater_acre_feet_month",
            "huc12_count",
        ]
    ]
    monthly.to_csv(MONTHLY_OUTPUT, index=False)

    summary = {
        "source_files": [path.name for path in shard_paths],
        "row_count_long": int(len(combined)),
        "row_count_monthly": int(len(monthly)),
        "year_month_min": str(monthly["year_month"].min()),
        "year_month_max": str(monthly["year_month"].max()),
        "unique_huc12_count": int(combined["huc12_id"].nunique()),
        "duplicate_huc12_year_month_pairs": duplicate_count,
        "min_public_supply_groundwater_mgd": float(monthly["public_supply_groundwater_mgd"].min()),
        "max_public_supply_groundwater_mgd": float(monthly["public_supply_groundwater_mgd"].max()),
        "zero_rows_in_long": int((combined["pswdgw_mgd"] == 0).sum()),
        "positive_rows_in_long": int((combined["pswdgw_mgd"] > 0).sum()),
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"Wrote {LONG_OUTPUT.relative_to(REPO_ROOT)}")
    print(f"Wrote {MONTHLY_OUTPUT.relative_to(REPO_ROOT)}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0

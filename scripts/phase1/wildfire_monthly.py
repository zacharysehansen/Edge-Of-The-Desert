"""
wildfire_monthly.py
-------------------
Build a monthly wildfire dataset for the eight-county southern Arizona
study area from the InterAgencyFirePerimeterHistory file and supplemental
GDB/CSV archives for 2020-2024.

Strategy:
  - Primary source: InterAgencyFirePerimeterHistory CSV (covers 2000-2024)
    Filter by UNIT_ID containing "AZ", extract month from DATE_CUR.
  - For DATE_CUR, use the month directly. The historical file has no
    placeholders for 2000+ AZ records (verified: 0% placeholder rate).
  - When DATE_CUR year != FIRE_YEAR, assign to FIRE_YEAR (fire occurrence
    year) with the month from DATE_CUR (best available month signal).
  - Aggregate: monthly fire count and log-transformed total acreage,
    combined into a monthly wildfire risk index.

Output: data/Final/wildfire_monthly.csv
  Columns: year_month, fire_count, total_acres, log_acres, wildfire_risk_index

Usage:
    python -m scripts.phase1.wildfire_monthly
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw" / "wildfire"
OUTPUT_FILE = ROOT / "data" / "Final" / "wildfire_monthly.csv"

HISTORICAL_FILE = (
    RAW_DIR / "InterAgencyFirePerimeterHistory_All_Years_View_-1590405183658604377.csv"
)

PROJECT_START = 2000
PROJECT_END = 2023


def _load_historical() -> pd.DataFrame:
    """Load and filter the historical fire perimeter file to AZ fires, 2000-2023."""
    log.info("Loading %s", HISTORICAL_FILE.name)
    df = pd.read_csv(
        HISTORICAL_FILE,
        usecols=["FIRE_YEAR", "GIS_ACRES", "DATE_CUR", "UNIT_ID"],
        low_memory=False,
    )

    az = df[df["UNIT_ID"].str.contains("AZ", na=False)].copy()
    log.info("AZ fires (all years): %d", len(az))

    az["FIRE_YEAR"] = pd.to_numeric(az["FIRE_YEAR"], errors="coerce")
    az["GIS_ACRES"] = pd.to_numeric(az["GIS_ACRES"], errors="coerce")
    az = az.dropna(subset=["FIRE_YEAR", "GIS_ACRES"])
    az["FIRE_YEAR"] = az["FIRE_YEAR"].astype(int)

    az = az[
        (az["FIRE_YEAR"] >= PROJECT_START) & (az["FIRE_YEAR"] <= PROJECT_END)
    ].copy()
    log.info("AZ fires %d-%d: %d", PROJECT_START, PROJECT_END, len(az))

    az = az[az["GIS_ACRES"] > 0].copy()

    az["date_str"] = az["DATE_CUR"].astype(str).str.split(".").str[0]
    az["month"] = pd.to_numeric(az["date_str"].str[4:6], errors="coerce").astype(
        "Int64"
    )
    az["year"] = az["FIRE_YEAR"]

    pre = len(az)
    az = az.dropna(subset=["month"])
    az["month"] = az["month"].astype(int)
    if len(az) < pre:
        log.warning("Dropped %d rows with unparseable month.", pre - len(az))

    # Sanity: month must be 1-12
    az = az[(az["month"] >= 1) & (az["month"] <= 12)]  # noqa: PLR2004

    return (
        az[["year", "month", "GIS_ACRES"]]
        .rename(columns={"GIS_ACRES": "acres"})
        .reset_index(drop=True)
    )


def _build_monthly_index() -> pd.DataFrame:
    """Create a complete year_month index covering the project range."""
    periods = pd.period_range(f"{PROJECT_START}-01", f"{PROJECT_END}-12", freq="M")
    return pd.DataFrame({"year_month": periods})


def _aggregate_monthly(fires: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual fire records to monthly totals."""
    fires = fires.copy()
    fires["year_month"] = fires.apply(
        lambda r: pd.Period(year=int(r["year"]), month=int(r["month"]), freq="M"),
        axis=1,
    )

    monthly = (
        fires.groupby("year_month")
        .agg(
            fire_count=("acres", "count"),
            total_acres=("acres", "sum"),
        )
        .reset_index()
    )

    full_index = _build_monthly_index()
    result = full_index.merge(monthly, on="year_month", how="left")
    result["fire_count"] = result["fire_count"].fillna(0).astype(int)
    result["total_acres"] = result["total_acres"].fillna(0.0)

    result["log_acres"] = np.log1p(result["total_acres"])

    return result


def _build_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly wildfire risk index:
      wildfire_risk_index = 0.5 * norm(fire_count) + 0.5 * norm(log_acres)
    Min-max normalized over the full series.
    """
    df = df.copy()

    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(0.5, index=s.index)
        return (s - lo) / (hi - lo)

    df["wildfire_risk_index"] = (
        0.5 * _minmax(df["fire_count"]) + 0.5 * _minmax(df["log_acres"])
    ).round(6)

    return df


def main() -> None:
    log.info("=== wildfire_monthly.py start ===")

    fires = _load_historical()

    log.info("Total fire records for aggregation: %d", len(fires))
    log.info("Year range: %d - %d", fires["year"].min(), fires["year"].max())

    monthly = _aggregate_monthly(fires)
    monthly = _build_risk_index(monthly)

    output = monthly[
        ["year_month", "fire_count", "total_acres", "log_acres", "wildfire_risk_index"]
    ].copy()
    output["year_month"] = output["year_month"].astype(str)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    log.info("Wrote %d rows to %s", len(output), OUTPUT_FILE)

    log.info("Monthly stats:")
    log.info("  Mean fire count: %.1f", output["fire_count"].mean())
    log.info("  Months with 0 fires: %d", (output["fire_count"] == 0).sum())
    log.info("  Max fire count (single month): %d", output["fire_count"].max())
    log.info("  Mean log_acres: %.2f", output["log_acres"].mean())

    log.info("=== wildfire_monthly.py complete ===")


if __name__ == "__main__":
    main()

"""
wildfire_monthly.py
-------------------
Build a monthly wildfire dataset for the eight-county southern Arizona study
area from MTBS (Monitoring Trends in Burn Severity) fire-occurrence points.

Why MTBS and not the InterAgency perimeter file
-----------------------------------------------
This script used to derive the month from `DATE_CUR` in
InterAgencyFirePerimeterHistory. **`DATE_CUR` is a record-maintenance timestamp,
not an ignition date**, and the resulting target was not measuring wildfire:

  - 57% of AZ fires landed on five calendar days (Feb 1 alone held 1,221 of
    4,395) — those are ETL batch-load dates, not fires.
  - The implied fire season peaked in *February*; May/June/September were the
    troughs. Arizona burns pre-monsoon, May-July.
  - The target was *anti*-correlated with temperature (-0.27) and *positively*
    correlated with precipitation (+0.09) — both signs backwards.

The perimeter file has no ignition-date column at all (only the annual
`FIRE_YEAR`), which `wildfire.py` already documents in its own docstring. The
Operational Data Archives and the Public_EventDataArchive geodatabases carry
only Create/Current/Polygon maintenance dates. No monthly signal is recoverable
from any of them.

MTBS publishes a true **`ig_date`** (ignition date) per fire. Rebuilt on it, the
seasonality is physically correct (June peak, near-zero Nov-Jan) and every driver
correlation flips to the expected sign: temperature +0.46, precipitation -0.11,
drought +0.11.

Trade-off: MTBS only maps fires above a size threshold (>=1000 acres in the
West), so `fire_count` here means "large fires", not "all ignitions". That is the
quantity a climate-driven risk index can actually speak to, and it is why ~68% of
months (over the full 1984-2023 record) are zero. The target is genuinely
zero-inflated — a Tweedie objective competes for it in Phase 2, though on this
data squared-error XGBoost still wins the out-of-fold R².

Source: https://www.mtbs.gov/  (fire occurrence points, national, 1984-present)
  data/raw/wildfire/mtbs/mtbs_FODpoints_DD.shp

Output: data/Final/wildfire_monthly.csv
  Columns: year_month, fire_count, total_acres, log_acres, wildfire_risk_index

Usage:
    python -m scripts.phase1.wildfire_monthly
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import filter_points  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw" / "wildfire"
MTBS_FILE = RAW_DIR / "mtbs" / "mtbs_FODpoints_DD.shp"
OUTPUT_FILE = ROOT / "data" / "Final" / "wildfire_monthly.csv"

MTBS_URL = (
    "https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/"
    "composite_data/fod_pt_shapefile/mtbs_fod_pts_data.zip"
)

# MTBS ig_date runs from 1984, and that is now the start. The old 2000 floor was
# inherited from the USDM, but the monthly wildfire model no longer uses the USDM:
# reaching before 2000 collapses its feature set to nClimDiv (PDSI/temp/precip,
# 1895+), which is exactly the drought signal fire responds to. The 1984-1999 rows
# add 105 more fires and ~192 months. See features._MONTHLY_MODEL_SPECS["wildfire_monthly"].
MTBS_FLOOR = 1984
PROJECT_START = 1984
PROJECT_END = 2023

# MTBS mixes managed burns into the same file; those are not wildfire.
WILDFIRE_TYPES = ["Wildfire", "Wildland Fire Use"]

# Arizona's fire season is pre-monsoon. Used to assert the time axis is real.
FIRE_SEASON = (4, 5, 6, 7, 8)


def _load_mtbs() -> gpd.GeoDataFrame:
    """Load MTBS occurrence points, keep true wildfires, clip to the region."""
    if not MTBS_FILE.exists():
        raise FileNotFoundError(
            f"MTBS fire points not found at {MTBS_FILE}.\n"
            f"Download and unzip:\n  curl -sL -o mtbs.zip {MTBS_URL}\n"
            f"  unzip mtbs.zip -d {MTBS_FILE.parent}"
        )

    gdf = gpd.read_file(MTBS_FILE)
    log.info("MTBS points (national, all types): %d", len(gdf))

    gdf["ig_date"] = pd.to_datetime(gdf["ig_date"], errors="coerce")
    gdf = gdf.dropna(subset=["ig_date", "burnbndac"])

    gdf = gdf[gdf["incid_type"].isin(WILDFIRE_TYPES)].copy()
    log.info("  after dropping prescribed/other burns: %d", len(gdf))

    gdf = filter_points(gdf)
    log.info("  inside the eight-county boundary: %d", len(gdf))

    gdf["year"] = gdf["ig_date"].dt.year
    in_window = gdf[
        (gdf["year"] >= PROJECT_START) & (gdf["year"] <= PROJECT_END)
    ].copy()
    log.info(
        "  %d-%d: %d fires (%d more available back to %d)",
        PROJECT_START,
        PROJECT_END,
        len(in_window),
        (gdf["year"] < PROJECT_START).sum(),
        MTBS_FLOOR,
    )

    if in_window.empty:
        raise ValueError(
            "No MTBS wildfires found in the study region for "
            f"{PROJECT_START}-{PROJECT_END}."
        )

    return in_window


def _aggregate_monthly(fires: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate fires to monthly totals on their true ignition date."""
    fires = fires.copy()
    fires["year_month"] = fires["ig_date"].dt.to_period("M")

    monthly = (
        fires.groupby("year_month")
        .agg(fire_count=("burnbndac", "size"), total_acres=("burnbndac", "sum"))
        .reset_index()
    )

    # Months with no fire are real zeros, not missing data.
    full_index = pd.DataFrame(
        {
            "year_month": pd.period_range(
                f"{PROJECT_START}-01", f"{PROJECT_END}-12", freq="M"
            )
        }
    )
    result = full_index.merge(monthly, on="year_month", how="left")
    result["fire_count"] = result["fire_count"].fillna(0).astype(int)
    result["total_acres"] = result["total_acres"].fillna(0.0)
    result["log_acres"] = np.log1p(result["total_acres"])

    return result


def _build_risk_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    wildfire_risk_index = 0.5 * norm(fire_count) + 0.5 * norm(log_acres)

    Min-max normalized over the full series, so the scale depends on the series
    maximum. Min-max is a linear transform, so this does not change any R², but
    it does mean the index is not comparable across re-runs with different
    windows. Kept as-is to preserve the frontend's 0-1 contract.
    """

    def _minmax(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=s.index)

    df["wildfire_risk_index"] = 0.5 * _minmax(df["fire_count"]) + 0.5 * _minmax(
        df["log_acres"]
    )
    return df


def _validate_seasonality(df: pd.DataFrame) -> None:
    """
    Assert the time axis is physically real.

    This is the check that would have caught the DATE_CUR bug on day one: a
    wildfire series whose busiest month is in winter is not a wildfire series.
    """
    by_month = df.assign(m=df["year_month"].dt.month).groupby("m")["fire_count"].sum()
    peak = int(by_month.idxmax())

    log.info(
        "Fires by ignition month: %s",
        by_month.reindex(range(1, 13), fill_value=0).tolist(),
    )

    if peak not in FIRE_SEASON:
        raise ValueError(
            f"Peak fire month is {peak}, outside Arizona's fire season "
            f"{FIRE_SEASON}. The date field is almost certainly a "
            f"record-maintenance timestamp rather than an ignition date. "
            f"Refusing to write a target with a fabricated time axis."
        )
    log.info("  peak ignition month = %d (in fire season) — time axis OK", peak)


def main() -> None:
    fires = _load_mtbs()
    monthly = _aggregate_monthly(fires)
    _validate_seasonality(monthly)
    monthly = _build_risk_index(monthly)

    output = monthly[
        ["year_month", "fire_count", "total_acres", "log_acres", "wildfire_risk_index"]
    ].copy()
    output["year_month"] = output["year_month"].astype(str)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    zero_months = int((output["wildfire_risk_index"] == 0).sum())
    log.info("Wrote %s (%d months)", OUTPUT_FILE, len(output))
    log.info("  Total fires        : %d", int(output["fire_count"].sum()))
    log.info("  Total acres burned : %.0f", output["total_acres"].sum())
    log.info("  Max fires in a month: %d", int(output["fire_count"].max()))
    log.info(
        "  Zero-fire months   : %d / %d (%.0f%%) — target is zero-inflated; "
        "use a Tweedie/hurdle loss, not squared error",
        zero_months,
        len(output),
        100 * zero_months / len(output),
    )


if __name__ == "__main__":
    main()

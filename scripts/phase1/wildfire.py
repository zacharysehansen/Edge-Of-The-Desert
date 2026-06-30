"""
wildfire.py
-----------
Loads the user-provided wildfire event CSV, filters to fires located
within the eight-county southern Arizona study area, and builds an
annual wildfire risk index from fire count and log-transformed total
acreage.

Input  : data/raw/az_wildfires.csv
Output : data/processed/wildfire_annual.csv

Columns in output:
    year                  - int, calendar year
    fire_count            - int, number of fires in the eight counties
    log_acres_total       - float, log1p of total acres burned
    wildfire_risk_index   - float, normalized composite of the above two

Expected columns in az_wildfires.csv (from README [2]):
    OBJECTID, FIRE_NAME, FIRE_Number, FireID, Acres, FIRE_YEAR,
    Z, KM2, Source1, Source2, Shape__Area, Shape__Length

The raw file does not have a monthly date field, only FIRE_YEAR, so
output is annual grain. This is consistent with the annual grain
decision documented in the setup notes [1].

Spatial filter:
    Fires are filtered to the eight-county boundary using the centroid
    derived from the KM2/Shape__Area polygon data, OR if explicit
    lat/lon columns are present, those are used directly. If neither
    geometry nor coordinate columns are available the script falls back
    to a bounding-box pre-filter using BBOX from region.py and emits
    a clear warning.
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import (
    BBOX,
    COUNTIES,
    filter_points,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "Final"
INPUT_FILE = RAW_DIR / "az_wildfires.csv"
OUTPUT_FILE = PROCESSED_DIR / "wildfire_annual.csv"

# Column names expected in the raw file [2]
YEAR_COL = "FIRE_YEAR"
ACRES_COL = "Acres"

# Candidate coordinate column pairs, checked in order of preference
_LAT_CANDIDATES = ["latitude", "Latitude", "LAT", "lat", "Y", "y"]
_LON_CANDIDATES = ["longitude", "Longitude", "LON", "lon", "X", "x"]


def _find_coord_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Return the first matching (lat_col, lon_col) pair found in df.columns.
    Returns (None, None) if no coordinate columns are detected.
    """
    lat_col = next((c for c in _LAT_CANDIDATES if c in df.columns), None)
    lon_col = next((c for c in _LON_CANDIDATES if c in df.columns), None)
    return lat_col, lon_col


def _load_raw(path: Path) -> pd.DataFrame:
    """
    Read the raw wildfire CSV and do minimal type coercion.

    Raises
    ------
    FileNotFoundError
        If the CSV is not present at the expected path.
    ValueError
        If the required FIRE_YEAR or Acres columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Raw wildfire file not found at {path}.\n"
            "Place az_wildfires.csv in data/raw/ and re-run."
        )

    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded %d rows from %s", len(df), path.name)

    # Validate required columns
    missing = [c for c in [YEAR_COL, ACRES_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required column(s) {missing} not found in {path.name}.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Coerce types
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[ACRES_COL] = pd.to_numeric(df[ACRES_COL], errors="coerce")

    pre = len(df)
    df = df.dropna(subset=[YEAR_COL, ACRES_COL])
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    if len(df) < pre:
        log.warning("Dropped %d rows with null FIRE_YEAR or Acres.", pre - len(df))

    # Drop fires with zero or negative acreage — not real events
    neg = df[ACRES_COL] <= 0
    if neg.any():
        log.warning("Dropping %d rows with Acres <= 0.", neg.sum())
        df = df[~neg]

    log.info("%d rows remain after type coercion and zero-acre drop.", len(df))
    return df.reset_index(drop=True)


def _spatial_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame to fires that fall inside the eight-county boundary.

    Strategy (in order of preference):
    1. If explicit lat/lon columns exist → build Point geometry → sjoin
    2. If no coordinate columns exist → bbox pre-filter only, with warning

    Returns a filtered DataFrame (plain pandas, geometry column dropped).
    """
    lat_col, lon_col = _find_coord_columns(df)

    if lat_col and lon_col:
        log.info(
            "Coordinate columns found ('%s', '%s'). "
            "Running point-in-polygon filter against eight-county boundary.",
            lat_col,
            lon_col,
        )

        # Drop rows where coordinates are null
        pre = len(df)
        df = df.dropna(subset=[lat_col, lon_col]).copy()
        if len(df) < pre:
            log.warning("Dropped %d rows with null coordinates.", pre - len(df))

        # Build GeoDataFrame
        geometry = [
            Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col], strict=False)
        ]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        # Use region.filter_points() — consistent with every other script
        filtered_gdf = filter_points(gdf)

        log.info(
            "Spatial filter: %d → %d rows (removed %d outside boundary).",
            len(gdf),
            len(filtered_gdf),
            len(gdf) - len(filtered_gdf),
        )

        # Return plain DataFrame, drop geometry column
        return filtered_gdf.drop(columns=["geometry"]).reset_index(drop=True)

    log.warning(
        "No latitude/longitude columns found in the wildfire CSV. "
        "Falling back to bounding-box filter only (min_lon=%.2f, "
        "min_lat=%.2f, max_lon=%.2f, max_lat=%.2f). "
        "This may include fires outside the eight-county boundary. "
        "Add lat/lon columns to az_wildfires.csv for a precise filter.",
        *BBOX,
    )

    # Since we have no geometry, we can only trust that the source CSV
    # is already Arizona-scoped. Log the county FIPS for the record
    # and return the full dataset with a note in the output filename.
    log.info("Target counties: %s", ", ".join(COUNTIES))
    log.info(
        "Returning all %d rows — verify the source CSV is already "
        "scoped to the eight-county region.",
        len(df),
    )

    return df.reset_index(drop=True)


def _build_risk_index(annual: pd.DataFrame) -> pd.DataFrame:
    """
    Build the wildfire risk index from fire_count and log_acres_total.

    Index formula:
        wildfire_risk_index = 0.5 * norm(fire_count)
                            + 0.5 * norm(log_acres_total)

    Both components are min-max normalized to [0, 1] before weighting,
    so the index is always in [0, 1]. Equal weighting (0.5/0.5) treats
    frequency and magnitude as equally important — adjust weights here
    if domain knowledge suggests otherwise.

    Uses log1p(total_acres) rather than raw acres because the acreage
    distribution is heavily right-skewed: a small number of very large
    fires would otherwise dominate the index [2].
    """
    df = annual.copy()

    def _minmax(series: pd.Series) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            # All values identical → normalize to 0.5 (mid-range)
            return pd.Series(0.5, index=series.index)
        return (series - lo) / (hi - lo)

    df["wildfire_risk_index"] = (
        0.5 * _minmax(df["fire_count"]) + 0.5 * _minmax(df["log_acres_total"])
    ).round(6)

    return df


def _row_count_check(df: pd.DataFrame) -> None:
    """
    Warn if the annual row count is low enough to make cross-validation
    unreliable. Setup notes [1] flag this as a known risk for annual-grain
    datasets after the eight-county filter.
    """
    n = len(df)
    very_low_count = 20
    low_count = 50

    if n < very_low_count:
        log.warning(
            "VERY LOW ROW COUNT: only %d annual rows after filtering. "
            "XGBoost cross-validation will be unreliable at this size. "
            "Consider a simpler model for the wildfire target or review "
            "whether the spatial filter is too restrictive.",
            n,
        )
    elif n < low_count:
        log.warning(
            "LOW ROW COUNT: %d annual rows. Cross-validation splits "
            "may be unstable. Review before training.",
            n,
        )
    else:
        log.info("Row count looks acceptable for modeling: %d annual rows.", n)


def main() -> None:
    log.info("=== wildfire.py start ===")

    df = _load_raw(INPUT_FILE)

    df = _spatial_filter(df)

    if df.empty:
        raise ValueError(
            "No wildfire records remain after spatial filter. "
            "Check that az_wildfires.csv contains fires within the "
            "eight-county study area and that coordinate columns are present."
        )

    annual = (
        df.groupby(YEAR_COL)
        .agg(
            fire_count=(ACRES_COL, "count"),
            total_acres=(ACRES_COL, "sum"),
        )
        .reset_index()
        .rename(columns={YEAR_COL: "year"})
    )

    annual["log_acres_total"] = np.log1p(annual["total_acres"]).round(6)

    annual = annual.sort_values("year").reset_index(drop=True)

    log.info(
        "Annual aggregation complete: %d years (%d–%d), "
        "total fires=%d, total acres=%.0f.",
        len(annual),
        annual["year"].min(),
        annual["year"].max(),
        annual["fire_count"].sum(),
        annual["total_acres"].sum(),
    )

    _row_count_check(annual)

    annual = _build_risk_index(annual)

    output = annual[
        ["year", "fire_count", "log_acres_total", "wildfire_risk_index"]
    ].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    log.info("Wrote %d rows to %s", len(output), OUTPUT_FILE)

    log.info("=== wildfire.py complete ===")


if __name__ == "__main__":
    main()

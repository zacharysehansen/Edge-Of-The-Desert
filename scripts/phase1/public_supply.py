"""
public_supply.py
----------------
Loads the HUC12-level public supply groundwater withdrawal source
matrix, filters to HUC12s within the eight-county boundary using
region.py, and re-aggregates to a monthly regional total.

Same pattern as irrigation.py, loading the NWAA HUC12 shards instead. [1]

Input  : data/raw/ps_huc12_tot_az_2000_2020.csv
Output : data/processed/public_supply_monthly.csv

Columns in output:
    year_month                       - str, format YYYY-MM
    public_supply_groundwater_mgd    - float, regional total (million
                                       gallons per day)

Input file structure [3]:
    Wide-format with HUC12 codes as columns. Each row represents one
    month (Year + Month columns). Values are groundwater withdrawal
    rates in MGD for each HUC12 watershed.

    Columns: Year, Month, HUC12_1, HUC12_2, ..., HUC12_N

Spatial filter approach:
    Identical to irrigation.py — HUC12 watershed centroids are matched
    against the eight-county boundary. Falls back gracefully if the
    WBD shapefile or crosswalk is not available.

Date range: 2000-01 to 2020-12 [3]
    The public supply source data covers 2000-2020. This is shorter
    than the full project range (2000-2023) so downstream scripts will
    need to handle the gap.
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import (
    COUNTY_FIPS,
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
INPUT_FILE = RAW_DIR / "PS_HUC12_Tot_2000_2020.csv"
OUTPUT_FILE = PROCESSED_DIR / "public_supply_monthly.csv"

# Optional: HUC12 shapefile for spatial filtering (shared with irrigation.py)
HUC12_SHAPEFILE = RAW_DIR / "wbd" / "WBDHU12.shp"

# Optional: HUC12-to-county crosswalk (shared with irrigation.py)
HUC12_COUNTY_CROSSWALK = RAW_DIR / "wbd" / "huc12_county_crosswalk.csv"

YEAR_COL = "Year"
MONTH_COL = "Month"


def _load_raw(path: Path) -> pd.DataFrame:
    """
    Load the wide-format HUC12 public supply CSV.

    Expected structure:
        Year, Month, HUC12_col_1, HUC12_col_2, ..., HUC12_col_N

    Raises
    ------
    FileNotFoundError
        If the input CSV is not at the expected path.
    ValueError
        If Year or Month columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Public supply source file not found at {path}.\n"
            "Place ps_huc12_tot_az_2000_2020.csv in data/raw/ and re-run."
        )

    df = pd.read_csv(path, low_memory=False)
    log.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path.name)

    # Validate required columns
    missing = [c for c in [YEAR_COL, MONTH_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Required column(s) {missing} not found in {path.name}.\n"
            f"Available columns (first 10): {df.columns.tolist()[:10]}"
        )

    return df


def _identify_huc12_columns(df: pd.DataFrame) -> list[str]:
    """
    Identify which columns in the DataFrame are HUC12 watershed columns
    vs. metadata columns (Year, Month, etc.).

    HUC12 codes are 12-digit numeric strings. Columns that are purely
    numeric (or can be interpreted as HUC12 codes) are treated as
    watershed data columns.

    Returns
    -------
    List of column names that represent HUC12 watersheds.
    """
    metadata_cols = {YEAR_COL, MONTH_COL}
    huc12_cols = []

    for col in df.columns:
        if col in metadata_cols:
            continue
        # Check if column name looks like a HUC12 code (12 digits)
        clean = str(col).strip().replace(".", "").replace("-", "")
        clean_digit_length = 8
        if (
            clean.isdigit()
            and len(clean) >= clean_digit_length
            or col not in metadata_cols
            and pd.to_numeric(df[col], errors="coerce").notna().sum() > 0
        ):
            huc12_cols.append(col)

    log.info("Identified %d HUC12 watershed columns.", len(huc12_cols))

    if len(huc12_cols) == 0:
        raise ValueError(
            "Could not identify any HUC12 columns in the input file.\n"
            "Non-metadata columns: "
            f"{[c for c in df.columns if c not in metadata_cols][:10]}"
        )

    return huc12_cols


def _filter_huc12s_with_shapefile(
    huc12_cols: list[str],
) -> list[str] | None:
    """
    Use the WBD HUC12 shapefile to determine which HUC12 watersheds
    have centroids inside the eight-county boundary.

    Returns
    -------
    List of HUC12 column names inside the study area, or None if
    the shapefile is not available.
    """
    if not HUC12_SHAPEFILE.exists():
        return None

    log.info("Loading HUC12 shapefile from %s...", HUC12_SHAPEFILE)
    huc12_gdf = gpd.read_file(HUC12_SHAPEFILE)

    # Find the HUC12 code column
    huc_col_candidates = ["HUC12", "huc12", "HUC_12", "TOHUC"]
    huc_col = next((c for c in huc_col_candidates if c in huc12_gdf.columns), None)
    if huc_col is None:
        log.warning(
            "Cannot find HUC12 code column in shapefile. " "Columns: %s",
            huc12_gdf.columns.tolist()[:10],
        )
        return None

    # Compute centroids
    huc12_gdf = huc12_gdf.to_crs("EPSG:4326")
    huc12_gdf["centroid"] = huc12_gdf.geometry.centroid
    centroids = gpd.GeoDataFrame(
        huc12_gdf[[huc_col]],
        geometry=huc12_gdf["centroid"],
        crs="EPSG:4326",
    )

    # Filter centroids to eight-county boundary
    filtered = filter_points(centroids)
    regional_hucs = set(filtered[huc_col].astype(str).tolist())

    # Match against column names in the public supply CSV
    regional_cols = [col for col in huc12_cols if str(col).strip() in regional_hucs]

    log.info(
        "HUC12 shapefile filter: %d total HUC12s, "
        "%d inside boundary, %d matched to public supply columns.",
        len(huc12_gdf),
        len(regional_hucs),
        len(regional_cols),
    )

    return regional_cols if regional_cols else None


def _filter_huc12s_with_crosswalk(
    huc12_cols: list[str],
) -> list[str] | None:
    """
    Use a HUC12-to-county crosswalk table to determine which HUC12s
    fall inside the eight counties.

    Returns
    -------
    List of HUC12 column names in the study area, or None if the
    crosswalk file is not available.
    """
    if not HUC12_COUNTY_CROSSWALK.exists():
        return None

    log.info("Loading HUC12-county crosswalk from %s...", HUC12_COUNTY_CROSSWALK)
    xwalk = pd.read_csv(HUC12_COUNTY_CROSSWALK, dtype=str)

    # Look for HUC12 and county FIPS columns
    huc_col = next(
        (c for c in ["HUC12", "huc12", "HUC_12"] if c in xwalk.columns), None
    )
    fips_col = next(
        (
            c
            for c in ["COUNTY_FIPS", "county_fips", "FIPS", "GEOID"]
            if c in xwalk.columns
        ),
        None,
    )

    if huc_col is None or fips_col is None:
        log.warning("Crosswalk file missing HUC12 or FIPS column.")
        return None

    # Filter to eight counties
    regional_xwalk = xwalk[xwalk[fips_col].isin(COUNTY_FIPS)]
    regional_hucs = set(regional_xwalk[huc_col].astype(str).tolist())

    # Match against public supply columns
    regional_cols = [col for col in huc12_cols if str(col).strip() in regional_hucs]

    log.info(
        "Crosswalk filter: %d HUC12s in eight counties, "
        "%d matched to public supply columns.",
        len(regional_hucs),
        len(regional_cols),
    )

    return regional_cols if regional_cols else None


def _filter_huc12s_bbox_fallback(
    huc12_cols: list[str],
) -> list[str]:
    """
    Fallback: if no shapefile or crosswalk is available, return ALL
    HUC12 columns with a warning. Since the source file is already
    Arizona-scoped [3], this may be acceptable but imprecise.
    """
    log.warning(
        "NO HUC12 SPATIAL FILTER AVAILABLE.\n"
        "  Neither the WBD shapefile (%s) nor the crosswalk file (%s) "
        "was found.\n"
        "  Using ALL %d HUC12 columns from the input file.\n"
        "  This includes watersheds outside the eight-county boundary.\n"
        "  For a precise filter, provide either:\n"
        "    - WBD HUC12 shapefile at %s\n"
        "    - HUC12-to-county crosswalk CSV at %s",
        HUC12_SHAPEFILE,
        HUC12_COUNTY_CROSSWALK,
        len(huc12_cols),
        HUC12_SHAPEFILE,
        HUC12_COUNTY_CROSSWALK,
    )
    return huc12_cols


def _get_regional_huc12_columns(huc12_cols: list[str]) -> list[str]:
    """
    Determine which HUC12 columns fall inside the eight-county region
    using the best available method:
        1. WBD shapefile → centroid point-in-polygon
        2. HUC12-to-county crosswalk table
        3. Fallback: use all columns with warning

    Returns
    -------
    List of HUC12 column names to include in the regional sum.
    """
    # Try shapefile first
    result = _filter_huc12s_with_shapefile(huc12_cols)
    if result:
        return result

    # Try crosswalk
    result = _filter_huc12s_with_crosswalk(huc12_cols)
    if result:
        return result

    # Fallback
    return _filter_huc12s_bbox_fallback(huc12_cols)


def _aggregate_to_monthly(
    df: pd.DataFrame,
    regional_cols: list[str],
) -> pd.DataFrame:
    """
    Sum the regional HUC12 columns per row to produce a monthly
    regional total public supply withdrawal.

    Parameters
    ----------
    df : DataFrame
        The full wide-format input with Year, Month, and HUC12 columns.
    regional_cols : list of str
        HUC12 columns that fall inside the eight-county region.

    Returns
    -------
    DataFrame with columns: year_month, public_supply_groundwater_mgd.
    """
    # Coerce HUC12 columns to numeric
    for col in regional_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Build year_month
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype(int)
    df[MONTH_COL] = pd.to_numeric(df[MONTH_COL], errors="coerce").astype(int)
    df["year_month"] = (
        df[YEAR_COL].astype(str) + "-" + df[MONTH_COL].astype(str).str.zfill(2)
    )

    # Sum across regional HUC12 columns
    df["public_supply_groundwater_mgd"] = df[regional_cols].sum(axis=1).round(4)

    result = (
        df[["year_month", "public_supply_groundwater_mgd"]]
        .sort_values("year_month")
        .reset_index(drop=True)
    )

    log.info(
        "Monthly aggregation: %d rows (%s to %s), "
        "summing %d HUC12 columns, "
        "mean withdrawal=%.2f MGD.",
        len(result),
        result["year_month"].iloc[0],
        result["year_month"].iloc[-1],
        len(regional_cols),
        result["public_supply_groundwater_mgd"].mean(),
    )

    return result


def _sanity_checks(df: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the output.
    """
    # Null check
    nulls = df["public_supply_groundwater_mgd"].isna().sum()
    if nulls:
        log.warning("%d null withdrawal values in output.", nulls)

    # Negative check
    negatives = (df["public_supply_groundwater_mgd"] < 0).sum()
    if negatives:
        log.warning("%d negative withdrawal values.", negatives)

    # Row count — 2000-01 through 2020-12 = 21 years × 12 months = 252
    expected = 21 * 12
    if len(df) != expected:
        log.warning(
            "Expected %d monthly rows (2000-2020) but got %d.",
            expected,
            len(df),
        )
    else:
        log.info("Row count correct: %d monthly rows.", len(df))

    # Seasonal pattern check — public supply typically peaks in summer
    # (more outdoor water use, higher demand) but the pattern is less
    # pronounced than irrigation
    df_check = df.copy()
    df_check["month"] = df_check["year_month"].str[5:7].astype(int)
    summer = df_check[df_check["month"].isin([6, 7, 8])][
        "public_supply_groundwater_mgd"
    ].mean()
    winter = df_check[df_check["month"].isin([12, 1, 2])][
        "public_supply_groundwater_mgd"
    ].mean()

    if summer > winter:
        log.info(
            "Seasonal pattern check passed: summer mean=%.2f MGD > "
            "winter mean=%.2f MGD.",
            summer,
            winter,
        )
    else:
        log.warning(
            "Summer mean (%.2f) is NOT greater than winter mean (%.2f). "
            "Public supply typically peaks in summer in Arizona. "
            "This may be acceptable if the seasonal signal is weak.",
            summer,
            winter,
        )

    # Magnitude comparison — public supply should be smaller than
    # irrigation in an agricultural region like southern Arizona [2]
    log.info(
        "Mean public supply withdrawal: %.2f MGD. "
        "Compare with irrigation output to verify relative magnitude.",
        df["public_supply_groundwater_mgd"].mean(),
    )


def main() -> None:
    log.info("=== public_supply.py start ===")

    df = _load_raw(INPUT_FILE)

    huc12_cols = _identify_huc12_columns(df)

    regional_cols = _get_regional_huc12_columns(huc12_cols)
    log.info(
        "Using %d of %d HUC12 columns for the eight-county region.",
        len(regional_cols),
        len(huc12_cols),
    )

    monthly = _aggregate_to_monthly(df, regional_cols)

    _sanity_checks(monthly)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_FILE, index=False)
    log.info("Wrote %d rows to %s", len(monthly), OUTPUT_FILE)

    log.info("=== public_supply.py complete ===")


if __name__ == "__main__":
    main()

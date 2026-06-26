"""
irrigation.py
-------------
Loads the existing HUC12-level irrigation source matrix, filters to
HUC12s within the eight-county boundary using region.py, and
re-aggregates to a monthly regional total.

The original statewide pre-aggregated file
(irrigation_huc12_monthly_az_2000_2020.csv) cannot be un-aggregated
back to county level, so we go back to the underlying HUC12-level
source matrix and re-filter/re-sum. [2]

Input  : data/raw/ir_huc12_tot_wd_az_2000_2020.csv
Output : data/processed/irrigation_monthly.csv

Columns in output:
    year_month                      - str, format YYYY-MM
    irrigation_total_withdrawal_mgd - float, regional total (million
                                      gallons per day)

Input file structure [3]:
    The raw file is wide-format with HUC12 codes as columns. Each row
    represents one month (Year + Month columns). The values are
    withdrawal rates in MGD for each HUC12 watershed.

    Columns: Year, Month, HUC12_1, HUC12_2, ..., HUC12_N

    The "wide HUC12 columns to aggregated monthly feature" reshape
    described in the config [3] means:
        1. Identify which HUC12 columns fall inside the eight counties
        2. Sum across only those columns per row
        3. Output year_month + the summed value

Spatial filter approach:
    HUC12 watershed centroids are matched against the eight-county
    boundary. A HUC12 is included if its centroid falls inside any of
    the eight counties. This requires either:
        (a) A HUC12 shapefile to derive centroids, OR
        (b) A crosswalk table mapping HUC12 to county FIPS

    If neither is available, the script falls back to using all Arizona
    HUC12s that intersect the bounding box, with a clear warning.

Date range: 2000-01 to 2020-12 [3]
    The irrigation source data covers 2000-2020. This is shorter than
    the full project range (2000-2023) so downstream scripts will need
    to handle the gap or extrapolate.
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import (
    COUNTY_FIPS,
    filter_points,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "Final"
INPUT_FILE = RAW_DIR / "IR_HUC12_Tot_WD_monthly_2000_2020.csv"
OUTPUT_FILE = PROCESSED_DIR / "irrigation_monthly.csv"

# Optional: HUC12 shapefile for spatial filtering
# WBD (Watershed Boundary Dataset) national or state extract
HUC12_SHAPEFILE = RAW_DIR / "wbd" / "WBDHU12.shp"

# Optional: HUC12-to-county crosswalk (if available)
HUC12_COUNTY_CROSSWALK = RAW_DIR / "wbd" / "huc12_county_crosswalk.csv"

# ---------------------------------------------------------------------------
# Config [3]
# ---------------------------------------------------------------------------
YEAR_COL = "Year"
MONTH_COL = "Month"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_raw(path: Path) -> pd.DataFrame:
    """
    Load the wide-format HUC12 irrigation CSV.

    Expected structure:
        Year, Month, HUC12_col_1, HUC12_col_2, ..., HUC12_col_N

    where each HUC12 column is a 12-digit HUC code (or similar ID)
    and the values are withdrawal rates in MGD.

    Raises
    ------
    FileNotFoundError
        If the input CSV is not at the expected path.
    ValueError
        If Year or Month columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Irrigation source file not found at {path}.\n"
            "Place ir_huc12_tot_wd_az_2000_2020.csv in data/raw/ and re-run."
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
        # or is at least numeric-ish
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
            f"Non-metadata columns: "
            f"{[c for c in df.columns if c not in metadata_cols][:10]}"
        )

    return huc12_cols


# ---------------------------------------------------------------------------
# Spatial filter — determine which HUC12s are in the eight-county region
# ---------------------------------------------------------------------------


def _filter_huc12s_with_shapefile(
    huc12_cols: list[str],
) -> list[str]:
    """
    Use the WBD HUC12 shapefile to determine which HUC12 watersheds
    have centroids inside the eight-county boundary.

    Parameters
    ----------
    huc12_cols : list of str
        Column names from the irrigation CSV that represent HUC12 codes.

    Returns
    -------
    List of HUC12 column names that fall inside the study area.
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

    # Match against column names in the irrigation CSV
    regional_cols = [col for col in huc12_cols if str(col).strip() in regional_hucs]

    log.info(
        "HUC12 shapefile filter: %d total HUC12s in file, "
        "%d inside boundary, %d matched to irrigation columns.",
        len(huc12_gdf),
        len(regional_hucs),
        len(regional_cols),
    )

    return regional_cols if regional_cols else None


def _filter_huc12s_with_crosswalk(
    huc12_cols: list[str],
) -> list[str]:
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

    # Match against irrigation columns
    regional_cols = [col for col in huc12_cols if str(col).strip() in regional_hucs]

    log.info(
        "Crosswalk filter: %d HUC12s in eight counties, "
        "%d matched to irrigation columns.",
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

    Returns
    -------
    All HUC12 columns from the input file.
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
    using the best available method (in order of preference):
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


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_to_monthly(
    df: pd.DataFrame,
    regional_cols: list[str],
) -> pd.DataFrame:
    """
    Sum the regional HUC12 columns per row to produce a monthly
    regional total withdrawal.

    Parameters
    ----------
    df : DataFrame
        The full wide-format input with Year, Month, and HUC12 columns.
    regional_cols : list of str
        HUC12 columns that fall inside the eight-county region.

    Returns
    -------
    DataFrame with columns: year_month, irrigation_total_withdrawal_mgd.
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
    df["irrigation_total_withdrawal_mgd"] = df[regional_cols].sum(axis=1).round(4)

    result = (
        df[["year_month", "irrigation_total_withdrawal_mgd"]]
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
        result["irrigation_total_withdrawal_mgd"].mean(),
    )

    return result


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_checks(df: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the output.
    """
    # Null check
    nulls = df["irrigation_total_withdrawal_mgd"].isna().sum()
    if nulls:
        log.warning("%d null withdrawal values in output.", nulls)

    # Negative check
    negatives = (df["irrigation_total_withdrawal_mgd"] < 0).sum()
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

    # Seasonal pattern check — irrigation should peak in summer months
    df_check = df.copy()
    df_check["month"] = df_check["year_month"].str[5:7].astype(int)
    summer = df_check[df_check["month"].isin([6, 7, 8])][
        "irrigation_total_withdrawal_mgd"
    ].mean()
    winter = df_check[df_check["month"].isin([12, 1, 2])][
        "irrigation_total_withdrawal_mgd"
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
            "Irrigation typically peaks in summer. Review the data.",
            summer,
            winter,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Full irrigation pipeline:
        load wide CSV → identify HUC12 columns → spatial filter →
        aggregate to monthly → sanity checks → write CSV.

    Parameters
    ----------
    input_file : Path
        Path to ir_huc12_tot_wd_az_2000_2020.csv.
        Defaults to data/raw/ir_huc12_tot_wd_az_2000_2020.csv.
    output_file : Path
        Path for output CSV.
        Defaults to data/processed/irrigation_monthly.csv.

    Returns
    -------
    DataFrame
        Final monthly irrigation table, also written to output_file.
    """
    log.info("=== irrigation.py start ===")

    # 1. Load raw wide-format file
    df = _load_raw(input_file)

    # 2. Identify HUC12 columns
    huc12_cols = _identify_huc12_columns(df)

    # 3. Determine which HUC12s are in the eight-county region
    regional_cols = _get_regional_huc12_columns(huc12_cols)
    log.info(
        "Using %d of %d HUC12 columns for the eight-county region.",
        len(regional_cols),
        len(huc12_cols),
    )

    # 4. Aggregate to monthly regional total
    monthly = _aggregate_to_monthly(df, regional_cols)

    # 5. Sanity checks
    _sanity_checks(monthly)

    # 6. Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_file, index=False)
    log.info("Wrote %d rows to %s", len(monthly), output_file)

    log.info("=== irrigation.py complete ===")
    return monthly


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build monthly regional irrigation withdrawal for the "
        "eight-county southern Arizona study area from HUC12 "
        "source matrix."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_FILE,
        help=f"Path to raw HUC12 irrigation CSV (default: {INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Path for output CSV (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    result = run(input_file=args.input, output_file=args.output)
    print(result.head(12).to_string(index=False))
    print("...")
    print(result.tail(12).to_string(index=False))

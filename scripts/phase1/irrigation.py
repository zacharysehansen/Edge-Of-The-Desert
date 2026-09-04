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

    The raw file is wide-format with HUC12 codes as columns. Each row
    represents one month (Year + Month columns). The values are
    withdrawal rates in MGD for each HUC12 watershed.

    Columns: Year, Month, HUC12_1, HUC12_2, ..., HUC12_N

    The "wide HUC12 columns to aggregated monthly feature" reshape
    described in the config [3] means:
        1. Identify which HUC12 columns fall inside the eight counties
        2. Sum across only those columns per row
        3. Output year_month + the summed value

    HUC12 watershed centroids are matched against the eight-county
    boundary. A HUC12 is included if its centroid falls inside any of
    the eight counties. This requires either:
        (a) A HUC12 shapefile to derive centroids, OR
        (b) A crosswalk table mapping HUC12 to county FIPS

    If neither is available the script RAISES. It used to fall back to
    summing every column, which is how a national total shipped for the
    life of the project labelled as an eight-county one -- the source
    file is not Arizona-scoped, it spans HUC regions 01-18.

    Build the shapefile from the WBD geodatabases for HUC regions 14 and
    15 (the two covering Arizona), layer WBDHU12, into data/raw/wbd/:
        https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GDB/
    That yields 7,558 polygons, of which 1,133 have centroids inside the
    eight counties.

Two nodata sentinels (999 and 888) are masked before summing; see
NODATA_SENTINELS. They accounted for 70.4% of the regional cells and,
unmasked, dominated the output by roughly four orders of magnitude.

Date range: 2000-01 to 2020-12. Will need to extrapolate
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
INPUT_FILE = RAW_DIR / "IR_HUC12_Tot_WD_monthly_2000_2020.csv"
OUTPUT_FILE = PROCESSED_DIR / "irrigation_monthly.csv"

HUC12_SHAPEFILE = RAW_DIR / "wbd" / "WBDHU12.shp"
HUC12_COUNTY_CROSSWALK = RAW_DIR / "wbd" / "huc12_county_crosswalk.csv"

YEAR_COL = "Year"
MONTH_COL = "Month"

# Nodata sentinels in the source matrix. These are NOT withdrawals and must be
# masked before summing.
#
# The file encodes missing data as literal numbers, which is why this went
# unnoticed: the columns parse as clean floats and sum without complaint.
#
#   999 - "no data for this HUC12". 73.2% of all cells in the Arizona regions
#         (14/15). 5,179 of 7,558 AZ columns are 999 for every single month;
#         another 938 are 999 for part of the record.
#   888 - "no data for this month". Intermittent, never a whole column, present
#         in 1,163 AZ columns (7,133 cells, 1.4%). It is simultaneously the most
#         common non-zero value in the file AND its maximum, in a variable with
#         189,456 distinct values -- a continuous physical quantity does not land
#         on exactly 888.000 seven thousand times.
#
# Summing without masking produced a regional "withdrawal" of 5.43e7 MGD, about
# four orders of magnitude above Arizona's entire water use, that was really a
# count of missing watersheds. It was near-constant (std/mean 0.014 vs 1.81 for
# the real signal) and therefore behaved as a disguised time trend, which is
# worse than useless: it correlated -0.37 with well depth and -0.52 with GRACE
# purely by trend-matching, while carrying no month-specific information.
NODATA_SENTINELS = (999, 888)

# Plausibility bounds for the regional total, used by _sanity_checks. Arizona's
# total water withdrawal across all sectors is roughly 6,000-7,000 MGD, so an
# eight-county irrigation figure has no business approaching five digits.
MAX_PLAUSIBLE_REGIONAL_MGD = 20_000.0
# The real signal's std/mean is ~1.8; a sentinel sum's is ~0.01.
MIN_PLAUSIBLE_CV = 0.10

# An equal-area projection for centroid computation. Taking a centroid in a
# geographic CRS (EPSG:4326) is not well defined; this follows the same
# convention nclimdiv.py uses for its area weights.
EQUAL_AREA_CRS = "EPSG:5070"


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

    # Compute centroids in an equal-area projection, then return to WGS84 for
    # the point-in-polygon test. Taking .centroid on geographic coordinates is
    # not a well-defined operation and emits a warning.
    centroid_geom = (
        huc12_gdf.to_crs(EQUAL_AREA_CRS).geometry.centroid.to_crs("EPSG:4326")
    )
    centroids = gpd.GeoDataFrame(
        huc12_gdf[[huc_col]].copy(),
        geometry=centroid_geom,
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

    regional_xwalk = xwalk[xwalk[fips_col].isin(COUNTY_FIPS)]
    regional_hucs = set(regional_xwalk[huc_col].astype(str).tolist())

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
    No spatial filter is available -- raise.

    This used to return ALL HUC12 columns with a warning, on the stated
    assumption that "the source file is already Arizona-scoped". It is not:
    the file spans HUC regions 01-18 (Maine to Oregon), so the fallback
    silently produced a national sum labelled as an eight-county total.

    A warning that nobody reads is not a safeguard. Failing here is the
    same discipline as wildfire_monthly._validate_seasonality() and
    wildlife._effort_confound_check(): refuse to write a wrong number.

    Raises
    ------
    FileNotFoundError
        Always. The caller must supply a shapefile or crosswalk.
    """
    raise FileNotFoundError(
        "NO HUC12 SPATIAL FILTER AVAILABLE -- refusing to write a regional total.\n"
        f"  Neither the WBD shapefile ({HUC12_SHAPEFILE}) nor the crosswalk file "
        f"({HUC12_COUNTY_CROSSWALK}) was found.\n"
        f"  The input carries {len(huc12_cols)} HUC12 columns spanning the entire\n"
        "  continental US (HUC regions 01-18, Maine to Oregon). The docstring's\n"
        "  assumption that the source file is 'already Arizona-scoped' is false.\n"
        "  Summing them all produces a national figure labelled as an eight-county\n"
        "  one -- which is exactly what this script used to do, silently, for the\n"
        "  entire life of the project.\n"
        "  Provide one of:\n"
        f"    - WBD HUC12 shapefile at {HUC12_SHAPEFILE}\n"
        "      (build it from WBD_14_HU2_GDB.zip + WBD_15_HU2_GDB.zip, layer WBDHU12,\n"
        "       at https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GDB/)\n"
        f"    - HUC12-to-county crosswalk CSV at {HUC12_COUNTY_CROSSWALK}"
    )


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
    result = _filter_huc12s_with_shapefile(huc12_cols)
    if result:
        return result

    result = _filter_huc12s_with_crosswalk(huc12_cols)
    if result:
        return result

    return _filter_huc12s_bbox_fallback(huc12_cols)


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
    # Mask the nodata sentinels BEFORE any arithmetic. fillna(0) after masking is
    # correct -- a missing watershed-month contributes no withdrawal to the
    # regional total -- but filling *before* masking would have summed the
    # sentinels themselves.
    sentinel_cells = 0
    total_cells = 0
    for col in regional_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        is_sentinel = values.isin(NODATA_SENTINELS)
        sentinel_cells += int(is_sentinel.sum())
        total_cells += int(len(values))
        df[col] = values.mask(is_sentinel).fillna(0)

    log.info(
        "Masked %d of %d watershed-month cells (%.1f%%) as nodata sentinels %s.",
        sentinel_cells,
        total_cells,
        100.0 * sentinel_cells / total_cells if total_cells else 0.0,
        NODATA_SENTINELS,
    )

    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce").astype(int)
    df[MONTH_COL] = pd.to_numeric(df[MONTH_COL], errors="coerce").astype(int)
    df["year_month"] = (
        df[YEAR_COL].astype(str) + "-" + df[MONTH_COL].astype(str).str.zfill(2)
    )

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


def _sanity_checks(df: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the output.
    """
    nulls = df["irrigation_total_withdrawal_mgd"].isna().sum()
    if nulls:
        log.warning("%d null withdrawal values in output.", nulls)

    negatives = (df["irrigation_total_withdrawal_mgd"] < 0).sum()
    if negatives:
        log.warning("%d negative withdrawal values.", negatives)

    # Magnitude guard. Arizona's *entire* water use across all sectors is roughly
    # 6,000-7,000 MGD, and the eight-county irrigation share is a fraction of that.
    # The sentinel-summed version of this file reported 5.43e7 MGD and nothing
    # caught it, because no check ever asked whether the number was physical.
    mean_mgd = df["irrigation_total_withdrawal_mgd"].mean()
    if mean_mgd > MAX_PLAUSIBLE_REGIONAL_MGD:
        raise ValueError(
            f"Regional irrigation withdrawal averages {mean_mgd:,.0f} MGD, above the "
            f"{MAX_PLAUSIBLE_REGIONAL_MGD:,.0f} MGD plausibility ceiling.\n"
            "  Arizona's total water use across every sector is roughly 6,000-7,000 MGD.\n"
            "  This almost certainly means nodata sentinels (999/888) were summed as "
            "data, or the spatial filter selected watersheds outside the region."
        )
    log.info("Magnitude check passed: mean %.1f MGD.", mean_mgd)

    # Variability guard. The real signal swings by more than its own mean
    # (std/mean ~1.8). A near-constant series is the signature of a sentinel sum,
    # and a near-constant feature is a disguised time trend rather than a pressure.
    ratio = df["irrigation_total_withdrawal_mgd"].std() / mean_mgd if mean_mgd else 0.0
    if ratio < MIN_PLAUSIBLE_CV:
        log.warning(
            "Irrigation series is nearly constant (std/mean = %.4f, expected > %.2f). "
            "This is the signature of a nodata sum and makes the feature a time "
            "trend rather than a pressure.",
            ratio,
            MIN_PLAUSIBLE_CV,
        )
    else:
        log.info("Variability check passed: std/mean = %.3f.", ratio)

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


def main() -> None:
    log.info("=== irrigation.py start ===")

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

    log.info("=== irrigation.py complete ===")


if __name__ == "__main__":
    main()

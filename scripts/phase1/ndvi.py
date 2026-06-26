"""
ndvi.py
-------
Loads existing MODIS MOD13A3 HDF files from data/raw/modis_ndvi,
extracts the 1km monthly NDVI subdataset, clips to the eight-county
boundary, computes the regional mean NDVI per month, and writes
a clean monthly CSV.

Input  : data/raw/modis_ndvi/*.hdf (MOD13A3 granules)
Output : data/processed/ndvi_monthly.csv

Columns in output:
    year_month  - str, format YYYY-MM
    ndvi        - float, regional mean NDVI (scaled to [-1, 1])

Source: MODIS MOD13A3 v061, 1km monthly NDVI [3]
    Product short_name: MOD13A3
    Subdataset contains: "1 km monthly NDVI"
    Scale factor: 10000 (raw integer values / 10000 = NDVI)
    Invalid values: anything below -2000 (before scaling) [3]

The raw HDF files are assumed to already exist in data/raw/modis_ndvi/.
If they need to be re-pulled, use earthaccess with the config bbox and
date range. This script only processes what is already on disk. [2]

Spatial filter:
    Each HDF granule covers a MODIS sinusoidal tile. The NDVI array is
    reprojected/clipped to the eight-county bounding box before computing
    the regional mean.

Date range: 2000-01 to 2023-12 [3]
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import BBOX

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
RAW_DIR = ROOT / "data" / "raw" / "modis_ndvi"
PROCESSED_DIR = ROOT / "data" / "Final"
OUTPUT_FILE = PROCESSED_DIR / "ndvi_monthly.csv"

# ---------------------------------------------------------------------------
# Config from phase1.example.json [3]
# ---------------------------------------------------------------------------
START_DATE = "2000-01"
END_DATE = "2023-12"
SCALE_FACTOR = 10000.0
INVALID_BELOW = -2000  # Raw integer values below this are invalid [3]
SUBDATASET_KEY = "1 km monthly NDVI"

# Bounding box [3]
MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = BBOX


# ---------------------------------------------------------------------------
# HDF file discovery
# ---------------------------------------------------------------------------


def _discover_hdf_files(raw_dir: Path) -> list[Path]:
    """
    Find all HDF files in the raw MODIS directory.

    Returns
    -------
    Sorted list of Path objects for .hdf files.

    Raises
    ------
    FileNotFoundError
        If no HDF files are found.
    """
    hdf_files = sorted(raw_dir.glob("*.hdf"))

    if not hdf_files:
        # Also check for .HDF (case insensitive)
        hdf_files = sorted(raw_dir.glob("*.HDF"))

    if not hdf_files:
        raise FileNotFoundError(
            f"No HDF files found in {raw_dir}.\n"
            "Place MOD13A3 .hdf files in data/raw/modis_ndvi/ and re-run.\n"
            "Files can be downloaded via earthaccess using the config bbox."
        )

    log.info("Found %d HDF files in %s", len(hdf_files), raw_dir)
    return hdf_files


def _extract_date_from_filename(filepath: Path) -> str | None:
    """
    Extract the year-month from a MOD13A3 filename.

    MOD13A3 filenames follow the pattern:
        MOD13A3.AYYYYDDD.hXXvYY.VVV.TIMESTAMP.hdf
    where YYYY = year, DDD = day of year.

    Returns
    -------
    str in YYYY-MM format, or None if parsing fails.
    """
    name = filepath.name

    # Pattern: AYYYYDDD in the filename
    match = re.search(r"A(\d{4})(\d{3})", name)
    if match:
        year = int(match.group(1))
        doy = int(match.group(2))

        # Convert day-of-year to month
        date = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        return date.strftime("%Y-%m")

    # Alternative: try to find YYYY.MM pattern
    match = re.search(r"(\d{4})\.(\d{2})", name)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    log.warning("Could not extract date from filename: %s", name)
    return None


# ---------------------------------------------------------------------------
# HDF processing
# ---------------------------------------------------------------------------


def _open_ndvi_subdataset(hdf_path: Path) -> rasterio:
    """
    Open the NDVI subdataset from a MOD13A3 HDF4 file.

    Uses rasterio with the HDF4 driver to access subdatasets.

    Parameters
    ----------
    hdf_path : Path
        Path to the .hdf file.

    Returns
    -------
    rasterio dataset reader for the NDVI subdataset, or None if
    the subdataset cannot be found.
    """
    # List subdatasets
    with rasterio.open(hdf_path) as src:
        subdatasets = src.subdatasets

    if not subdatasets:
        log.warning("No subdatasets found in %s", hdf_path.name)
        return None

    # Find the NDVI subdataset
    ndvi_ds = None
    for sd in subdatasets:
        if SUBDATASET_KEY.lower() in sd.lower() or "ndvi" in sd.lower():
            ndvi_ds = sd
            break

    # Fallback: take the first subdataset (MOD13A3 typically has NDVI first)
    if ndvi_ds is None:
        ndvi_ds = subdatasets[0]
        log.info(
            "  Could not find '%s' subdataset. Using first: %s",
            SUBDATASET_KEY,
            ndvi_ds,
        )

    return rasterio.open(ndvi_ds)


def _compute_mean_ndvi_from_hdf(hdf_path: Path) -> float | None:
    """
    Open an HDF file, extract the NDVI subdataset, clip/window to the
    bounding box, apply scale factor and validity mask, and compute
    the regional mean NDVI.

    Parameters
    ----------
    hdf_path : Path
        Path to the MOD13A3 .hdf file.

    Returns
    -------
    float (mean NDVI in [-1, 1] range) or None if processing fails.
    """

    try:
        src = _open_ndvi_subdataset(hdf_path)
        if src is None:
            return None

        with src:
            # Transform bounding box from WGS84 to the raster's CRS
            # MODIS uses sinusoidal projection
            raster_crs = src.crs
            if raster_crs is not None:
                try:
                    left, bottom, right, top = transform_bounds(
                        "EPSG:4326",
                        raster_crs,
                        MIN_LON,
                        MIN_LAT,
                        MAX_LON,
                        MAX_LAT,
                    )
                except Exception:
                    # If CRS transform fails, try reading the full array
                    left, bottom, right, top = src.bounds
            else:
                left, bottom, right, top = src.bounds

            # Compute window for the bounding box
            try:
                window = from_bounds(left, bottom, right, top, src.transform)
                # Clamp window to valid raster bounds
                window = window.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height)
                )
            except Exception:
                # If windowing fails, read the full raster
                window = None

            # Read data
            if window is not None and window.width > 0 and window.height > 0:
                data = src.read(1, window=window).astype(float)
            else:
                data = src.read(1).astype(float)

            # Apply validity mask [3]
            # Invalid: values below INVALID_BELOW (before scaling)
            valid_mask = data >= INVALID_BELOW

            # Also mask fill values (common MODIS fill = -3000, 32767, etc.)
            valid_mask &= data <= 10000  # Max valid NDVI raw = 10000  # noqa: PLR2004

            valid_data = data[valid_mask]

            if len(valid_data) == 0:
                log.warning("  No valid NDVI pixels in %s", hdf_path.name)
                return None

            # Apply scale factor [3]
            ndvi_scaled = valid_data / SCALE_FACTOR

            # Final validity check: NDVI should be in [-1, 1]
            ndvi_final = ndvi_scaled[(ndvi_scaled >= -1) & (ndvi_scaled <= 1)]

            if len(ndvi_final) == 0:
                log.warning(
                    "  No valid NDVI values after scaling in %s",
                    hdf_path.name,
                )
                return None

            mean_ndvi = float(np.mean(ndvi_final))
            return round(mean_ndvi, 6)

    except Exception as e:
        log.error("  Error processing %s: %s", hdf_path.name, e)
        return None


# ---------------------------------------------------------------------------
# Process all files
# ---------------------------------------------------------------------------


def _process_all_hdf_files(hdf_files: list[Path]) -> pd.DataFrame:
    """
    Process all HDF files and build a year_month → mean NDVI table.

    Parameters
    ----------
    hdf_files : list of Path
        Sorted list of HDF file paths.

    Returns
    -------
    DataFrame with columns: year_month, ndvi.
    """
    results = []

    for i, hdf_path in enumerate(hdf_files, 1):
        log.info("Processing file %d/%d: %s", i, len(hdf_files), hdf_path.name)

        # Extract date from filename
        year_month = _extract_date_from_filename(hdf_path)
        if year_month is None:
            log.warning("  Skipping — could not extract date.")
            continue

        # Filter to project date range [3]
        if year_month < START_DATE or year_month > END_DATE:
            log.info("  Skipping — outside project date range.")
            continue

        # Compute mean NDVI
        mean_ndvi = _compute_mean_ndvi_from_hdf(hdf_path)

        if mean_ndvi is not None:
            results.append({"year_month": year_month, "ndvi": mean_ndvi})
            log.info("  %s: NDVI = %.6f", year_month, mean_ndvi)
        else:
            log.warning("  %s: could not compute NDVI.", year_month)

    if not results:
        raise RuntimeError(
            "Could not compute NDVI for any HDF file.\n"
            "Check that files in data/raw/modis_ndvi/ are valid MOD13A3 "
            "granules and that rasterio can read HDF4 format.\n"
            "You may need to install the HDF4 driver: "
            "conda install -c conda-forge hdf4"
        )

    df = pd.DataFrame(results)

    # Handle duplicate months (multiple tiles for same month)
    # Average across tiles for the same month
    if df["year_month"].duplicated().any():
        n_dupes = df["year_month"].duplicated().sum()
        log.info(
            "%d duplicate year_month entries (multiple tiles). "
            "Averaging across tiles.",
            n_dupes,
        )
        df = df.groupby("year_month")["ndvi"].mean().reset_index()

    df = df.sort_values("year_month").reset_index(drop=True)
    df["ndvi"] = df["ndvi"].round(6)

    log.info(
        "NDVI processing complete: %d months (%s to %s), "
        "mean=%.4f, min=%.4f, max=%.4f.",
        len(df),
        df["year_month"].iloc[0],
        df["year_month"].iloc[-1],
        df["ndvi"].mean(),
        df["ndvi"].min(),
        df["ndvi"].max(),
    )

    return df


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------


def _check_and_fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing months in the output and optionally fill small
    gaps via linear interpolation.

    Parameters
    ----------
    df : DataFrame
        Columns: year_month, ndvi.

    Returns
    -------
    DataFrame with gaps identified and optionally filled.
    """
    # Build complete month index
    all_months = (
        pd.date_range(
            start=f"{START_DATE}-01",
            end=f"{END_DATE}-01",
            freq="MS",
        )
        .strftime("%Y-%m")
        .tolist()
    )

    present = set(df["year_month"].tolist())
    missing = [m for m in all_months if m not in present]

    if not missing:
        log.info("No gaps — all %d months are present.", len(all_months))
        return df

    log.warning(
        "%d months are missing from NDVI output: %s%s",
        len(missing),
        missing[:5],
        "..." if len(missing) > 5 else "",  # noqa: PLR2004
    )

    # Create a complete index and interpolate gaps
    complete = pd.DataFrame({"year_month": all_months})
    merged = complete.merge(df, on="year_month", how="left")

    # Linear interpolation for small gaps (up to 3 consecutive months)
    merged["ndvi"] = merged["ndvi"].interpolate(
        method="linear", limit=3, limit_direction="both"
    )

    filled_count = merged["ndvi"].notna().sum() - len(df)
    if filled_count > 0:
        log.info("Filled %d gaps via linear interpolation (limit=3).", filled_count)

    # Remaining NaN after interpolation
    remaining_nulls = merged["ndvi"].isna().sum()
    if remaining_nulls > 0:
        log.warning(
            "%d months still have no NDVI value after interpolation. "
            "These will be NaN in the output.",
            remaining_nulls,
        )

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_checks(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the NDVI output."""

    # Row count
    expected = 288  # 24 years × 12 months
    if len(df) != expected:
        log.warning("Expected %d monthly rows but got %d.", expected, len(df))
    else:
        log.info("Row count correct: %d monthly rows.", len(df))

    # Null check
    nulls = df["ndvi"].isna().sum()
    if nulls:
        log.warning("%d null NDVI values in output.", nulls)

    # Range check — NDVI should be in [-1, 1], desert regions typically 0.1-0.4
    valid = df["ndvi"].dropna()
    if valid.min() < -1 or valid.max() > 1:
        log.warning(
            "NDVI values outside [-1, 1] range: [%.4f, %.4f]. "
            "Check scale factor application.",
            valid.min(),
            valid.max(),
        )

    # Magnitude check — southern Arizona desert should be 0.1-0.4 mean
    mean_ndvi = valid.mean()
    if mean_ndvi < 0.05:  # noqa: PLR2004
        log.warning(
            "Mean NDVI (%.4f) is very low. " "Check validity masking and scale factor.",
            mean_ndvi,
        )
    elif mean_ndvi > 0.6:  # noqa: PLR2004
        log.warning(
            "Mean NDVI (%.4f) is high for a desert region. "
            "Check that the bounding box clip is correct.",
            mean_ndvi,
        )
    else:
        log.info("Magnitude check passed: mean NDVI=%.4f.", mean_ndvi)

    # Seasonal pattern check — NDVI should peak in monsoon season (Jul-Sep)
    df_check = df.copy()
    df_check["month"] = df_check["year_month"].str[5:7].astype(int)
    monsoon = df_check[df_check["month"].isin([7, 8, 9])]["ndvi"].mean()
    dry = df_check[df_check["month"].isin([4, 5, 6])]["ndvi"].mean()

    if monsoon > dry:
        log.info(
            "Seasonal check passed: monsoon mean=%.4f > dry season mean=%.4f.",
            monsoon,
            dry,
        )
    else:
        log.warning(
            "Monsoon NDVI (%.4f) is NOT higher than dry season (%.4f). "
            "Arizona vegetation typically greens up during summer monsoon.",
            monsoon,
            dry,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    raw_dir: Path = RAW_DIR,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Full NDVI pipeline:
        discover HDF files → extract NDVI → clip to bbox →
        compute monthly mean → fill gaps → sanity checks → write CSV.

    Parameters
    ----------
    raw_dir : Path
        Directory containing MOD13A3 .hdf files.
        Defaults to data/raw/modis_ndvi/.
    output_file : Path
        Path for output CSV.
        Defaults to data/processed/ndvi_monthly.csv.

    Returns
    -------
    DataFrame
        Final monthly NDVI table, also written to output_file.
    """
    log.info("=== ndvi.py start ===")

    # 1. Discover HDF files
    hdf_files = _discover_hdf_files(raw_dir)

    # 2. Process all files
    monthly = _process_all_hdf_files(hdf_files)

    # 3. Check for gaps and fill small ones
    monthly = _check_and_fill_gaps(monthly)

    # 4. Sanity checks
    _sanity_checks(monthly)

    # 5. Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_file, index=False)
    log.info("Wrote %d rows to %s", len(monthly), output_file)

    log.info("=== ndvi.py complete ===")
    return monthly


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Build monthly NDVI for the eight-county southern "
        "Arizona study area from existing MOD13A3 HDF files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory with .hdf files (default: {RAW_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Path for output CSV (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    result = run(raw_dir=args.raw_dir, output_file=args.output)
    print(result.head(12).to_string(index=False))
    print("...")
    print(result.tail(12).to_string(index=False))

"""
precipitation.py
----------------
Loads existing MERRA-2 precipitation NetCDF4 files from
data/raw/merra_precipitation, extracts the PRECTOT variable
(total precipitation), clips to the eight-county bounding box,
converts from kg/m²/s to mm/day, computes the regional monthly mean,
and writes a clean monthly CSV.

Input  : data/raw/merra_precipitation/*.nc4 (MERRA-2 M2TMNXFLX granules)
Output : data/processed/precipitation_monthly.csv

Columns in output:
    year_month              - str, format YYYY-MM
    precipitation_mm_day    - float, regional mean precipitation (mm/day)

Source: NASA MERRA-2 M2TMNXFLX (Monthly mean, Time-averaged,
        Single-Level, Full Horizontal Resolution) [3]
    Variable: PRECTOT (total precipitation, kg/m²/s)
    Transform: multiply by 86400 to convert kg/m²/s → mm/day [3]
        (1 kg/m²/s × 86400 s/day = 86400 mm/day;
         since PRECTOT is already a rate in kg/m²/s and 1 kg/m² = 1 mm)

The raw .nc4 files are assumed to already exist in
data/raw/merra_precipitation/. If they need to be re-pulled, use
earthaccess with short_name="M2TMNXFLX" and the config bbox [3].

Date range: 2000-01 to 2023-12 [3]
"""

import logging
import re
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

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
RAW_DIR = ROOT / "data" / "raw" / "merra_precipitation"
PROCESSED_DIR = ROOT / "data" / "Final"
OUTPUT_FILE = PROCESSED_DIR / "precipitation_monthly.csv"

# ---------------------------------------------------------------------------
# Config from phase1.example.json [3]
# ---------------------------------------------------------------------------
START_DATE = "2000-01"
END_DATE = "2023-12"
VARIABLE_NAME = "PRECTOT"

# Unit conversion: kg/m²/s → mm/day [3]
# 1 kg/m²/s = 1 mm/s × 86400 s/day = 86400 mm/day
# MERRA-2 monthly PRECTOT is a time-averaged rate in kg/m²/s
SECONDS_PER_DAY = 86400.0

# Bounding box [3]
MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = BBOX

# Alternative variable names
VARIABLE_CANDIDATES = [
    "PRECTOT",
    "PRECTOTLAND",
    "PRECLSC",
    "PRECCON",
    "prectot",
    "precipitation",
]

# Coordinate name candidates
LAT_CANDIDATES = ["lat", "latitude", "Latitude", "LAT"]
LON_CANDIDATES = ["lon", "longitude", "Longitude", "LON"]
TIME_CANDIDATES = ["time", "Time", "TIME", "t"]


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _discover_nc4_files(raw_dir: Path) -> list[Path]:
    """
    Find all NetCDF4 files in the raw MERRA-2 precipitation directory.

    Returns
    -------
    Sorted list of Path objects for .nc4 or .nc files.

    Raises
    ------
    FileNotFoundError
        If no NetCDF files are found.
    """
    nc4_files = sorted(raw_dir.glob("*.nc4"))
    nc_files = sorted(raw_dir.glob("*.nc"))
    all_files = sorted(set(nc4_files + nc_files))

    if not all_files:
        raise FileNotFoundError(
            f"No NetCDF files (.nc4 or .nc) found in {raw_dir}.\n"
            "Place MERRA-2 M2TMNXFLX .nc4 files in "
            "data/raw/merra_precipitation/ and re-run.\n"
            "Files can be downloaded via earthaccess with "
            "short_name='M2TMNXFLX'."
        )

    log.info("Found %d NetCDF files in %s", len(all_files), raw_dir)
    return all_files


# ---------------------------------------------------------------------------
# NetCDF processing
# ---------------------------------------------------------------------------


def _find_variable(ds: xr.Dataset) -> str:
    """
    Find the precipitation variable name in the dataset.
    """
    for candidate in VARIABLE_CANDIDATES:
        if candidate in ds.data_vars:
            return candidate

    # Fallback: look for any variable with 'prec' or 'rain' in name
    for var in ds.data_vars:
        if "prec" in var.lower() or "rain" in var.lower():
            return var

    raise ValueError(
        f"Could not find precipitation variable in dataset.\n"
        f"Available variables: {list(ds.data_vars)}"
    )


def _find_coord(ds: xr.Dataset, candidates: list[str]) -> str:
    """Find a coordinate name from a list of candidates."""
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise ValueError(
        f"Could not find coordinate from candidates {candidates}.\n"
        f"Available coords: {list(ds.coords)}"
    )


def _process_single_file(  # noqa: C901, PLR0912, PLR0915
    nc_path: Path,
) -> pd.DataFrame | None:
    """
    Open a single MERRA-2 NetCDF file, extract PRECTOT for the
    bounding box, convert to mm/day, and return monthly mean values.

    Parameters
    ----------
    nc_path : Path
        Path to a .nc4 file.

    Returns
    -------
    DataFrame with columns: year_month, precipitation_mm_day.
    Or None if the file cannot be processed.
    """
    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")
    except Exception as e:
        log.warning("Could not open %s: %s", nc_path.name, e)
        return None

    try:
        # Find variable and coordinate names
        var_name = _find_variable(ds)
        lat_name = _find_coord(ds, LAT_CANDIDATES)
        lon_name = _find_coord(ds, LON_CANDIDATES)
        time_name = _find_coord(ds, TIME_CANDIDATES)

        log.info(
            "  File: %s | var=%s, lat=%s, lon=%s, time=%s",
            nc_path.name,
            var_name,
            lat_name,
            lon_name,
            time_name,
        )

        # Handle longitude convention (0-360 vs -180-180)
        lon_vals = ds[lon_name].values
        if lon_vals.max() > 180:  # noqa: PLR2004
            min_lon_adj = MIN_LON % 360
            max_lon_adj = MAX_LON % 360
        else:
            min_lon_adj = MIN_LON
            max_lon_adj = MAX_LON

        # Subset to bounding box
        lat_vals = ds[lat_name].values
        if lat_vals[0] > lat_vals[-1]:
            lat_slice = slice(MAX_LAT, MIN_LAT)
        else:
            lat_slice = slice(MIN_LAT, MAX_LAT)

        lon_slice = slice(min_lon_adj, max_lon_adj)

        subset = ds[var_name].sel({lat_name: lat_slice, lon_name: lon_slice})

        if subset.size == 0:
            log.warning(
                "  No data in bounding box for %s. "
                "Trying nearest-neighbor selection.",
                nc_path.name,
            )
            center_lat = (MIN_LAT + MAX_LAT) / 2
            center_lon = (MIN_LON + MAX_LON) / 2
            if lon_vals.max() > 180:  # noqa: PLR2004
                center_lon = center_lon % 360

            subset = ds[var_name].sel(
                {lat_name: center_lat, lon_name: center_lon},
                method="nearest",
            )

        # Compute spatial mean per timestep
        if time_name in subset.dims:
            spatial_mean = subset.mean(
                dim=[d for d in subset.dims if d != time_name],
                skipna=True,
            )
        else:
            spatial_mean = subset.mean(skipna=True)

        # Convert to DataFrame
        if time_name in subset.dims:
            times = pd.to_datetime(ds[time_name].values)
            values = spatial_mean.values

            df = pd.DataFrame(
                {
                    "date": times,
                    "precip_raw": values,
                }
            )
        else:
            date = _extract_date_from_filename(nc_path)
            if date is None:
                log.warning("  Cannot determine date for %s", nc_path.name)
                ds.close()
                return None

            val = float(spatial_mean.values)
            df = pd.DataFrame(
                {
                    "date": [pd.Timestamp(date)],
                    "precip_raw": [val],
                }
            )

        ds.close()

        # Convert kg/m²/s to mm/day [3]
        df["precipitation_mm_day"] = df["precip_raw"] * SECONDS_PER_DAY

        # Convert to year_month
        df["date"] = pd.to_datetime(df["date"])
        df["year_month"] = df["date"].dt.to_period("M").astype(str)

        # Drop NaN
        df = df.dropna(subset=["precipitation_mm_day"])

        if df.empty:
            return None

        # Average if multiple values per month
        monthly = df.groupby("year_month")["precipitation_mm_day"].mean().reset_index()

        monthly["precipitation_mm_day"] = monthly["precipitation_mm_day"].round(6)

        return monthly

    except Exception as e:
        log.error("  Error processing %s: %s", nc_path.name, e)
        ds.close()
        return None


def _extract_date_from_filename(filepath: Path) -> str | None:
    """
    Extract a date from a MERRA-2 filename.

    MERRA-2 filenames follow patterns like:
        MERRA2_400.tavgM_2d_flx_Nx.200001.nc4
    """
    name = filepath.stem

    # Try YYYYMM pattern
    match = re.search(r"(\d{4})(\d{2})", name)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        if 2000 <= year <= 2030 and 1 <= month <= 12:  # noqa: PLR2004
            return f"{year}-{month:02d}-01"

    # Try YYYY-MM pattern
    match = re.search(r"(\d{4})-(\d{2})", name)
    if match:
        return f"{match.group(1)}-{match.group(2)}-01"

    return None


# ---------------------------------------------------------------------------
# Process all files
# ---------------------------------------------------------------------------


def _process_all_nc_files(nc_files: list[Path]) -> pd.DataFrame:
    """
    Process all NetCDF files and build a year_month → precipitation table.

    Parameters
    ----------
    nc_files : list of Path
        Sorted list of .nc4 file paths.

    Returns
    -------
    DataFrame with columns: year_month, precipitation_mm_day.
    """
    results = []

    for i, nc_path in enumerate(nc_files, 1):
        log.info("Processing file %d/%d: %s", i, len(nc_files), nc_path.name)

        result = _process_single_file(nc_path)
        if result is not None and not result.empty:
            results.append(result)
            log.info("  Extracted %d monthly values.", len(result))

    if not results:
        raise RuntimeError(
            "Could not extract precipitation from any NetCDF file.\n"
            "Check that files in data/raw/merra_precipitation/ are valid "
            "MERRA-2 M2TMNXFLX granules and that xarray/netCDF4 are installed."
        )

    df = pd.concat(results, ignore_index=True)

    # Handle duplicates
    if df["year_month"].duplicated().any():
        n_dupes = df["year_month"].duplicated().sum()
        log.info("%d duplicate months found. Averaging values.", n_dupes)
        df = df.groupby("year_month")["precipitation_mm_day"].mean().reset_index()

    # Filter to project date range [3]
    df = df[(df["year_month"] >= START_DATE) & (df["year_month"] <= END_DATE)].copy()

    df = df.sort_values("year_month").reset_index(drop=True)
    df["precipitation_mm_day"] = df["precipitation_mm_day"].round(6)

    log.info(
        "Precipitation processing complete: %d months (%s to %s), "
        "mean=%.4f mm/day, min=%.4f mm/day, max=%.4f mm/day.",
        len(df),
        df["year_month"].iloc[0],
        df["year_month"].iloc[-1],
        df["precipitation_mm_day"].mean(),
        df["precipitation_mm_day"].min(),
        df["precipitation_mm_day"].max(),
    )

    return df


# ---------------------------------------------------------------------------
# Gap filling
# ---------------------------------------------------------------------------


def _check_and_fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing months and fill small gaps via interpolation.
    """
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
        log.info("No gaps — all %d months present.", len(all_months))
        return df

    log.warning(
        "%d months missing: %s%s",
        len(missing),
        missing[:5],
        "..." if len(missing) > 5 else "",  # noqa: PLR2004
    )

    complete = pd.DataFrame({"year_month": all_months})
    merged = complete.merge(df, on="year_month", how="left")

    # Linear interpolation for small gaps (up to 3 months)
    merged["precipitation_mm_day"] = merged["precipitation_mm_day"].interpolate(
        method="linear", limit=3, limit_direction="both"
    )

    filled = merged["precipitation_mm_day"].notna().sum() - len(df)
    if filled > 0:
        log.info("Filled %d gaps via linear interpolation (limit=3).", filled)

    remaining = merged["precipitation_mm_day"].isna().sum()
    if remaining > 0:
        log.warning("%d months still NaN after interpolation.", remaining)

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_checks(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the precipitation output."""

    # Row count
    expected = 288  # 24 years × 12 months
    if len(df) != expected:
        log.warning("Expected %d rows but got %d.", expected, len(df))
    else:
        log.info("Row count correct: %d monthly rows.", len(df))

    # Null check
    nulls = df["precipitation_mm_day"].isna().sum()
    if nulls:
        log.warning("%d null precipitation values.", nulls)

    # Range check — southern Arizona is arid, typical monthly mean precip
    # is 0.1-3.0 mm/day. Values outside 0-10 mm/day would be suspicious.
    valid = df["precipitation_mm_day"].dropna()
    if valid.min() < 0:
        log.warning(
            "Negative precipitation values found (min=%.4f). " "Check unit conversion.",
            valid.min(),
        )
    elif valid.max() > 10:  # noqa: PLR2004
        log.warning(
            "Max precipitation %.4f mm/day seems high for southern Arizona. "
            "Verify unit conversion (kg/m²/s × 86400).",
            valid.max(),
        )
    else:
        log.info(
            "Range check passed: [%.4f, %.4f] mm/day.",
            valid.min(),
            valid.max(),
        )

    # Seasonal pattern — precipitation should peak during monsoon (Jul-Sep)
    # and have a secondary winter peak (Dec-Feb)
    df_check = df.copy()
    df_check["month"] = df_check["year_month"].str[5:7].astype(int)
    monsoon = df_check[df_check["month"].isin([7, 8, 9])]["precipitation_mm_day"].mean()
    dry = df_check[df_check["month"].isin([4, 5, 6])]["precipitation_mm_day"].mean()

    if monsoon > dry:
        log.info(
            "Seasonal check passed: monsoon mean=%.4f mm/day > "
            "dry season mean=%.4f mm/day.",
            monsoon,
            dry,
        )
    else:
        log.warning(
            "Monsoon (%.4f) is NOT wetter than dry season (%.4f). "
            "Southern Arizona has a strong monsoon signal.",
            monsoon,
            dry,
        )

    # Overall mean — southern Arizona gets ~250-400mm/year = ~0.7-1.1 mm/day
    mean_precip = valid.mean()
    if 0.1 < mean_precip < 5.0:  # noqa: PLR2004
        log.info("Mean precipitation check passed: %.4f mm/day.", mean_precip)
    else:
        log.warning(
            "Mean precipitation %.4f mm/day is outside expected range "
            "[0.1, 5.0] for southern Arizona.",
            mean_precip,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    raw_dir: Path = RAW_DIR,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Full precipitation pipeline:
        discover .nc4 files → extract PRECTOT for bbox →
        convert kg/m²/s to mm/day → compute monthly spatial mean →
        fill gaps → sanity checks → write CSV.

    Parameters
    ----------
    raw_dir : Path
        Directory containing MERRA-2 .nc4 files.
        Defaults to data/raw/merra_precipitation/.
    output_file : Path
        Path for output CSV.
        Defaults to data/processed/precipitation_monthly.csv.

    Returns
    -------
    DataFrame
        Final monthly precipitation table, also written to output_file.
    """
    log.info("=== precipitation.py start ===")

    # 1. Discover NetCDF files
    nc_files = _discover_nc4_files(raw_dir)

    # 2. Process all files
    monthly = _process_all_nc_files(nc_files)

    # 3. Check for gaps and fill
    monthly = _check_and_fill_gaps(monthly)

    # 4. Sanity checks
    _sanity_checks(monthly)

    # 5. Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_file, index=False)
    log.info("Wrote %d rows to %s", len(monthly), output_file)

    log.info("=== precipitation.py complete ===")
    return monthly


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build monthly precipitation for the eight-county "
        "southern Arizona study area from MERRA-2 NetCDF files."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help=f"Directory with .nc4 files (default: {RAW_DIR})",
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

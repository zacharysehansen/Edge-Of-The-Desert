"""
temperature.py
--------------
Loads existing MERRA-2 temperature NetCDF4 files from
data/raw/merra_temperature_2m, extracts the T2MMEAN variable
(2-meter mean temperature), clips to the eight-county bounding box,
converts from Kelvin to Celsius, computes the regional monthly mean,
and writes a clean monthly CSV.

Input  : data/raw/merra_temperature_2m/*.nc4 (MERRA-2 M2SMNXSLV granules)
Output : data/processed/temperature_monthly.csv

Columns in output:
    year_month          - str, format YYYY-MM
    temperature_2m_c    - float, regional mean 2-meter temperature (°C)

Source: NASA MERRA-2 M2SMNXSLV (Stationary Monthly means, Single-Level) [3]
    Variable: T2MMEAN (2-meter air temperature, monthly mean)
    Units in file: Kelvin
    Transform: subtract 273.15 to get Celsius [3]

The raw .nc4 files are assumed to already exist in
data/raw/merra_temperature_2m/. If they need to be re-pulled, use
earthaccess with short_name="M2SMNXSLV" and the config bbox.

Date range: 2000-01 to 2023-12 [3]
"""

import logging
import re
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import BBOX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw" / "merra_temperature_2m"
PROCESSED_DIR = ROOT / "data" / "Final"
OUTPUT_FILE = PROCESSED_DIR / "temperature_monthly.csv"

START_DATE = "2000-01"
END_DATE = "2023-12"
VARIABLE_NAME = "T2MMEAN"
KELVIN_OFFSET = 273.15

MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = BBOX

# Alternative variable names in case of product version differences
VARIABLE_CANDIDATES = [
    "T2MMEAN",
    "T2M",
    "t2m",
    "T2MMAX",
    "T2MMIN",
]

# Coordinate name candidates
LAT_CANDIDATES = ["lat", "latitude", "Latitude", "LAT"]
LON_CANDIDATES = ["lon", "longitude", "Longitude", "LON"]
TIME_CANDIDATES = ["time", "Time", "TIME", "t"]


def _discover_nc4_files(raw_dir: Path) -> list[Path]:
    """
    Find all NetCDF4 files in the raw MERRA-2 temperature directory.

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
            "Place MERRA-2 M2SMNXSLV .nc4 files in "
            "data/raw/merra_temperature_2m/ and re-run.\n"
            "Files can be downloaded via earthaccess with "
            "short_name='M2SMNXSLV'."
        )

    log.info("Found %d NetCDF files in %s", len(all_files), raw_dir)
    return all_files


def _find_variable(ds: xr.Dataset) -> str:
    """
    Find the temperature variable name in the dataset.
    MERRA-2 products may use slightly different names across versions.
    """
    for candidate in VARIABLE_CANDIDATES:
        if candidate in ds.data_vars:
            return candidate

    for var in ds.data_vars:
        if "t2m" in var.lower() or "temp" in var.lower():
            return var

    raise ValueError(
        f"Could not find temperature variable in dataset.\n"
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
    Open a single MERRA-2 NetCDF file, extract T2MMEAN for the
    bounding box, convert to Celsius, and return monthly mean values.
    """
    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")
    except Exception as e:
        log.warning("Could not open %s: %s", nc_path.name, e)
        return None

    try:
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

        if time_name in subset.dims:
            spatial_mean = subset.mean(
                dim=[d for d in subset.dims if d != time_name],
                skipna=True,
            )
        else:
            spatial_mean = subset.mean(skipna=True)

        if time_name in subset.dims:
            times = pd.to_datetime(ds[time_name].values)
            values = spatial_mean.values

            df = pd.DataFrame(
                {
                    "date": times,
                    "temperature_k": values,
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
                    "temperature_k": [val],
                }
            )

        ds.close()

        # Convert Kelvin to Celsius [3]
        df["temperature_2m_c"] = df["temperature_k"] - KELVIN_OFFSET

        # Convert to year_month
        df["date"] = pd.to_datetime(df["date"])
        df["year_month"] = df["date"].dt.to_period("M").astype(str)

        df = df.dropna(subset=["temperature_2m_c"])

        if df.empty:
            return None

        monthly = df.groupby("year_month")["temperature_2m_c"].mean().reset_index()

        monthly["temperature_2m_c"] = monthly["temperature_2m_c"].round(4)

        return monthly

    except Exception as e:
        log.error("  Error processing %s: %s", nc_path.name, e)
        ds.close()
        return None


def _extract_date_from_filename(filepath: Path) -> str | None:
    """
    Extract a date from a MERRA-2 filename.

    MERRA-2 filenames follow patterns like:
        MERRA2_400.statM_2d_slv_Nx.200001.nc4
        MERRA2_400.tavgM_2d_slv_Nx.200001.nc4
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


def _process_all_nc_files(nc_files: list[Path]) -> pd.DataFrame:
    """
    Process all NetCDF files and build a year_month → temperature table.

    Parameters
    ----------
    nc_files : list of Path
        Sorted list of .nc4 file paths.

    Returns
    -------
    DataFrame with columns: year_month, temperature_2m_c.
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
            "Could not extract temperature from any NetCDF file.\n"
            "Check that files in data/raw/merra_temperature_2m/ are valid "
            "MERRA-2 M2SMNXSLV granules and that xarray/netCDF4 are installed."
        )

    df = pd.concat(results, ignore_index=True)

    # Handle duplicates (multiple files for same month)
    if df["year_month"].duplicated().any():
        n_dupes = df["year_month"].duplicated().sum()
        log.info("%d duplicate months found. Averaging values.", n_dupes)
        df = df.groupby("year_month")["temperature_2m_c"].mean().reset_index()

    # Filter to project date range [3]
    df = df[(df["year_month"] >= START_DATE) & (df["year_month"] <= END_DATE)].copy()

    df = df.sort_values("year_month").reset_index(drop=True)
    df["temperature_2m_c"] = df["temperature_2m_c"].round(4)

    log.info(
        "Temperature processing complete: %d months (%s to %s), "
        "mean=%.2f°C, min=%.2f°C, max=%.2f°C.",
        len(df),
        df["year_month"].iloc[0],
        df["year_month"].iloc[-1],
        df["temperature_2m_c"].mean(),
        df["temperature_2m_c"].min(),
        df["temperature_2m_c"].max(),
    )

    return df


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
    merged["temperature_2m_c"] = merged["temperature_2m_c"].interpolate(
        method="linear", limit=3, limit_direction="both"
    )

    filled = merged["temperature_2m_c"].notna().sum() - len(df)
    if filled > 0:
        log.info("Filled %d gaps via linear interpolation (limit=3).", filled)

    remaining = merged["temperature_2m_c"].isna().sum()
    if remaining > 0:
        log.warning("%d months still NaN after interpolation.", remaining)

    return merged.reset_index(drop=True)


def _sanity_checks(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the temperature output."""

    # Row count
    expected = 288  # 24 years × 12 months
    if len(df) != expected:
        log.warning("Expected %d rows but got %d.", expected, len(df))
    else:
        log.info("Row count correct: %d monthly rows.", len(df))

    # Null check
    nulls = df["temperature_2m_c"].isna().sum()
    if nulls:
        log.warning("%d null temperature values.", nulls)

    # Range check — southern Arizona monthly mean temp should be ~5-35°C
    valid = df["temperature_2m_c"].dropna()
    if valid.min() < -10 or valid.max() > 50:  # noqa: PLR2004
        log.warning(
            "Temperature outside [-10, 50]°C: [%.2f, %.2f]. "
            "Check Kelvin conversion.",
            valid.min(),
            valid.max(),
        )
    else:
        log.info(
            "Range check passed: [%.2f, %.2f]°C.",
            valid.min(),
            valid.max(),
        )

    # Seasonal pattern — should be hotter in summer, cooler in winter
    df_check = df.copy()
    df_check["month"] = df_check["year_month"].str[5:7].astype(int)
    summer = df_check[df_check["month"].isin([6, 7, 8])]["temperature_2m_c"].mean()
    winter = df_check[df_check["month"].isin([12, 1, 2])]["temperature_2m_c"].mean()

    if summer > winter:
        log.info(
            "Seasonal check passed: summer mean=%.2f°C > winter mean=%.2f°C.",
            summer,
            winter,
        )
    else:
        log.warning(
            "Summer (%.2f°C) is NOT warmer than winter (%.2f°C). "
            "Check data processing.",
            summer,
            winter,
        )

    # Mean should be roughly 18-25°C for southern Arizona
    mean_temp = valid.mean()
    if 10 < mean_temp < 35:  # noqa: PLR2004
        log.info("Mean temperature check passed: %.2f°C.", mean_temp)
    else:
        log.warning(
            "Mean temperature %.2f°C is outside expected range [10, 35] "
            "for southern Arizona.",
            mean_temp,
        )


def main() -> None:
    log.info("=== temperature.py start ===")

    nc_files = _discover_nc4_files(RAW_DIR)

    monthly = _process_all_nc_files(nc_files)

    monthly = _check_and_fill_gaps(monthly)

    _sanity_checks(monthly)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_FILE, index=False)
    log.info("Wrote %d rows to %s", len(monthly), OUTPUT_FILE)

    log.info("=== temperature.py complete ===")


if __name__ == "__main__":
    main()

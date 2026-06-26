"""
grace_groundwater.py
--------------------
Re-extracts GRACE/GRACE-FO groundwater storage anomaly from raw .nc4
files using a bounding box around the eight counties, applies the same
grace_available flag and pre-2002-04 neutral fill as the original
pipeline. [1]

Input  : data/raw/grace_groundwater_anomaly/*.nc4
         (TELLUS_GRAC_L3_JPL_RL06_LND_v04 and
          TELLUS_GRFO_L3_JPL_RL06.3_LND_v04)
Output : data/processed/grace_monthly.csv

Columns in output:
    year_month                  - str, format YYYY-MM
    grace_groundwater_anomaly   - float, liquid water equivalent
                                  thickness (cm) averaged over the
                                  eight-county bounding box
    grace_available             - int, 1 if GRACE data exists for
                                  that month, 0 if filled

Source: NASA GRACE / GRACE-FO JPL RL06 Mascon [3]
    Variable: lwe_thickness (liquid water equivalent thickness, cm)
    GRACE mission: 2002-04 through 2017-06
    GRACE-FO mission: 2018-06 through 2023-12
    Gap: 2017-07 through 2018-05 (no satellite in orbit)

Spatial resolution:
    GRACE is a coarse-resolution satellite product (~300km effective
    resolution, gridded to 0.5° or 1° cells). It was never truly
    statewide-precise to begin with, so re-extracting it with a
    bounding box around the eight counties from the raw files is
    straightforward. [2]

Fill logic:
    - Months before 2002-04: filled with 0.0 (neutral), grace_available=0
    - Months in the 2017-07 to 2018-05 gap: linear interpolation,
      grace_available=0
    - Months with valid data: grace_available=1

Date range: 2000-01 to 2023-12 [3]
    The config start is 2000-01 but GRACE data begins 2002-04.
    Pre-GRACE months are filled with 0.0 as documented above. [1]
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
RAW_DIR = ROOT / "data" / "raw" / "grace_groundwater_anomaly"
PROCESSED_DIR = ROOT / "data" / "Final"
OUTPUT_FILE = PROCESSED_DIR / "grace_monthly.csv"

# ---------------------------------------------------------------------------
# Config from phase1.example.json [3]
# ---------------------------------------------------------------------------
START_DATE = "2000-01"
END_DATE = "2023-12"
VARIABLE_NAME = "lwe_thickness"
GRACE_START = "2002-04"  # First valid GRACE month
GRACE_GAP_START = "2017-07"  # Inter-mission gap begins
GRACE_GAP_END = "2018-05"  # Inter-mission gap ends
GRACE_FO_START = "2018-06"  # GRACE-FO begins
PRE_GRACE_FILL = 0.0  # Neutral fill for pre-GRACE months [1]

# Bounding box [3]
MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = BBOX

# Alternative variable names found in different GRACE products
VARIABLE_CANDIDATES = [
    "lwe_thickness",
    "lwe_thickness_jpl",
    "lwe_thickness_csr",
    "lwe_thickness_gfz",
    "Liquid_Water_Equivalent_Thickness",
]

# Coordinate name candidates
LAT_CANDIDATES = ["lat", "latitude", "Latitude", "LAT"]
LON_CANDIDATES = ["lon", "longitude", "Longitude", "LON"]
TIME_CANDIDATES = ["time", "Time", "TIME", "t"]


# ---------------------------------------------------------------------------
# NetCDF file discovery
# ---------------------------------------------------------------------------


def _discover_nc4_files(raw_dir: Path) -> list[Path]:
    """
    Find all NetCDF4 files in the raw GRACE directory.

    Returns
    -------
    Sorted list of Path objects for .nc4 or .nc files.

    Raises
    ------
    FileNotFoundError
        If no NetCDF files are found.
    """
    nc4_files = sorted(raw_dir.glob("*.nc4"))

    # Also check for .nc extension
    nc_files = sorted(raw_dir.glob("*.nc"))
    all_files = sorted(set(nc4_files + nc_files))

    if not all_files:
        raise FileNotFoundError(
            f"No NetCDF files (.nc4 or .nc) found in {raw_dir}.\n"
            "Place GRACE/GRACE-FO .nc4 files in "
            "data/raw/grace_groundwater_anomaly/ and re-run.\n"
            "Files can be downloaded via earthaccess with short_names:\n"
            "  TELLUS_GRAC_L3_JPL_RL06_LND_v04\n"
            "  TELLUS_GRFO_L3_JPL_RL06.3_LND_v04"
        )

    log.info("Found %d NetCDF files in %s", len(all_files), raw_dir)
    return all_files


# ---------------------------------------------------------------------------
# NetCDF processing
# ---------------------------------------------------------------------------


def _find_variable(ds: xr.Dataset) -> str:
    """
    Find the lwe_thickness variable name in the dataset.
    GRACE products use slightly different naming conventions.
    """
    for candidate in VARIABLE_CANDIDATES:
        if candidate in ds.data_vars:
            return candidate

    # Fallback: look for any variable with 'lwe' or 'thickness' in the name
    for var in ds.data_vars:
        if "lwe" in var.lower() or "thickness" in var.lower():
            return var

    raise ValueError(
        f"Could not find lwe_thickness variable in dataset.\n"
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
    Open a single NetCDF file, extract lwe_thickness for the
    bounding box, and return monthly mean values.

    Parameters
    ----------
    nc_path : Path
        Path to a .nc4 file.

    Returns
    -------
    DataFrame with columns: year_month, grace_groundwater_anomaly.
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
            # Convert bbox to 0-360 convention
            min_lon_adj = MIN_LON % 360
            max_lon_adj = MAX_LON % 360
        else:
            min_lon_adj = MIN_LON
            max_lon_adj = MAX_LON

        # Subset to bounding box
        # Handle case where lat might be decreasing
        lat_vals = ds[lat_name].values
        if lat_vals[0] > lat_vals[-1]:
            # Latitude is decreasing
            lat_slice = slice(MAX_LAT, MIN_LAT)
        else:
            lat_slice = slice(MIN_LAT, MAX_LAT)

        if min_lon_adj <= max_lon_adj:
            lon_slice = slice(min_lon_adj, max_lon_adj)
        else:
            # Wraps around the date line (shouldn't happen for Arizona)
            lon_slice = slice(min_lon_adj, max_lon_adj)

        subset = ds[var_name].sel({lat_name: lat_slice, lon_name: lon_slice})

        if subset.size == 0:
            log.warning(
                "  No data in bounding box for %s. "
                "Trying nearest-neighbor selection.",
                nc_path.name,
            )
            # Fallback: select nearest grid cell to center of bbox
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
            # Single timestep file
            spatial_mean = subset.mean(skipna=True)

        # Convert to DataFrame
        if time_name in subset.dims:
            times = pd.to_datetime(ds[time_name].values)
            values = spatial_mean.values

            df = pd.DataFrame(
                {
                    "date": times,
                    "grace_groundwater_anomaly": values,
                }
            )
        else:
            # Try to extract date from filename
            date = _extract_date_from_filename(nc_path)
            if date is None:
                log.warning("  Cannot determine date for %s", nc_path.name)
                ds.close()
                return None

            val = float(spatial_mean.values)
            df = pd.DataFrame(
                {
                    "date": [pd.Timestamp(date)],
                    "grace_groundwater_anomaly": [val],
                }
            )

        ds.close()

        # Convert to year_month
        df["date"] = pd.to_datetime(df["date"])
        df["year_month"] = df["date"].dt.to_period("M").astype(str)
        df["grace_groundwater_anomaly"] = pd.to_numeric(
            df["grace_groundwater_anomaly"], errors="coerce"
        )

        # Drop NaN values
        df = df.dropna(subset=["grace_groundwater_anomaly"])

        if df.empty:
            return None

        # Average if multiple values per month (shouldn't happen but safety)
        monthly = (
            df.groupby("year_month")["grace_groundwater_anomaly"].mean().reset_index()
        )

        return monthly

    except Exception as e:
        log.error("  Error processing %s: %s", nc_path.name, e)
        ds.close()
        return None


def _extract_date_from_filename(filepath: Path) -> str | None:
    """
    Try to extract a date from a GRACE NetCDF filename.

    Common patterns:
        GRCTellus.JPL.200204_202312.GLO.RL06.1M.MSCNv03CRI.nc
        GRACE_GRFO_L3_JPL_RL06.3_LND_v04_2018-06_2023-12.nc4
    """
    name = filepath.stem

    # Try to find YYYYMM pattern
    match = re.search(r"(\d{4})(\d{2})", name)
    if match:
        return f"{match.group(1)}-{match.group(2)}-01"

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
    Process all NetCDF files and combine into a single monthly table.

    Parameters
    ----------
    nc_files : list of Path
        Sorted list of .nc4 file paths.

    Returns
    -------
    DataFrame with columns: year_month, grace_groundwater_anomaly.
    """
    frames = []

    for i, nc_path in enumerate(nc_files, 1):
        log.info("Processing file %d/%d: %s", i, len(nc_files), nc_path.name)
        result = _process_single_file(nc_path)
        if result is not None and not result.empty:
            frames.append(result)
            log.info("  Extracted %d monthly values.", len(result))

    if not frames:
        raise RuntimeError(
            "Could not extract GRACE data from any NetCDF file.\n"
            "Check that files in data/raw/grace_groundwater_anomaly/ are "
            "valid GRACE/GRACE-FO NetCDF4 files and that the xarray/netCDF4 "
            "packages are installed correctly."
        )

    combined = pd.concat(frames, ignore_index=True)

    # Deduplicate — same month may appear in multiple files
    # (GRACE and GRACE-FO overlap slightly at boundaries)
    if combined["year_month"].duplicated().any():
        n_dupes = combined["year_month"].duplicated().sum()
        log.info(
            "%d duplicate months found (overlap between files). " "Averaging values.",
            n_dupes,
        )
        combined = (
            combined.groupby("year_month")["grace_groundwater_anomaly"]
            .mean()
            .reset_index()
        )

    combined = combined.sort_values("year_month").reset_index(drop=True)
    combined["grace_groundwater_anomaly"] = combined["grace_groundwater_anomaly"].round(
        6
    )

    log.info(
        "GRACE extraction complete: %d months (%s to %s), "
        "mean=%.4f cm, min=%.4f cm, max=%.4f cm.",
        len(combined),
        combined["year_month"].iloc[0],
        combined["year_month"].iloc[-1],
        combined["grace_groundwater_anomaly"].mean(),
        combined["grace_groundwater_anomaly"].min(),
        combined["grace_groundwater_anomaly"].max(),
    )

    return combined


# ---------------------------------------------------------------------------
# Fill logic — pre-GRACE and inter-mission gap [1]
# ---------------------------------------------------------------------------


def _apply_fill_and_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the complete monthly index from START_DATE to END_DATE,
    merge with extracted GRACE data, apply fill logic, and set
    the grace_available flag.

    Fill rules [1]:
        - Pre-GRACE (before 2002-04): fill with 0.0, grace_available=0
        - Inter-mission gap (2017-07 to 2018-05): linear interpolation,
          grace_available=0
        - Months with actual data: grace_available=1
        - Any other missing months: linear interpolation, grace_available=0

    Parameters
    ----------
    df : DataFrame
        Columns: year_month, grace_groundwater_anomaly.

    Returns
    -------
    DataFrame with columns: year_month, grace_groundwater_anomaly,
    grace_available. Complete from START_DATE to END_DATE.
    """
    # Build complete monthly index
    all_months = (
        pd.date_range(
            start=f"{START_DATE}-01",
            end=f"{END_DATE}-01",
            freq="MS",
        )
        .strftime("%Y-%m")
        .tolist()
    )

    complete = pd.DataFrame({"year_month": all_months})

    # Merge with extracted data
    merged = complete.merge(df, on="year_month", how="left")

    # Set grace_available flag
    merged["grace_available"] = merged["grace_groundwater_anomaly"].notna().astype(int)

    # --- Fill pre-GRACE months with 0.0 [1] ---
    pre_grace_mask = merged["year_month"] < GRACE_START
    pre_grace_count = pre_grace_mask.sum()
    merged.loc[pre_grace_mask, "grace_groundwater_anomaly"] = PRE_GRACE_FILL
    merged.loc[pre_grace_mask, "grace_available"] = 0
    log.info(
        "Pre-GRACE fill: %d months before %s set to %.1f.",
        pre_grace_count,
        GRACE_START,
        PRE_GRACE_FILL,
    )

    # --- Interpolate the inter-mission gap [1] ---
    gap_mask = (merged["year_month"] >= GRACE_GAP_START) & (
        merged["year_month"] <= GRACE_GAP_END
    )
    gap_count = gap_mask.sum()

    # Linear interpolation across all remaining NaN values
    merged["grace_groundwater_anomaly"] = merged[
        "grace_groundwater_anomaly"
    ].interpolate(method="linear", limit_direction="both")

    # Set grace_available=0 for the gap months
    merged.loc[gap_mask, "grace_available"] = 0
    log.info(
        "Inter-mission gap: %d months (%s to %s) interpolated.",
        gap_count,
        GRACE_GAP_START,
        GRACE_GAP_END,
    )

    # Any remaining NaN (shouldn't happen after interpolation, but safety)
    remaining_nan = merged["grace_groundwater_anomaly"].isna().sum()
    if remaining_nan > 0:
        log.warning(
            "%d months still have NaN after fill/interpolation. " "Setting to %.1f.",
            remaining_nan,
            PRE_GRACE_FILL,
        )
        merged["grace_groundwater_anomaly"] = merged[
            "grace_groundwater_anomaly"
        ].fillna(PRE_GRACE_FILL)
        # These are also not real data
        merged.loc[
            merged["grace_groundwater_anomaly"] == PRE_GRACE_FILL, "grace_available"
        ] = 0

    merged["grace_groundwater_anomaly"] = merged["grace_groundwater_anomaly"].round(6)

    # Summary
    available = merged["grace_available"].sum()
    filled = len(merged) - available
    log.info(
        "Fill summary: %d months total, %d with real data, %d filled.",
        len(merged),
        available,
        filled,
    )

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_checks(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the GRACE output."""

    # Row count — expect 288 months (2000-01 to 2023-12)
    expected = 288
    if len(df) != expected:
        log.warning("Expected %d monthly rows but got %d.", expected, len(df))
    else:
        log.info("Row count correct: %d monthly rows.", len(df))

    # Null check
    nulls = df["grace_groundwater_anomaly"].isna().sum()
    if nulls:
        log.warning("%d null values remain after fill logic.", nulls)

    # grace_available flag check
    available = df["grace_available"].sum()
    log.info(
        "grace_available: %d months with real data, %d filled.",
        available,
        len(df) - available,
    )

    # Pre-GRACE months should all be 0.0
    pre_grace = df[df["year_month"] < GRACE_START]
    if not (pre_grace["grace_groundwater_anomaly"] == PRE_GRACE_FILL).all():
        log.warning("Some pre-GRACE months are not filled with 0.0.")
    if not (pre_grace["grace_available"] == 0).all():
        log.warning("Some pre-GRACE months have grace_available=1.")

    # Magnitude check — GRACE lwe_thickness is typically in range
    # [-30, +10] cm for Arizona (declining water storage)
    valid_data = df[df["grace_available"] == 1]["grace_groundwater_anomaly"]
    if not valid_data.empty:
        if valid_data.min() < -50 or valid_data.max() > 30:  # noqa: PLR2004
            log.warning(
                "GRACE values [%.2f, %.2f] cm exceed expected range "
                "[-50, +30] for Arizona. Verify units.",
                valid_data.min(),
                valid_data.max(),
            )
        else:
            log.info(
                "Magnitude check passed: range [%.2f, %.2f] cm.",
                valid_data.min(),
                valid_data.max(),
            )

    # Declining trend check — southern Arizona groundwater has generally
    # declined since 2002 [2]
    if not valid_data.empty:
        early = df[(df["year_month"] >= "2002-04") & (df["year_month"] < "2005-01")][
            "grace_groundwater_anomaly"
        ].mean()
        late = df[df["year_month"] >= "2021-01"]["grace_groundwater_anomaly"].mean()

        if late < early:
            log.info(
                "Trend check passed: early mean=%.2f cm, "
                "late mean=%.2f cm (declining as expected).",
                early,
                late,
            )
        else:
            log.warning(
                "GRACE anomaly increased from early (%.2f) to late (%.2f). "
                "Unexpected for southern Arizona — verify data.",
                early,
                late,
            )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(
    raw_dir: Path = RAW_DIR,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Full GRACE groundwater pipeline:
        discover .nc4 files → extract lwe_thickness for bbox →
        compute monthly spatial mean → apply fill logic and flags →
        sanity checks → write CSV.

    Parameters
    ----------
    raw_dir : Path
        Directory containing GRACE .nc4 files.
        Defaults to data/raw/grace_groundwater_anomaly/.
    output_file : Path
        Path for output CSV.
        Defaults to data/processed/grace_monthly.csv.

    Returns
    -------
    DataFrame
        Final monthly GRACE table, also written to output_file.
    """
    log.info("=== grace_groundwater.py start ===")

    # 1. Discover NetCDF files
    nc_files = _discover_nc4_files(raw_dir)

    # 2. Process all files and extract monthly spatial means
    monthly_raw = _process_all_nc_files(nc_files)

    # 3. Apply fill logic and grace_available flag [1]
    monthly = _apply_fill_and_flags(monthly_raw)

    # 4. Sanity checks
    _sanity_checks(monthly)

    # 5. Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_file, index=False)
    log.info("Wrote %d rows to %s", len(monthly), output_file)

    log.info("=== grace_groundwater.py complete ===")
    return monthly


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build monthly GRACE groundwater storage anomaly for "
        "the eight-county southern Arizona study area from "
        "existing .nc4 files."
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

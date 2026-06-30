"""
urbanization.py
---------------
Reads pre-extracted NLCD Annual Fractional Impervious Surface rasters
from data/raw/NLCD/, clips to the eight-county bounding box, computes
mean impervious percentage per year, and interpolates to monthly.

Input  : Three extracted MRLC bundles in data/raw/NLCD/:
         - Annual_NLCD_FctImp_1995-2004_CU_C1V1/
         - Annual_NLCD_FctImp_2005-2014_CU_C1V1/
         - Annual_NLCD_FctImp_2015-2024_CU_C1V1/

Output : data/Final/urbanization_monthly.csv

Columns in output:
    year_month      - str, format YYYY-MM
    impervious_pct  - float, mean fractional impervious surface (%)
                      across all non-null pixels in the eight-county region

Strategy:
    1. Scan the three extracted bundle directories for year-specific TIFs
    2. For each year 2000-2023, do a windowed read limited to the
       bounding box (avoids loading the full CONUS raster into memory)
    3. Compute mean impervious % for valid pixels in the window
    4. Interpolate annual values to monthly via cubic spline
    5. Write output CSV

Date range: 2000-01 to 2023-12
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import BBOX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw" / "NLCD"
OUTPUT_FILE = ROOT / "data" / "Final" / "urbanization_monthly.csv"

START_YEAR = 2000
END_YEAR = 2023

MIN_LON, MIN_LAT, MAX_LON, MAX_LAT = BBOX

BUNDLE_DIRS = [
    RAW_DIR / "Annual_NLCD_FctImp_1995-2004_CU_C1V1",
    RAW_DIR / "Annual_NLCD_FctImp_2005-2014_CU_C1V1",
    RAW_DIR / "Annual_NLCD_FctImp_2015-2024_CU_C1V1",
]


def _discover_rasters() -> dict[int, Path]:
    """
    Scan bundle directories for TIF files and map year → file path.
    """
    year_to_path = {}

    for bundle_dir in BUNDLE_DIRS:
        if not bundle_dir.exists():
            log.warning("Bundle directory not found: %s", bundle_dir)
            continue

        for tif in bundle_dir.glob("*.tif"):
            for year in range(1985, 2030):
                if str(year) in tif.name:
                    year_to_path[year] = tif
                    break

    log.info(
        "Discovered %d TIF rasters across %d bundle directories.",
        len(year_to_path),
        len(BUNDLE_DIRS),
    )

    return year_to_path


def _compute_mean_impervious(tif_path: Path, year: int) -> float:
    """
    Do a windowed read of the TIF covering only the bounding box,
    and compute mean impervious percentage for valid pixels.
    """
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        if raster_crs and raster_crs.to_epsg() != 4326:  # noqa: PLR2004
            left, bottom, right, top = transform_bounds(
                "EPSG:4326",
                raster_crs,
                MIN_LON,
                MIN_LAT,
                MAX_LON,
                MAX_LAT,
            )
        else:
            left, bottom, right, top = MIN_LON, MIN_LAT, MAX_LON, MAX_LAT

        try:
            window = from_bounds(left, bottom, right, top, src.transform)
        except Exception as e:
            log.warning("  Could not compute window for year %d: %s", year, e)
            return np.nan

        data = src.read(1, window=window).astype(float)

        nodata = src.nodata
        if nodata is None:
            nodata = 250

        data_range = [0, 100]
        valid_mask = (
            (data >= data_range[0]) & (data <= data_range[1]) & (data != nodata)
        )
        valid_data = data[valid_mask]

        if len(valid_data) == 0:
            log.warning("  No valid pixels for year %d.", year)
            return np.nan

        mean_pct = float(np.mean(valid_data))
        log.info(
            "  Year %d: window shape=%s, valid pixels=%d, mean impervious=%.4f%%",
            year,
            data.shape,
            len(valid_data),
            mean_pct,
        )

        return round(mean_pct, 4)


def _build_annual_impervious() -> pd.DataFrame:
    """
    Scan for TIFs, process each year in range, compute annual mean
    impervious percentage.
    """
    year_to_path = _discover_rasters()

    results = []
    for year in range(START_YEAR, END_YEAR + 1):
        if year not in year_to_path:
            log.warning("  No raster found for year %d. Skipping.", year)
            continue

        log.info("Processing year %d...", year)
        try:
            mean_pct = _compute_mean_impervious(year_to_path[year], year)
            if not np.isnan(mean_pct):
                results.append({"year": year, "impervious_pct": mean_pct})
        except Exception as e:
            log.error("  Error processing year %d: %s", year, e)

    if not results:
        raise RuntimeError(
            "Could not process any NLCD impervious rasters.\n"
            "Check that data/raw/NLCD/ contains extracted bundle directories."
        )

    annual = pd.DataFrame(results).sort_values("year").reset_index(drop=True)

    log.info(
        "Annual impervious summary: %d years (%d–%d), min=%.4f%%, max=%.4f%%.",
        len(annual),
        annual["year"].min(),
        annual["year"].max(),
        annual["impervious_pct"].min(),
        annual["impervious_pct"].max(),
    )

    return annual


def _interpolate_to_monthly(annual: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate annual impervious values to monthly using cubic spline.
    Annual values are treated as July 1 anchor points (year + 0.5).
    """
    anchor_x = annual["year"].values + 0.5
    anchor_y = annual["impervious_pct"].values.astype(float)

    cs = CubicSpline(anchor_x, anchor_y, extrapolate=True)

    months = pd.date_range(
        start=f"{START_YEAR}-01-01",
        end=f"{END_YEAR}-12-01",
        freq="MS",
    )

    monthly_x = months.year + (months.month - 1) / 12
    monthly_pct = cs(monthly_x)

    monthly_pct = np.clip(monthly_pct, 0, 100)

    result = pd.DataFrame(
        {
            "year_month": months.strftime("%Y-%m"),
            "impervious_pct": np.round(monthly_pct, 4),
        }
    )

    log.info(
        "Monthly interpolation: %d rows (%s to %s), min=%.4f%%, max=%.4f%%.",
        len(result),
        result["year_month"].iloc[0],
        result["year_month"].iloc[-1],
        result["impervious_pct"].min(),
        result["impervious_pct"].max(),
    )

    return result


def main() -> None:
    log.info("=== urbanization.py start ===")

    annual = _build_annual_impervious()

    years_present = set(annual["year"].tolist())
    years_expected = set(range(START_YEAR, END_YEAR + 1))
    years_missing = years_expected - years_present
    if years_missing:
        log.warning(
            "Missing data for %d years: %s.", len(years_missing), sorted(years_missing)
        )
    else:
        log.info("Full year coverage: %d–%d.", START_YEAR, END_YEAR)

    monthly = _interpolate_to_monthly(annual)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_FILE, index=False)
    log.info("Wrote %d rows to %s", len(monthly), OUTPUT_FILE)

    log.info("=== urbanization.py complete ===")


if __name__ == "__main__":
    main()

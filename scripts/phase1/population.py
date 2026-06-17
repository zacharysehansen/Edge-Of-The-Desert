"""
population.py
-------------
Pulls U.S. Census Bureau county population estimates for the eight
southern Arizona study counties, sums to a regional annual total,
and interpolates to monthly.

This script replaces the old statewide azpop_monthly.csv rather than
re-aggregating it, since the original interpolation kept no county
breakdown. [1]

Input  : Census Bureau Population Estimates API (no key required for
         small requests, API key recommended for repeated use)
         - Vintage 2000-2010: intercensal estimates
         - Vintage 2010-2020: intercensal estimates
         - Vintage 2020-2023: postcensal estimates (most recent release)
Output : data/processed/population_monthly.csv

Columns in output:
    year_month    - str, format YYYY-MM
    population    - float, interpolated regional monthly population

Interpolation method:
    Annual Census estimates are point-in-time figures (July 1 of each
    year). Monthly values are derived via cubic spline interpolation
    between July 1 anchor points, which preserves the smooth growth
    curve expected from demographic change and matches the method used
    in the original statewide interpolation. [1]

County FIPS codes used [3]:
    04019 Pima
    04021 Pinal
    04023 Santa Cruz
    04003 Cochise
    04013 Graham
    04011 Greenlee
    04027 Yuma
    04007 La Paz

Date range: 2000-01 to 2023-12 [3]
    The config end date is 2023-12-31. Population estimates through
    2023 are available from the Census vintage 2023 postcensal release.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import COUNTIES, COUNTY_FIPS, STATE_FIPS

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
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "population_monthly.csv"

# ---------------------------------------------------------------------------
# Config [3]
# ---------------------------------------------------------------------------
START_YEAR = 2000
END_YEAR = 2023

# Census API base URL
CENSUS_API_BASE = "https://api.census.gov/data"

# Five-digit FIPS → three-digit county code (Census API uses county
# code without the state prefix)
COUNTY_CODES = [fips[2:] for fips in COUNTY_FIPS]

# Optional: set CENSUS_API_KEY environment variable to avoid rate limits
# The API works without a key for small requests but will throttle
# repeated calls. A free key is available at:
# https://api.census.gov/data/key_signup.html
import os

CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", None)

# ---------------------------------------------------------------------------
# Census API endpoints
# ---------------------------------------------------------------------------
# The Census population estimates are split across three vintages to
# cover the full 2000-2023 date range. Each vintage uses a different
# endpoint and variable naming convention.
#
# Vintage structure:
#   2000-2010 : /2000/pep/int_population  → POP, DATE_DESC
#   2010-2020 : /2019/pep/population      → POP, DATE_CODE
#   2020-2023 : /2023/pep/population      → POP, YEAR
#
# All three are queried and merged before interpolation.

VINTAGE_ENDPOINTS = [
    {
        "label": "2000-2010 intercensal",
        "url": f"{CENSUS_API_BASE}/2000/pep/int_population",
        "get": "POP,DATE_DESC",
        "year_key": "DATE_DESC",
        "year_type": "intercensal_2000",
    },
    {
        "label": "2010-2019 postcensal",
        "url": f"{CENSUS_API_BASE}/2019/pep/population",
        "get": "POP,DATE_CODE",
        "year_key": "DATE_CODE",
        "year_type": "postcensal_2010",
    },
    {
        "label": "2020-2023 postcensal",
        "url": f"{CENSUS_API_BASE}/2023/pep/population",
        "get": "POP,YEAR",
        "year_key": "YEAR",
        "year_type": "postcensal_2020",
    },
]

# ---------------------------------------------------------------------------
# Census API helpers
# ---------------------------------------------------------------------------


def _build_params(get: str, year_key: str) -> dict:
    """
    Build the Census API query parameter dict for a given vintage.
    Requests all eight study counties in one call using a comma-
    separated county list.
    """
    params = {
        "get": get,
        "for": f"county:{','.join(COUNTY_CODES)}",
        "in": f"state:{STATE_FIPS}",
    }
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY
    return params


def _fetch_vintage(endpoint: dict, retries: int = 3) -> pd.DataFrame:
    """
    Fetch one Census vintage endpoint and return a clean DataFrame
    with columns: year (int), county_fips (str), population (int).

    Parameters
    ----------
    endpoint : dict
        One entry from VINTAGE_ENDPOINTS.
    retries : int
        Number of retry attempts on transient HTTP errors.

    Returns
    -------
    DataFrame with columns: year, county_fips, population.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    params = _build_params(endpoint["get"], endpoint["year_key"])
    url = endpoint["url"]
    label = endpoint["label"]

    for attempt in range(1, retries + 1):
        try:
            log.info(
                "Fetching Census %s (attempt %d/%d)...",
                label,
                attempt,
                retries,
            )
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            log.warning("Attempt %d failed: %s", attempt, e)
            if attempt == retries:
                raise RuntimeError from e(
                    f"Failed to fetch Census {label} after {retries} attempts.\n"
                    f"URL: {url}\nError: {e}"
                )
            time.sleep(2**attempt)  # exponential backoff

    data = response.json()
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)
    log.info("  %s: %d raw rows received.", label, len(df))

    # Build five-digit FIPS from state + county columns
    df["county_fips"] = df["state"] + df["county"]

    # Keep only our eight counties
    df = df[df["county_fips"].isin(COUNTY_FIPS)].copy()

    # Extract the calendar year from whichever year column this vintage uses
    df["year"] = _parse_year(df, endpoint["year_type"], endpoint["year_key"])

    # Coerce population to int
    df["POP"] = pd.to_numeric(df["POP"], errors="coerce")
    df = df.dropna(subset=["year", "POP"])
    df["population"] = df["POP"].astype(int)
    df["year"] = df["year"].astype(int)

    return df[["year", "county_fips", "population"]].reset_index(drop=True)


def _parse_year(
    df: pd.DataFrame,
    year_type: str,
    year_key: str,
) -> pd.Series:
    """
    Extract the integer calendar year from the Census vintage-specific
    year column. Each vintage encodes the year differently:

    intercensal_2000 : DATE_DESC contains strings like
                       "7/1/2005 population estimate"
    postcensal_2010  : DATE_CODE is an integer 1–10 where
                       1 = April 1 2010 census, 2 = July 1 2010, etc.
    postcensal_2020  : YEAR is a plain four-digit integer string.
    """
    if year_type == "intercensal_2000":
        # Extract four-digit year from date string
        return df[year_key].str.extract(r"(\d{4})")[0].astype(float)

    elif year_type == "postcensal_2010":
        # DATE_CODE 2 = July 1 2010, 3 = July 1 2011, ..., 11 = July 1 2019
        # DATE_CODE 1 is the April 1 2010 census count — exclude it
        code = pd.to_numeric(df[year_key], errors="coerce")
        year = 2008 + code  # code 2 → 2010, code 3 → 2011, etc.
        return year

    elif year_type == "postcensal_2020":
        return pd.to_numeric(df[year_key], errors="coerce")

    else:
        raise ValueError(f"Unknown year_type: {year_type}")


# ---------------------------------------------------------------------------
# Multi-vintage pull and merge
# ---------------------------------------------------------------------------


def _pull_all_vintages() -> pd.DataFrame:
    """
    Pull all three Census vintages and merge into a single annual
    population table for the eight-county region.

    Overlap handling:
        The 2010 and 2020 census years appear in multiple vintages.
        When duplicates exist the most recent vintage is preferred,
        since postcensal estimates are revised as new data becomes
        available.

    Returns
    -------
    DataFrame with columns: year, county_fips, population.
    Sorted by county_fips, year.
    """
    frames = []
    for endpoint in VINTAGE_ENDPOINTS:
        try:
            df = _fetch_vintage(endpoint)
            frames.append(df)
        except RuntimeError as e:
            log.error("Could not fetch %s: %s", endpoint["label"], e)
            log.error(
                "Continuing without this vintage. "
                "Output may have gaps in the year range."
            )

    if not frames:
        raise RuntimeError(
            "All Census API requests failed. "
            "Check network access and Census API availability."
        )

    combined = pd.concat(frames, ignore_index=True)

    # Filter to project date range [3]
    combined = combined[
        (combined["year"] >= START_YEAR) & (combined["year"] <= END_YEAR)
    ].copy()

    # Deduplicate: keep the last occurrence of each (year, county_fips)
    # pair — this keeps the most recently fetched (newest) vintage value
    combined = (
        combined.sort_values(["county_fips", "year"])
        .drop_duplicates(subset=["year", "county_fips"], keep="last")
        .reset_index(drop=True)
    )

    log.info(
        "Combined vintage pull: %d county-year rows, "
        "years %d–%d, %d unique counties.",
        len(combined),
        combined["year"].min(),
        combined["year"].max(),
        combined["county_fips"].nunique(),
    )

    # Warn if any of the eight counties are missing entirely
    missing_counties = set(COUNTY_FIPS) - set(combined["county_fips"].unique())
    if missing_counties:
        names = [COUNTIES[COUNTY_FIPS.index(f)] for f in missing_counties]
        log.warning(
            "Missing data for %d counties: %s. "
            "Check Census API response for these FIPS codes.",
            len(missing_counties),
            ", ".join(names),
        )

    return combined


# ---------------------------------------------------------------------------
# Regional sum
# ---------------------------------------------------------------------------


def _sum_to_regional(county_annual: pd.DataFrame) -> pd.DataFrame:
    """
    Sum the eight county populations to a single regional annual total.

    Parameters
    ----------
    county_annual : DataFrame
        Output of _pull_all_vintages(). Columns: year, county_fips,
        population.

    Returns
    -------
    DataFrame with columns: year, population (regional sum).
    """
    regional = (
        county_annual.groupby("year")["population"]
        .sum()
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )

    log.info(
        "Regional annual population: %d years, " "range %s–%s, " "min=%.0f, max=%.0f.",
        len(regional),
        regional["year"].min(),
        regional["year"].max(),
        regional["population"].min(),
        regional["population"].max(),
    )

    return regional


# ---------------------------------------------------------------------------
# Monthly interpolation
# ---------------------------------------------------------------------------


def _interpolate_to_monthly(annual: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate annual Census estimates (July 1 anchor points) to
    monthly values using cubic spline interpolation.

    Census population estimates are published as of July 1 each year.
    The interpolation treats each annual value as a July 1 data point
    and fills in the eleven other months of each year by fitting a
    smooth cubic spline through all anchor points. This matches the
    method used in the original statewide interpolation. [1]

    The output spans from START_YEAR-01 through END_YEAR-12 inclusive,
    matching the project date range in the config. [3]

    Parameters
    ----------
    annual : DataFrame
        Output of _sum_to_regional(). Columns: year, population.

    Returns
    -------
    DataFrame with columns: year_month (str YYYY-MM), population (float).
    """
    # Build July 1 anchor points as fractional years
    # July 1 = year + 6/12 = year + 0.5
    anchor_x = annual["year"].values + 0.5
    anchor_y = annual["population"].values

    # Fit cubic spline through anchor points
    cs = CubicSpline(anchor_x, anchor_y, extrapolate=True)

    # Build monthly time axis from START_YEAR-01 to END_YEAR-12
    months = pd.date_range(
        start=f"{START_YEAR}-01-01",
        end=f"{END_YEAR}-12-01",
        freq="MS",  # month start
    )

    # Convert month-start dates to fractional years for spline evaluation
    # January 1 = year + 0/12, February 1 = year + 1/12, etc.
    monthly_x = months.year + (months.month - 1) / 12

    # Evaluate spline
    monthly_pop = cs(monthly_x)

    # Clip to non-negative (spline overshoot at boundaries is rare but possible)
    monthly_pop = np.maximum(monthly_pop, 0)

    result = pd.DataFrame(
        {
            "year_month": months.strftime("%Y-%m"),
            "population": np.round(monthly_pop, 2),
        }
    )

    log.info(
        "Monthly interpolation complete: %d rows (%s to %s), " "min=%.0f, max=%.0f.",
        len(result),
        result["year_month"].iloc[0],
        result["year_month"].iloc[-1],
        result["population"].min(),
        result["population"].max(),
    )

    return result


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_checks(monthly: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the interpolated output and log warnings
    for anything suspicious.
    """
    # Check for nulls
    nulls = monthly["population"].isna().sum()
    if nulls:
        log.warning("%d null population values in output.", nulls)

    # Check for negative values (spline overshoot)
    negatives = (monthly["population"] < 0).sum()
    if negatives:
        log.warning(
            "%d negative population values after interpolation. "
            "These have been clipped to 0.",
            negatives,
        )

    # Check row count — expect exactly one row per month
    expected = (END_YEAR - START_YEAR + 1) * 12
    if len(monthly) != expected:
        log.warning(
            "Expected %d monthly rows but got %d. "
            "Check for gaps in the Census vintage pull.",
            expected,
            len(monthly),
        )
    else:
        log.info("Row count correct: %d monthly rows.", len(monthly))

    # Check that population is broadly increasing over the period
    # (Southern Arizona has grown consistently 2000-2023)
    first_decade_mean = monthly[monthly["year_month"] < "2010-01"]["population"].mean()
    last_decade_mean = monthly[monthly["year_month"] >= "2013-01"]["population"].mean()

    if last_decade_mean < first_decade_mean:
        log.warning(
            "Mean population in the last decade (%.0f) is lower than "
            "the first decade (%.0f). This is unexpected for southern "
            "Arizona — check the Census API pull for data issues.",
            last_decade_mean,
            first_decade_mean,
        )
    else:
        log.info(
            "Population growth check passed: "
            "first decade mean=%.0f, last decade mean=%.0f.",
            first_decade_mean,
            last_decade_mean,
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(output_file: Path = OUTPUT_FILE) -> pd.DataFrame:
    """
    Full population pipeline:
        pull Census vintages → sum to regional → interpolate monthly
        → write CSV.

    Parameters
    ----------
    output_file : Path
        Path for output CSV.
        Defaults to data/processed/population_monthly.csv.

    Returns
    -------
    DataFrame
        Final monthly population table, also written to output_file.
    """
    log.info("=== population.py start ===")

    # 1. Pull all Census vintages and merge
    county_annual = _pull_all_vintages()

    # 2. Sum eight counties to regional annual total
    regional_annual = _sum_to_regional(county_annual)

    # 3. Verify we have data across the full date range
    years_present = set(regional_annual["year"].tolist())
    years_expected = set(range(START_YEAR, END_YEAR + 1))
    years_missing = years_expected - years_present
    if years_missing:
        log.warning(
            "Missing annual data for years: %s. "
            "Interpolation will bridge these gaps but accuracy may be reduced.",
            sorted(years_missing),
        )

    # 4. Interpolate to monthly
    monthly = _interpolate_to_monthly(regional_annual)

    # 5. Sanity checks
    _sanity_checks(monthly)

    # 6. Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(output_file, index=False)
    log.info("Wrote %d rows to %s", len(monthly), output_file)

    log.info("=== population.py complete ===")
    return monthly


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build monthly regional population estimates for the "
        "eight-county southern Arizona study area using "
        "U.S. Census Bureau population estimates."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Path for output CSV (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    result = run(output_file=args.output)
    print(result.to_string(index=False))

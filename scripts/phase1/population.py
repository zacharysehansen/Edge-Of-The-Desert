"""
population.py
-------------
Builds monthly regional population estimates for the eight-county
southern Arizona study area using a hybrid approach:

    - 2000-2010 : Census Bureau API (intercensal estimates)
    - 2010-2020 : Direct CSV download from Census FTP
    - 2020-2023 : Direct CSV download from Census FTP

The API is used for the 2000-2010 vintage because that endpoint is
confirmed working. The 2010-2020 and 2020-2023 vintages use direct
CSV downloads because the Census PEP API endpoints for those periods
return 404 errors. [1]

Output : data/processed/population_monthly.csv

Columns in output:
    year_month    - str, format YYYY-MM
    population    - float, interpolated regional monthly population

Interpolation method:
    Annual Census estimates are point-in-time figures (July 1 of each
    year). Monthly values are derived via cubic spline interpolation
    between July 1 anchor points, which preserves the smooth growth
    curve expected from demographic change. [1]

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
"""

import logging
import os
import sys
import time
from io import StringIO
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
PROCESSED_DIR = ROOT / "data" / "Final"
OUTPUT_FILE = PROCESSED_DIR / "azpop_monthly.csv"

# ---------------------------------------------------------------------------
# Config [3]
# ---------------------------------------------------------------------------
START_YEAR = 2000
END_YEAR = 2023

# Census API key — set via environment variable or fallback
CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", None)

# Three-digit county codes (Census API uses county without state prefix)
COUNTY_CODES = [fips[2:] for fips in COUNTY_FIPS]

# Census API endpoint for 2000-2010 intercensal (confirmed working)
API_2000_2010_URL = "https://api.census.gov/data/2000/pep/int_population"

# Direct CSV download URLs for 2010-2020 and 2020-2023 (confirmed working)
CSV_2010_2020_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2010-2020/counties/totals/co-est2020-alldata.csv"
)
CSV_2020_2023_URL = (
    "https://www2.census.gov/programs-surveys/popest/datasets/"
    "2020-2023/counties/totals/co-est2023-alldata.csv"
)


# ---------------------------------------------------------------------------
# Source 1: Census API 2000-2010
# ---------------------------------------------------------------------------


def _fetch_api_2000_2010(retries: int = 3) -> pd.DataFrame:
    """
    Fetch 2000-2010 intercensal population estimates from the Census API.

    The API returns rows with a DATE_DESC column containing strings like
    "7/1/2005 population estimate". The year is extracted from that string.

    Returns
    -------
    DataFrame with columns: year, county_fips, population.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.
    """
    params = {
        "get": "POP,DATE_DESC",
        "for": f"county:{','.join(COUNTY_CODES)}",
        "in": f"state:{STATE_FIPS}",
    }
    if CENSUS_API_KEY:
        params["key"] = CENSUS_API_KEY

    for attempt in range(1, retries + 1):
        try:
            log.info(
                "Fetching Census API 2000-2010 (attempt %d/%d)...",
                attempt,
                retries,
            )
            response = requests.get(API_2000_2010_URL, params=params, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            log.warning("Attempt %d failed: %s", attempt, e)
            if attempt == retries:
                raise RuntimeError(
                    f"Census API 2000-2010 failed after {retries} attempts.\n"
                    f"Error: {e}"
                ) from e
            time.sleep(2**attempt)

    data = response.json()
    headers = data[0]
    rows = data[1:]

    df = pd.DataFrame(rows, columns=headers)

    # Build five-digit FIPS
    df["county_fips"] = df["state"] + df["county"]
    df = df[df["county_fips"].isin(COUNTY_FIPS)].copy()

    # Extract year from DATE_DESC (e.g. "7/1/2005 population estimate")
    df["year"] = df["DATE_DESC"].str.extract(r"(\d{4})")[0].astype(float)

    # Coerce population
    df["POP"] = pd.to_numeric(df["POP"], errors="coerce")
    df = df.dropna(subset=["year", "POP"])
    df["year"] = df["year"].astype(int)
    df["population"] = df["POP"].astype(int)

    # Filter to July 1 estimates only (exclude census day counts, base pops)
    # DATE_DESC for July 1 estimates contains "7/1/"
    july_mask = df["DATE_DESC"].str.contains("7/1/", na=False)
    df = df[july_mask].copy()

    # Keep only 2000-2009 from this source (2010 comes from the next source)
    df = df[(df["year"] >= 2000) & (df["year"] <= 2009)]  # noqa: PLR2004

    log.info(
        "API 2000-2010: %d county-year rows, years %d–%d.",
        len(df),
        df["year"].min(),
        df["year"].max(),
    )

    return df[["year", "county_fips", "population"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source 2: Direct CSV 2010-2020
# ---------------------------------------------------------------------------


def _fetch_csv_2010_2020() -> pd.DataFrame:
    """
    Download and parse the 2010-2020 intercensal population estimates
    from the Census FTP server.

    The file is wide-format with columns POPESTIMATE2010 through
    POPESTIMATE2020. These are melted to long format.

    Returns
    -------
    DataFrame with columns: year, county_fips, population.
    """
    log.info("Downloading Census CSV 2010-2020...")
    response = requests.get(CSV_2010_2020_URL, timeout=60)
    response.raise_for_status()
    log.info(
        "  Status: %d, size: %d bytes", response.status_code, len(response.content)
    )

    df = pd.read_csv(StringIO(response.text), encoding="latin-1")

    # Build five-digit FIPS and filter
    df["county_fips"] = df["STATE"].astype(str).str.zfill(2) + df["COUNTY"].astype(
        str
    ).str.zfill(3)
    df = df[df["county_fips"].isin(COUNTY_FIPS)].copy()

    # Identify POPESTIMATE columns (POPESTIMATE2010 through POPESTIMATE2020)
    pop_cols = [c for c in df.columns if c.startswith("POPESTIMATE20")]
    if not pop_cols:
        raise ValueError(
            "No POPESTIMATE columns found in the 2010-2020 CSV.\n"
            f"Available columns: {df.columns.tolist()[:20]}"
        )

    # Melt from wide to long
    melted = df.melt(
        id_vars=["county_fips"],
        value_vars=pop_cols,
        var_name="year_col",
        value_name="population",
    )
    melted["year"] = melted["year_col"].str.extract(r"(\d{4})")[0].astype(int)
    melted["population"] = pd.to_numeric(melted["population"], errors="coerce")
    melted = melted.dropna(subset=["population"])
    melted["population"] = melted["population"].astype(int)

    # Keep 2010-2019 from this source (2020 comes from the next source)
    melted = melted[
        (melted["year"] >= 2010) & (melted["year"] <= 2019)  # noqa: PLR2004
    ]

    log.info(
        "CSV 2010-2020: %d county-year rows, years %d–%d.",
        len(melted),
        melted["year"].min(),
        melted["year"].max(),
    )

    return melted[["year", "county_fips", "population"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source 3: Direct CSV 2020-2023
# ---------------------------------------------------------------------------


def _fetch_csv_2020_2023() -> pd.DataFrame:
    """
    Download and parse the 2020-2023 postcensal population estimates
    from the Census FTP server.

    Returns
    -------
    DataFrame with columns: year, county_fips, population.
    """
    log.info("Downloading Census CSV 2020-2023...")
    response = requests.get(CSV_2020_2023_URL, timeout=60)
    response.raise_for_status()
    log.info(
        "  Status: %d, size: %d bytes", response.status_code, len(response.content)
    )

    df = pd.read_csv(StringIO(response.text), encoding="latin-1")

    # Build five-digit FIPS and filter
    df["county_fips"] = df["STATE"].astype(str).str.zfill(2) + df["COUNTY"].astype(
        str
    ).str.zfill(3)
    df = df[df["county_fips"].isin(COUNTY_FIPS)].copy()

    # Identify POPESTIMATE columns
    pop_cols = [c for c in df.columns if c.startswith("POPESTIMATE20")]
    if not pop_cols:
        raise ValueError(
            "No POPESTIMATE columns found in the 2020-2023 CSV.\n"
            f"Available columns: {df.columns.tolist()[:20]}"
        )

    # Melt from wide to long
    melted = df.melt(
        id_vars=["county_fips"],
        value_vars=pop_cols,
        var_name="year_col",
        value_name="population",
    )
    melted["year"] = melted["year_col"].str.extract(r"(\d{4})")[0].astype(int)
    melted["population"] = pd.to_numeric(melted["population"], errors="coerce")
    melted = melted.dropna(subset=["population"])
    melted["population"] = melted["population"].astype(int)

    # Keep 2020-2023 from this source
    melted = melted[
        (melted["year"] >= 2020) & (melted["year"] <= 2023)  # noqa: PLR2004
    ]

    log.info(
        "CSV 2020-2023: %d county-year rows, years %d–%d.",
        len(melted),
        melted["year"].min(),
        melted["year"].max(),
    )

    return melted[["year", "county_fips", "population"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Combine all vintages
# ---------------------------------------------------------------------------


def _pull_all_vintages() -> pd.DataFrame:
    """
    Pull all three population sources and combine into a single
    county-year table.

    Year boundary decisions:
        - API provides 2000-2009
        - CSV 2010-2020 provides 2010-2019
        - CSV 2020-2023 provides 2020-2023

    This avoids overlap issues at the seams entirely. Each source
    owns a non-overlapping range of years.

    Returns
    -------
    DataFrame with columns: year, county_fips, population.
    """
    frames = []

    # Source 1: API 2000-2010
    try:
        df_api = _fetch_api_2000_2010()
        frames.append(df_api)
    except Exception as e:
        log.error("Census API 2000-2010 failed: %s", e)

    # Source 2: CSV 2010-2020
    try:
        df_2010 = _fetch_csv_2010_2020()
        frames.append(df_2010)
    except Exception as e:
        log.error("Census CSV 2010-2020 failed: %s", e)

    # Source 3: CSV 2020-2023
    try:
        df_2020 = _fetch_csv_2020_2023()
        frames.append(df_2020)
    except Exception as e:
        log.error("Census CSV 2020-2023 failed: %s", e)

    if not frames:
        raise RuntimeError(
            "All three population sources failed. "
            "Check network access and Census API/FTP availability."
        )

    combined = pd.concat(frames, ignore_index=True)

    # Final dedup safety net
    combined = (
        combined.sort_values(["county_fips", "year"])
        .drop_duplicates(subset=["year", "county_fips"], keep="last")
        .reset_index(drop=True)
    )

    log.info(
        "Combined: %d county-year rows, years %d–%d, %d counties.",
        len(combined),
        combined["year"].min(),
        combined["year"].max(),
        combined["county_fips"].nunique(),
    )

    # Warn if any counties are missing
    missing = set(COUNTY_FIPS) - set(combined["county_fips"].unique())
    if missing:
        names = [COUNTIES[COUNTY_FIPS.index(f)] for f in missing]
        log.warning("Missing data for counties: %s", ", ".join(names))

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
        Columns: year, county_fips, population.

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
        "Regional annual population: %d years (%d–%d), " "min=%s, max=%s.",
        len(regional),
        regional["year"].min(),
        regional["year"].max(),
        f"{regional['population'].min():,}",
        f"{regional['population'].max():,}",
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
    July 1 = year + 0.5 in fractional year space. The spline is fit
    through these anchor points and evaluated at each month-start
    from START_YEAR-01 through END_YEAR-12. [1]

    Parameters
    ----------
    annual : DataFrame
        Columns: year, population.

    Returns
    -------
    DataFrame with columns: year_month (str YYYY-MM), population (float).
    """
    # July 1 anchor points as fractional years
    anchor_x = annual["year"].values + 0.5
    anchor_y = annual["population"].values.astype(float)

    # Fit cubic spline
    cs = CubicSpline(anchor_x, anchor_y, extrapolate=True)

    # Monthly time axis
    months = pd.date_range(
        start=f"{START_YEAR}-01-01",
        end=f"{END_YEAR}-12-01",
        freq="MS",
    )

    # Fractional year for each month-start
    monthly_x = months.year + (months.month - 1) / 12

    # Evaluate spline
    monthly_pop = cs(monthly_x)

    # Clip to non-negative (spline overshoot at boundaries is rare)
    monthly_pop = np.maximum(monthly_pop, 0)

    result = pd.DataFrame(
        {
            "year_month": months.strftime("%Y-%m"),
            "population": np.round(monthly_pop, 2),
        }
    )

    log.info(
        "Monthly interpolation: %d rows (%s to %s), " "min=%s, max=%s.",
        len(result),
        result["year_month"].iloc[0],
        result["year_month"].iloc[-1],
        f"{result['population'].min():,.0f}",
        f"{result['population'].max():,.0f}",
    )

    return result


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_checks(monthly: pd.DataFrame) -> None:
    """
    Run basic sanity checks on the interpolated output.
    """
    # Null check
    nulls = monthly["population"].isna().sum()
    if nulls:
        log.warning("%d null population values in output.", nulls)

    # Negative check
    negatives = (monthly["population"] < 0).sum()
    if negatives:
        log.warning(
            "%d negative values after interpolation (clipped to 0).",
            negatives,
        )

    # Row count check
    expected = (END_YEAR - START_YEAR + 1) * 12
    if len(monthly) != expected:
        log.warning("Expected %d monthly rows but got %d.", expected, len(monthly))
    else:
        log.info("Row count correct: %d monthly rows.", len(monthly))

    # Growth direction check — southern Arizona grew 2000-2023 [2]
    first_decade = monthly[monthly["year_month"] < "2010-01"]["population"].mean()
    last_decade = monthly[monthly["year_month"] >= "2014-01"]["population"].mean()

    if last_decade < first_decade:
        log.warning(
            "Population decreased over time (first decade mean=%s, "
            "last decade mean=%s). This is unexpected — check data.",
            f"{first_decade:,.0f}",
            f"{last_decade:,.0f}",
        )
    else:
        log.info(
            "Growth check passed: first decade mean=%s, " "last decade mean=%s.",
            f"{first_decade:,.0f}",
            f"{last_decade:,.0f}",
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(output_file: Path = OUTPUT_FILE) -> pd.DataFrame:
    """
    Full population pipeline:
        pull three vintages → combine → sum to regional →
        interpolate monthly → sanity checks → write CSV.

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

    # 1. Pull all three sources
    county_annual = _pull_all_vintages()

    # 2. Sum to regional annual total
    regional_annual = _sum_to_regional(county_annual)

    # 3. Check year coverage
    years_present = set(regional_annual["year"].tolist())
    years_expected = set(range(START_YEAR, END_YEAR + 1))
    years_missing = years_expected - years_present
    if years_missing:
        log.warning(
            "Missing annual data for years: %s. "
            "Spline will bridge gaps but accuracy may be reduced.",
            sorted(years_missing),
        )
    else:
        log.info("Full year coverage confirmed: %d–%d.", START_YEAR, END_YEAR)

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
        "eight-county southern Arizona study area."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Path for output CSV (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    result = run(output_file=args.output)
    print(result.head(12).to_string(index=False))
    print("...")
    print(result.tail(12).to_string(index=False))

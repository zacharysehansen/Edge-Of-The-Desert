"""
water_stress.py
---------------
Pulls county-level USDM DSCI data from the USDM Statistics API for
each of the eight southern Arizona study counties, computes an
area-weighted regional DSCI, aggregates weekly to monthly, and
derives the water stress score.

Input  : USDM Statistics API (CountyStatistics/GetDSCI endpoint)
         https://usdmdataservices.unl.edu/api/CountyStatistics/GetDSCI
Output : data/processed/water_stress_monthly.csv

Columns in output:
    year_month          - str, format YYYY-MM
    usdm_dsci           - float, area-weighted monthly mean DSCI (0-500)
    water_stress_score  - float, derived (100 - dsci/5), range [0, 100]

DSCI Scale:
    0   = No drought (D0-D4 all at 0%)
    500 = Entire county in D4 (exceptional drought)

Area-weighted regional DSCI:
    Each county's weekly DSCI is weighted by its land area before
    averaging across the eight counties. This gives larger counties
    (Pima, Yuma, Pinal) proportionally more influence on the regional
    score than smaller counties (Santa Cruz, Greenlee). [1]

Water stress formula [1][2]:
    water_stress_score = 100 - (usdm_dsci / 5)

Date range: 2000-01 to 2023-12 [3]
"""

import logging
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import COUNTIES, COUNTY_FIPS

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
RAW_DIR = ROOT / "data" / "raw" / "usdm_sustainability"
PROCESSED_DIR = ROOT / "data" / "Final"
OUTPUT_FILE = PROCESSED_DIR / "water_stress_monthly.csv"

# ---------------------------------------------------------------------------
# Config [3]
# ---------------------------------------------------------------------------
START_DATE = "1/1/2000"
END_DATE = "12/31/2023"
START_MONTH = "2000-01"
END_MONTH = "2023-12"

# USDM Statistics API endpoint (confirmed working — returns CSV)
USDM_API_URL = "https://usdmdataservices.unl.edu/api/CountyStatistics/GetDSCI"

# County land areas in square miles (for area weighting)
# Source: U.S. Census Bureau
COUNTY_AREAS_SQMI = {
    "04019": 9189,  # Pima
    "04021": 5374,  # Pinal
    "04023": 1238,  # Santa Cruz
    "04003": 6219,  # Cochise
    "04013": 4641,  # Graham
    "04011": 1848,  # Greenlee
    "04027": 5519,  # Yuma
    "04007": 4513,  # La Paz
}

TOTAL_AREA = sum(COUNTY_AREAS_SQMI.values())

# Water stress formula [1][2]
DSCI_DIVISOR = 5.0
STRESS_BASELINE = 100.0


# ---------------------------------------------------------------------------
# API pull
# ---------------------------------------------------------------------------


def _fetch_county_dsci(
    fips: str, county_name: str, retries: int = 3
) -> pd.DataFrame | None:
    """
    Fetch weekly DSCI data for a single county from the USDM API.

    The API returns CSV format (not JSON) with columns:
        State, County, FIPS, MapDate, DSCI

    Parameters
    ----------
    fips : str
        Five-digit county FIPS code.
    county_name : str
        County name for logging.
    retries : int
        Retry attempts on failure.

    Returns
    -------
    DataFrame with columns: fips, date, dsci. Or None on failure.
    """
    params = {
        "aoi": fips,
        "startdate": START_DATE,
        "enddate": END_DATE,
        "statisticsType": "1",
    }

    for attempt in range(1, retries + 1):
        try:
            log.info(
                "  Fetching %s (%s) attempt %d/%d...",
                county_name,
                fips,
                attempt,
                retries,
            )
            response = requests.get(USDM_API_URL, params=params, timeout=60)
            response.raise_for_status()

            # Parse CSV response
            df = pd.read_csv(StringIO(response.text))

            if df.empty:
                log.warning("  Empty response for %s.", county_name)
                return None

            log.info("  %s: %d weekly rows received.", county_name, len(df))

            # Standardize columns
            df = df.rename(
                columns={
                    "FIPS": "fips",
                    "MapDate": "date",
                    "DSCI": "dsci",
                }
            )

            # Parse date — format is YYYYMMDD (e.g., 20000104)
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

            # Coerce DSCI to numeric
            df["dsci"] = pd.to_numeric(df["dsci"], errors="coerce")

            # Drop rows with bad dates or DSCI
            df = df.dropna(subset=["date", "dsci"])

            # Keep only needed columns
            df["fips"] = fips
            return df[["fips", "date", "dsci"]].reset_index(drop=True)

        except requests.exceptions.RequestException as e:
            log.warning("  Attempt %d failed: %s", attempt, e)
            if attempt == retries:
                log.error("  All attempts failed for %s.", county_name)
                return None
            time.sleep(2**attempt)

    return None


def _fetch_all_counties() -> pd.DataFrame:
    """
    Fetch DSCI for all eight counties and combine into a single table.

    Returns
    -------
    DataFrame with columns: fips, date, dsci.
    """
    log.info("Fetching DSCI for all eight counties...")

    frames = []
    for fips, county_name in zip(COUNTY_FIPS, COUNTIES, strict=False):
        df = _fetch_county_dsci(fips, county_name)
        if df is not None and not df.empty:
            frames.append(df)
        # Rate limit — be polite to the API
        time.sleep(0.5)

    if not frames:
        raise RuntimeError(
            "Could not fetch DSCI data for any county.\n"
            "Check network access and USDM API availability at:\n"
            f"{USDM_API_URL}"
        )

    combined = pd.concat(frames, ignore_index=True)

    # Report coverage
    counties_found = combined["fips"].nunique()
    if counties_found < len(COUNTY_FIPS):
        missing = set(COUNTY_FIPS) - set(combined["fips"].unique())
        missing_names = [COUNTIES[COUNTY_FIPS.index(f)] for f in missing]
        log.warning(
            "Missing DSCI data for %d counties: %s",
            len(missing),
            ", ".join(missing_names),
        )
    else:
        log.info("All eight counties retrieved successfully.")

    log.info(
        "Combined: %d total weekly rows, %d counties, " "date range %s to %s.",
        len(combined),
        counties_found,
        combined["date"].min().strftime("%Y-%m-%d"),
        combined["date"].max().strftime("%Y-%m-%d"),
    )

    return combined


# ---------------------------------------------------------------------------
# Area-weighted aggregation
# ---------------------------------------------------------------------------


def _compute_area_weighted_dsci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute area-weighted regional DSCI from county-level data.

    For each week (MapDate), the regional DSCI is:
        regional_dsci = sum(county_dsci * county_area) / total_area

    Parameters
    ----------
    df : DataFrame
        Columns: fips, date, dsci.

    Returns
    -------
    DataFrame with columns: date, dsci_weighted.
    """
    df = df.copy()

    # Add area weight for each county
    df["area"] = df["fips"].map(COUNTY_AREAS_SQMI)
    df["weighted_dsci"] = df["dsci"] * df["area"]

    # Group by date and compute area-weighted mean
    weekly = (
        df.groupby("date")
        .agg(
            total_weighted_dsci=("weighted_dsci", "sum"),
            total_area=("area", "sum"),
            county_count=("fips", "nunique"),
        )
        .reset_index()
    )

    weekly["dsci_weighted"] = (
        weekly["total_weighted_dsci"] / weekly["total_area"]
    ).round(2)

    log.info(
        "Area-weighted weekly DSCI: %d weeks, "
        "mean counties/week=%.1f, "
        "mean DSCI=%.1f, max DSCI=%.1f.",
        len(weekly),
        weekly["county_count"].mean(),
        weekly["dsci_weighted"].mean(),
        weekly["dsci_weighted"].max(),
    )

    return weekly[["date", "dsci_weighted"]].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Monthly aggregation
# ---------------------------------------------------------------------------


def _aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly area-weighted DSCI to monthly mean.

    Parameters
    ----------
    df : DataFrame
        Columns: date, dsci_weighted.

    Returns
    -------
    DataFrame with columns: year_month, usdm_dsci.
    """
    df = df.copy()
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    monthly = (
        df.groupby("year_month")["dsci_weighted"]
        .mean()
        .reset_index()
        .rename(columns={"dsci_weighted": "usdm_dsci"})
        .sort_values("year_month")
        .reset_index(drop=True)
    )

    monthly["usdm_dsci"] = monthly["usdm_dsci"].round(2)

    log.info(
        "Monthly aggregation: %d months (%s to %s), " "mean DSCI=%.1f, max DSCI=%.1f.",
        len(monthly),
        monthly["year_month"].iloc[0],
        monthly["year_month"].iloc[-1],
        monthly["usdm_dsci"].mean(),
        monthly["usdm_dsci"].max(),
    )

    return monthly


# ---------------------------------------------------------------------------
# Water stress score
# ---------------------------------------------------------------------------


def _derive_water_stress_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive water stress score from DSCI. [1][2]

    Formula: water_stress_score = 100 - (usdm_dsci / 5)
    """
    df = df.copy()
    df["water_stress_score"] = (
        STRESS_BASELINE - (df["usdm_dsci"] / DSCI_DIVISOR)
    ).round(2)

    df["water_stress_score"] = df["water_stress_score"].clip(0, 100)

    log.info(
        "Water stress score: range [%.1f, %.1f], mean=%.1f.",
        df["water_stress_score"].min(),
        df["water_stress_score"].max(),
        df["water_stress_score"].mean(),
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
            start=f"{START_MONTH}-01",
            end=f"{END_MONTH}-01",
            freq="MS",
        )
        .strftime("%Y-%m")
        .tolist()
    )

    present = set(df["year_month"].tolist())
    missing = [m for m in all_months if m not in present]

    if not missing:
        log.info("No gaps — all %d expected months present.", len(all_months))
        return df

    log.warning(
        "%d months missing: %s%s",
        len(missing),
        missing[:5],
        "..." if len(missing) > 5 else "",  # noqa: PLR2004
    )

    complete = pd.DataFrame({"year_month": all_months})
    merged = complete.merge(df, on="year_month", how="left")

    merged["usdm_dsci"] = merged["usdm_dsci"].interpolate(
        method="linear", limit=3, limit_direction="both"
    )
    merged["water_stress_score"] = merged["water_stress_score"].interpolate(
        method="linear", limit=3, limit_direction="both"
    )

    filled = merged["usdm_dsci"].notna().sum() - len(df)
    if filled > 0:
        log.info("Filled %d gap months via linear interpolation.", filled)

    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------


def _sanity_checks(df: pd.DataFrame) -> None:
    """Run basic sanity checks."""

    expected = 288
    if len(df) != expected:
        log.warning("Expected %d rows but got %d.", expected, len(df))
    else:
        log.info("Row count correct: %d monthly rows.", len(df))

    nulls = df["usdm_dsci"].isna().sum()
    if nulls:
        log.warning("%d null DSCI values.", nulls)

    valid = df["usdm_dsci"].dropna()
    if valid.min() < 0 or valid.max() > 500:  # noqa: PLR2004
        log.warning("DSCI outside [0, 500]: [%.1f, %.1f].", valid.min(), valid.max())
    else:
        log.info("DSCI range check passed: [%.1f, %.1f].", valid.min(), valid.max())

    # Known drought years — DSCI should be elevated
    drought_years = ["2002", "2007", "2011", "2012", "2020", "2021"]
    for yr in drought_years:
        yr_data = df[df["year_month"].str.startswith(yr)]["usdm_dsci"]
        if not yr_data.empty and yr_data.mean() < 50:  # noqa: PLR2004
            log.warning(
                "Year %s mean DSCI=%.1f — low for a known drought year.",
                yr,
                yr_data.mean(),
            )

    overall_mean = valid.mean()
    log.info("Overall mean DSCI=%.1f.", overall_mean)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(output_file: Path = OUTPUT_FILE) -> pd.DataFrame:
    """
    Full water stress pipeline:
        fetch county-level DSCI → area-weighted aggregation →
        monthly mean → water stress score → fill gaps →
        sanity checks → write CSV.
    """
    log.info("=== water_stress.py start ===")

    # 1. Fetch DSCI for all eight counties from USDM API
    county_weekly = _fetch_all_counties()

    # 2. Compute area-weighted regional DSCI
    regional_weekly = _compute_area_weighted_dsci(county_weekly)

    # 3. Aggregate to monthly
    monthly = _aggregate_to_monthly(regional_weekly)

    # 4. Derive water stress score [1][2]
    monthly = _derive_water_stress_score(monthly)

    # 5. Fill gaps
    monthly = _check_and_fill_gaps(monthly)

    # 6. Select output columns
    output = monthly[
        [
            "year_month",
            "usdm_dsci",
            "water_stress_score",
        ]
    ].copy()

    # 7. Sanity checks
    _sanity_checks(output)

    # 8. Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)
    log.info("Wrote %d rows to %s", len(output), output_file)

    log.info("=== water_stress.py complete ===")
    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build monthly area-weighted water stress score for "
        "the eight-county southern Arizona study area from "
        "USDM county-level DSCI data."
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

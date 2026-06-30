"""
Download USGS surface water level data for eight southern Arizona counties (2000-2020).

Uses the USGS NWIS Daily Values service:
https://waterservices.usgs.gov/nwis/dv/

Pulls daily mean discharge (cfs) and gage height (ft) for all active stream
gages in the eight-county region, then aggregates to monthly means per site
and a single regional monthly indicator.
"""

import time
from pathlib import Path

import pandas as pd
import requests
from region import COUNTY_FIPS

# USGS parameter codes
# 00060 = Discharge (cubic feet per second)
# 00065 = Gage height (feet)
PARAMETER_CODES = ["00060", "00065"]

YEAR_START = 2000
YEAR_END = 2020

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "Final"
OUTPUT_DAILY = OUTPUT_DIR / "water_surface_daily_8county_2000_2020.csv"
OUTPUT_MONTHLY = OUTPUT_DIR / "water_surface_monthly.csv"

REQUEST_DELAY = 0.5


BASE_URL = "https://waterservices.usgs.gov/nwis/dv/"


def fetch_daily_values(
    county: str, param_cd: str, start: str, end: str
) -> pd.DataFrame:
    """
    Fetch daily values for one county and parameter over a date range.
    Uses JSON format to avoid column-count issues with multi-site RDB responses.
    Returns a DataFrame with columns: site_no, datetime, value, param_cd.
    """
    params = {
        "format": "json",
        "countyCd": county,
        "parameterCd": param_cd,
        "startDT": start,
        "endDT": end,
        "statCd": "00003",  # Daily mean
        "siteType": "ST",  # Stream sites only
        "siteStatus": "all",
    }

    resp = requests.get(BASE_URL, params=params, timeout=300)

    if resp.status_code == 404:  # noqa: PLR2004
        return pd.DataFrame()
    resp.raise_for_status()

    data = resp.json()
    time_series = data.get("value", {}).get("timeSeries", [])
    if not time_series:
        return pd.DataFrame()

    rows = []
    for series in time_series:
        site_code = series["sourceInfo"]["siteCode"][0]["value"]
        for value_set in series.get("values", []):
            for record in value_set.get("value", []):
                val = record.get("value")
                if val in (None, "", "-999999"):
                    continue
                rows.append(
                    {
                        "site_no": site_code,
                        "datetime": record["dateTime"][:10],
                        "value": float(val),
                        "param_cd": param_cd,
                    }
                )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    result["county_fips"] = county
    return result


def main() -> None:
    all_frames = []

    for year in range(YEAR_START, YEAR_END + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        for county in COUNTY_FIPS:
            for param in PARAMETER_CODES:
                label = f"{year} | county {county} | param {param}"
                print(f"  Fetching {label}...", end=" ", flush=True)

                try:
                    df = fetch_daily_values(county, param, start_date, end_date)
                    n = len(df)
                    print(f"{n} rows")
                    if n > 0:
                        all_frames.append(df)
                except requests.HTTPError as e:
                    print(f"HTTP error: {e}")
                except Exception as e:
                    print(f"Error: {e}")

                time.sleep(REQUEST_DELAY)

    if not all_frames:
        print("No data retrieved. Check parameters and try again.")
        return

    # Combine all daily data
    daily = pd.concat(all_frames, ignore_index=True)
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="coerce")
    daily = daily.dropna(subset=["datetime"])
    daily.to_csv(OUTPUT_DAILY, index=False)
    print(f"\nSaved {len(daily):,} daily rows to '{OUTPUT_DAILY}'")

    # Aggregate to monthly means per parameter
    daily["year_month"] = daily["datetime"].dt.to_period("M").astype(str)

    monthly = (
        daily.groupby(["year_month", "param_cd"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "regional_mean"})
    )

    # Pivot so each parameter is its own column
    monthly_wide = monthly.pivot_table(
        index="year_month",
        columns="param_cd",
        values="regional_mean",
    ).reset_index()
    monthly_wide.columns.name = None

    # Rename parameter columns to readable names
    col_names = {"00060": "discharge_cfs_mean", "00065": "gage_height_ft_mean"}
    monthly_wide = monthly_wide.rename(columns=col_names)

    monthly_wide.to_csv(OUTPUT_MONTHLY, index=False)
    print(f"Saved {len(monthly_wide):,} monthly rows to '{OUTPUT_MONTHLY}'")
    print(
        f"Date range: {monthly_wide['year_month'].iloc[0]} "
        f"to {monthly_wide['year_month'].iloc[-1]}"
    )
    print(f"Columns: {list(monthly_wide.columns)}")


if __name__ == "__main__":
    main()

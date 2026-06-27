"""
Download USGS groundwater level data for eight southern Arizona counties (2000-2020).

Uses the USGS NWIS Daily Values service:
https://waterservices.usgs.gov/nwis/dv/

Pulls daily mean depth-to-water-level (ft below land surface) from monitoring
wells in the eight-county region, then aggregates to monthly means for a
regional groundwater level indicator.

This complements GRACE satellite data by providing in-situ well readings.
"""

import time
from pathlib import Path

import pandas as pd
import requests
from region import COUNTY_FIPS

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# USGS parameter codes for groundwater levels
# 72019 = Depth to water level, ft below land surface
# 72008 = Depth to water in well, periodic measurement, ft
# 62610 = Groundwater level above NGVD 1929, feet
# 62611 = Groundwater level above NAVD 1988, feet
PARAMETER_CODES = ["72019", "72008", "62610", "62611"]

# Date range
YEAR_START = 2000
YEAR_END = 2020

# Output files
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "Final"
OUTPUT_DAILY = OUTPUT_DIR / "groundwater_levels_daily_2000_2020.csv"
OUTPUT_MONTHLY = OUTPUT_DIR / "groundwater_levels_monthly.csv"

# Seconds between API requests
REQUEST_DELAY = 0.5

# -------------------------------------------------------------------
# NWIS Daily Values endpoint
# -------------------------------------------------------------------

BASE_URL = "https://waterservices.usgs.gov/nwis/dv/"


def fetch_daily_values(
    county: str, param_cd: str, start: str, end: str
) -> pd.DataFrame:
    """
    Fetch daily groundwater values for one county and parameter over a date range.
    Uses JSON format for reliable multi-site parsing.
    Returns a DataFrame with columns: site_no, datetime, value, param_cd.
    """
    params = {
        "format": "json",
        "countyCd": county,
        "parameterCd": param_cd,
        "startDT": start,
        "endDT": end,
        "statCd": "00003",  # Daily mean
        "siteType": "GW",  # Groundwater wells only
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_DAILY, index=False)
    print(f"\nSaved {len(daily):,} daily rows to '{OUTPUT_DAILY}'")
    print(f"  Unique wells: {daily['site_no'].nunique()}")

    # Aggregate to monthly regional mean (using param 72019 as primary)
    daily["year_month"] = daily["datetime"].dt.to_period("M").astype(str)

    # Prefer depth-to-water (72019) if available, fall back to others
    depth_data = daily[daily["param_cd"] == "72019"]
    if len(depth_data) > 0:
        monthly = (
            depth_data.groupby("year_month")["value"]
            .mean()
            .reset_index()
            .rename(columns={"value": "depth_to_water_ft_mean"})
        )
    else:
        # Use whatever parameter has the most data
        monthly = (
            daily.groupby("year_month")["value"]
            .mean()
            .reset_index()
            .rename(columns={"value": "groundwater_level_mean"})
        )

    monthly.to_csv(OUTPUT_MONTHLY, index=False)
    print(f"Saved {len(monthly):,} monthly rows to '{OUTPUT_MONTHLY}'")
    print(
        f"Date range: {monthly['year_month'].iloc[0]} "
        f"to {monthly['year_month'].iloc[-1]}"
    )
    print(f"Columns: {list(monthly.columns)}")


if __name__ == "__main__":
    main()

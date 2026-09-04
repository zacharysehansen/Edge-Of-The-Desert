"""
Download USGS surface water level data for eight southern Arizona counties (1980-2025).

Uses the USGS NWIS Daily Values service:
https://waterservices.usgs.gov/nwis/dv/

Pulls daily mean discharge (cfs) and gage height (ft) for all active stream
gages in the eight-county region, then aggregates to a regional monthly indicator.

**The regional indicator is a per-gage anomaly index, not a mean of raw discharge.**
A plain mean over whatever gages reported that month has two defects. The first is
compositional: the target moves when the *reporting set* changes, not when the water
does. On the 2000-2020 network that effect is small (gage turnover correlates -0.02
with the month-to-month jump), but the network of 1980 is not the network of 2020, so
any backward extension of YEAR_START would import it at full strength. The second bites
today: discharge spans four orders of magnitude across gages, so a raw mean is really
just the largest gage. Gage 09525503 alone contributes 45% of it.

Both are fixed the same way. Each gage is converted to log space, centered on its own
long-term mean, and only then averaged across gages. The result is dimensionless: "how
high is flow this month relative to normal, averaged over the region", where a small
wash and a major river count equally.

Note the centering constant is a full-record mean, so it is not available in real time.
That is standard for a station anomaly index and it is a per-gage *constant* -- it does
not leak any month-specific information into another month -- but it does mean the index
is defined relative to the record it was built from.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from region import COUNTY_FIPS

# USGS parameter codes
# 00060 = Discharge (cubic feet per second)
# 00065 = Gage height (feet)
PARAMETER_CODES = ["00060", "00065"]

# NWIS has discharge for these counties from at least 1980 (probed: 27-38 gages in
# Pima/Pinal/Cochise in every sampled year 1980-1995) through 2025. The old 2000-2020
# window was a project convention, not a source limit.  Reaching back is only safe
# because the target is now a per-gage anomaly index. The gage roster of 1980 is not
# the roster of 2020; a mean of raw discharge across that span would have amplified
# the compositional artifact, not diluted it. See the module docstring.
YEAR_START = 1980
YEAR_END = 2025

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "Final"
OUTPUT_DAILY = OUTPUT_DIR / "water_surface_daily_8county_1980_2025.csv"
OUTPUT_MONTHLY = OUTPUT_DIR / "water_surface_monthly.csv"

REQUEST_DELAY = 0.5

DISCHARGE_CD = "00060"
GAGE_HEIGHT_CD = "00065"

# A gage needs enough of a record for its own long-term mean to mean anything.
# Below this it is dropped from the anomaly index (it still counts toward the raw mean).
MIN_MONTHS_PER_GAGE = 24


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


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily per-gage values into one regional monthly row.

    Emits both target constructions so they can be compared:
      discharge_log_anomaly  -- the real target. Per-gage log anomaly, averaged.
      discharge_cfs_mean     -- the old raw mean. Retained for the lag1 baseline,
                                the frontend's level display, and A/B comparison.
      n_gages                -- how many gages reported. The diagnostic that exposes
                                compositional churn; without it it is invisible.
    """
    daily = daily.copy()
    daily["param_cd"] = daily["param_cd"].astype(str).str.zfill(5)
    daily["site_no"] = daily["site_no"].astype(str)
    daily["year_month"] = daily["datetime"].dt.to_period("M").astype(str)

    # Per gage, per month, first -- so a gage reporting daily does not outvote one
    # reporting weekly.
    site_month = (
        daily.groupby(["year_month", "param_cd", "site_no"])["value"]
        .mean()
        .reset_index()
    )

    raw = (
        site_month.groupby(["year_month", "param_cd"])["value"]
        .mean()
        .unstack("param_cd")
        .rename(
            columns={
                DISCHARGE_CD: "discharge_cfs_mean",
                GAGE_HEIGHT_CD: "gage_height_ft_mean",
            }
        )
    )
    raw.columns.name = None

    q = site_month[site_month["param_cd"] == DISCHARGE_CD].copy()

    # Drop short-record gages: a long-term mean over 3 months is not a long-term mean.
    counts = q.groupby("site_no")["year_month"].transform("size")
    q = q[counts >= MIN_MONTHS_PER_GAGE]

    # Log space, because discharge is multiplicative and spans four orders of magnitude
    # across gages. A doubling on a small wash should count the same as a doubling on
    # the Gila.
    q["log_q"] = np.log1p(q["value"].clip(lower=0.0))
    q["log_anomaly"] = q["log_q"] - q.groupby("site_no")["log_q"].transform("mean")

    index = q.groupby("year_month").agg(
        discharge_log_anomaly=("log_anomaly", "mean"),
        n_gages=("site_no", "nunique"),
    )

    monthly = index.join(raw, how="outer").reset_index()
    return monthly.sort_values("year_month").reset_index(drop=True)


def _load_cached_daily() -> pd.DataFrame | None:
    """Reuse the daily pull if it is already on disk. The NWIS fetch is slow and the
    aggregation is the part that changes."""
    if not OUTPUT_DAILY.exists():
        return None
    daily = pd.read_csv(OUTPUT_DAILY, dtype={"site_no": str, "param_cd": str})
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="coerce")
    return daily.dropna(subset=["datetime"])


def main(refetch: bool = False) -> None:
    if not refetch:
        cached = _load_cached_daily()
        if cached is not None:
            print(
                f"Using cached daily pull: {len(cached):,} rows "
                f"from '{OUTPUT_DAILY}'"
            )
            print("  (pass refetch=True to re-download from NWIS)")
            _write_monthly(aggregate_monthly(cached))
            return

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

    _write_monthly(aggregate_monthly(daily))


def _write_monthly(monthly: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_MONTHLY, index=False)
    print(f"Saved {len(monthly):,} monthly rows to '{OUTPUT_MONTHLY}'")
    print(
        f"Date range: {monthly['year_month'].iloc[0]} "
        f"to {monthly['year_month'].iloc[-1]}"
    )
    print(f"Columns: {list(monthly.columns)}")
    print(
        f"  gages reporting/month: {monthly['n_gages'].min():.0f}"
        f"-{monthly['n_gages'].max():.0f} (mean {monthly['n_gages'].mean():.1f})"
    )


if __name__ == "__main__":
    main()

"""
Download USGS groundwater level data for eight southern Arizona counties (2000-2020).

Uses the USGS NWIS Daily Values service:
https://waterservices.usgs.gov/nwis/dv/

Pulls daily mean depth-to-water-level (ft below land surface) from monitoring
wells in the eight-county region, then aggregates to a regional monthly indicator.

**The regional indicator is a per-well anomaly index, not a mean of raw depths.**
Wells sit at wildly different depths -- one well's water table is 40 ft down, another's
is 400 -- and only some of them report in any given month. A plain mean over "whatever
wells reported" therefore moves when the *reporting set* changes, with no water having
moved at all. This is not hypothetical: on the 2000-2020 pull, well turnover between
consecutive months correlates **+0.75** with the jump in the raw mean, and the raw
mean's standard deviation (27.8 ft) is nearly five times the anomaly's (5.9 ft). Most
of what the old target was "measuring" was its own roster.

That artifact is physically unlearnable, and it is why the groundwater model scored
*worse than predicting no change at all* (residual R^2 -0.32): the only way to fit a
roster is to predict nothing, so the optimizer drove regularization to the ceiling.

The fix is to subtract each well's own long-term mean before averaging, which is the
standard construction for a changing station network. The result answers "is the water
table high or low relative to normal, across the region", and a well joining or leaving
the roster no longer shifts it.

Sign convention follows the raw measurement: this is *depth to water*, so **positive
means deeper than that well's normal, i.e. less groundwater**.

**The index is over the COCHISE COUNTY wells (2026-09-12, PHASE3_PLAN.md §28-§29).**
The daily pull holds 66 wells: 44 in Cochise County (the Willcox and Douglas basins,
agricultural pumping, no CAP water), 14 in Pima (the Tucson AMA, under managed
recharge), and 8 elsewhere. The two big groups are uncorrelated (level r = -0.13,
month-to-month r = +0.07), and the ten Pima wells, six times as volatile, supplied
81% of the blended index's monthly variance while responding to nothing in the
panel. The Cochise sub-index responds to storage change, rain and pumping with the
physical signs and forecasts (skill +0.23); the blend forecast nothing for five
re-runs. So `depth_to_water_anomaly_ft` is now the Cochise index, and the old blend is
kept beside it as `depth_to_water_anomaly_ft_allwells` for the record. The output is
therefore "well depth vs normal, Willcox-Douglas basins", and PHASE3_PLAN.md §29
removes the Lake Mead lever from it: no CAP water reaches those basins.

Note the centering constant is a full-record mean, so it is not available in real time.
That is standard for a station anomaly index, and it is a per-well *constant* -- it
leaks no month-specific information across months -- but the index is defined relative
to the record it was built from.

This complements GRACE satellite data by providing in-situ well readings.
"""

import time
from pathlib import Path

import pandas as pd
import requests
from region import COUNTY_FIPS

# USGS parameter codes for groundwater levels
# 72019 = Depth to water level, ft below land surface
# 72008 = Depth to water in well, periodic measurement, ft
# 62610 = Groundwater level above NGVD 1929, feet
# 62611 = Groundwater level above NAVD 1988, feet
PARAMETER_CODES = ["72019", "72008", "62610", "62611"]

YEAR_START = 2000
YEAR_END = 2020

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "Final"
OUTPUT_DAILY = OUTPUT_DIR / "groundwater_levels_daily_2000_2020.csv"
OUTPUT_MONTHLY = OUTPUT_DIR / "groundwater_levels_monthly.csv"

REQUEST_DELAY = 0.5

# 72019 = depth to water level, ft below land surface. The only code with real coverage,
# and the only one on a common datum -- the 626xx codes are elevations above a geodetic
# datum, so mixing them into one mean would be meaningless.
DEPTH_CD = "72019"

# A well needs enough of a record for its own long-term mean to mean anything.
MIN_MONTHS_PER_WELL = 24
# The county whose wells define the target (see the docstring). 04003 = Cochise.
TARGET_COUNTY_FIPS = 4003


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


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily per-well depths into one regional monthly row.

    Emits both target constructions so they can be compared:
      depth_to_water_anomaly_ft -- the real target. Per-well anomaly, averaged.
                                   Positive = deeper than normal = less groundwater.
      depth_to_water_ft_mean    -- the old raw mean. Retained for the lag1 baseline,
                                   the frontend's level display, and A/B comparison.
      n_wells                   -- how many wells reported. The diagnostic that exposes
                                   the churn; without it the artifact is invisible.
    """
    daily = daily.copy()
    daily["param_cd"] = daily["param_cd"].astype(str)
    daily["site_no"] = daily["site_no"].astype(str)
    daily["year_month"] = daily["datetime"].dt.to_period("M").astype(str)

    depth = daily[daily["param_cd"] == DEPTH_CD]
    if depth.empty:
        raise ValueError(
            f"No param_cd={DEPTH_CD} (depth to water) rows found. The other "
            "groundwater codes are elevations on a geodetic datum and cannot be "
            "averaged together."
        )

    # Per well, per month, first -- so a well reporting daily does not outvote one
    # reporting once.
    well_month = depth.groupby(["site_no", "year_month"])["value"].mean().reset_index()

    raw = (
        well_month.groupby("year_month")["value"].mean().rename("depth_to_water_ft_mean")
    )

    counts = well_month.groupby("site_no")["year_month"].transform("size")
    long_record = well_month[counts >= MIN_MONTHS_PER_WELL].copy()

    long_record["anomaly"] = long_record["value"] - long_record.groupby("site_no")[
        "value"
    ].transform("mean")

    # The target: Cochise wells only. The blend over every county is kept as a
    # second column so the two can always be compared (PHASE3_PLAN.md §28).
    county_of = depth.groupby("site_no")["county_fips"].first().astype(int)
    long_record["county_fips"] = long_record["site_no"].map(county_of)
    cochise = long_record[long_record["county_fips"] == TARGET_COUNTY_FIPS]
    if cochise.empty:
        raise ValueError(f"No wells with county_fips == {TARGET_COUNTY_FIPS} in the daily pull.")

    target = cochise.groupby("year_month").agg(
        depth_to_water_anomaly_ft=("anomaly", "mean"),
        n_wells=("site_no", "nunique"),
    )
    blend = long_record.groupby("year_month").agg(
        depth_to_water_anomaly_ft_allwells=("anomaly", "mean"),
        n_wells_all=("site_no", "nunique"),
    )
    monthly = target.join(blend, how="outer").join(raw, how="outer").reset_index()
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

    daily = pd.concat(all_frames, ignore_index=True)
    daily["datetime"] = pd.to_datetime(daily["datetime"], errors="coerce")
    daily = daily.dropna(subset=["datetime"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_DAILY, index=False)
    print(f"\nSaved {len(daily):,} daily rows to '{OUTPUT_DAILY}'")
    print(f"  Unique wells: {daily['site_no'].nunique()}")

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
        f"  wells reporting/month: {monthly['n_wells'].min():.0f}"
        f"-{monthly['n_wells'].max():.0f} (mean {monthly['n_wells'].mean():.1f})"
    )


if __name__ == "__main__":
    main()

"""
wildlife_bbs.py
---------------
Loads manually downloaded BBS route metadata and Arizona count data,
filters to routes falling inside the eight-county southern Arizona
study area, and builds an annual wildlife abundance index.

Input  : data/raw/bbs/Routes.csv      (latin-1 encoding)
         data/raw/bbs/Arizona.csv     (utf-8 encoding)
Output : data/processed/wildlife_annual.csv

Columns in output:
    year              - int, calendar year
    route_count       - int, number of BBS routes active in the region
    total_abundance   - int, pooled bird count across all species and routes
    species_richness  - int, unique species observed across all routes
    abundance_anomaly - float, THE TARGET. Per-route log-abundance anomaly, averaged.
    abundance_index   - float, log1p(total_abundance) min-max normalized. LEGACY —
                        do not model on this. See below.

**Why `total_abundance` is not a measure of bird abundance.**

It is a *sum over whichever routes were surveyed that year*, and the number of
routes surveyed is not constant. In the eight-county region it swings between 15
and 26 per year; statewide the BBS ran ~15 routes/year in the 1970s-80s and ~45
from 1990 onward. So the sum rises and falls with **survey effort**, not birds.

Measured on the 2000-2024 region output, `route_count` correlates:

    route_count  vs  total_abundance    r = +0.840  (p = 2.8e-07)
    route_count  vs  abundance_index    r = +0.861  (p = 6.6e-08)
    route_count  vs  species_richness   r = +0.866  (p = 4.8e-08)

The old `abundance_index` was **86% correlated with how many people went
birdwatching**. It is a survey-effort index wearing a bird costume, and it is
physically unlearnable from climate.

Effort-correcting drops that to r = +0.324 (p = 0.12, not significant), and the
real signal survives: bird abundance per route declines ~23% across the record
(p < 0.0001). The decline is genuine; the old target simply conflated it with
routes being dropped.

**The fix: `abundance_anomaly`.** Each route is centered on its *own* long-term
mean log abundance before averaging, so (a) a year with more routes is not
automatically a bigger year, and (b) a productive riparian route and a sparse
desert route contribute equally. This is the same construction used for the well
and gage networks in `groundwater_levels.py` and `water_surface.py` — a mean over
a churning station roster is never a measurement of what the stations measure.

This also had to land **before** the window was extended to 1968. Route counts
triple around 1990, so summing across that would manufacture a spurious "bird
abundance tripled" step that is purely BBS adding routes.

`abundance_index` is retained for comparison only. It is additionally min-max
normalized over the full series, which is look-ahead (the same class as the
`consecutive_dry_years` bug).

Source files downloaded from:
    https://www.sciencebase.gov/catalog/item/691cfb53d4be021d1d89b482
    2025 Release - North American Breeding Bird Survey Dataset (1966-2024)

Confirmed column structure:
    Routes.csv : CountryNum, StateNum, Route, RouteName, Active,
                 Latitude, Longitude, Stratum, BCR,
                 RouteTypeID, RouteTypeDetailID
    Arizona.csv: RouteDataID, CountryNum, StateNum, Route, RPID,
                 Year, AOU, Count10, Count20, Count30, Count40,
                 Count50, StopTotal, SpeciesTotal

Count column used: SpeciesTotal
    SpeciesTotal is the sum across all 50 stops for a given species
    on a given route in a given year. This is the correct column for
    total abundance aggregation. StopTotal is the sum across species
    at a single stop — less useful for annual regional totals.

Missing year: 2020
    BBS field activity was cancelled. Excluded from output rather
    than filled with zeros. [2]

Date range: 1968-01-01 to 2024-12-31
"""

import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import (
    filter_points,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw" / "bbs"
PROCESSED_DIR = ROOT / "data" / "Final"
ROUTES_FILE = RAW_DIR / "Routes.csv"
COUNTS_FILE = RAW_DIR / "Arizona.csv"
OUTPUT_FILE = PROCESSED_DIR / "wildlife_annual.csv"

# BBS Arizona counts run 1968-2024. The old 2000 floor was a project convention, not a
# source limit, and it discarded ~30k rows. The anomaly target below is what makes using
# them safe — see the module docstring.
START_YEAR = 1968
END_YEAR = 2024
BBS_CANCELLED_YEARS = [2020]

# A route needs enough of a record for its own long-term mean to mean anything.
# Routes below this are dropped from the anomaly index (they still count toward the
# raw diagnostics).
MIN_YEARS_PER_ROUTE = 8

# Confirmed column names from the inspection output
COUNT_COL = "SpeciesTotal"  # sum across all 50 stops per species per route
YEAR_COL = "Year"
SPECIES_COL = "AOU"
STATE_COL = "StateNum"
ROUTE_COL = "Route"
LAT_COL = "Latitude"
LON_COL = "Longitude"


def _load_routes(path: Path) -> pd.DataFrame:
    """
    Load BBS routes metadata.

    Routes.csv uses latin-1 encoding — confirmed from the inspection
    run which found a 0xd1 byte that is invalid in utf-8 but valid in
    latin-1 (likely a special character in a RouteName field).

    Raises
    ------
    FileNotFoundError
        If Routes.csv is not present at the expected path.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Routes file not found at {path}.\n"
            "Download Routes.zip from:\n"
            "https://www.sciencebase.gov/catalog/item/691cfb53d4be021d1d89b482\n"
            "and place the unzipped Routes.csv in data/raw/bbs/"
        )

    df = pd.read_csv(path, encoding="latin-1")
    log.info("Loaded %d routes from %s", len(df), path.name)

    required = [STATE_COL, ROUTE_COL, LAT_COL, LON_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Routes file missing required columns: {missing}\n"
            f"Available: {df.columns.tolist()}"
        )

    return df


def _load_counts(path: Path) -> pd.DataFrame:
    """
    Load BBS Arizona count data.

    Arizona.csv uses utf-8 encoding — confirmed from inspection.

    Raises
    ------
    FileNotFoundError
        If Arizona.csv is not present at the expected path.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Counts file not found at {path}.\n"
            "Download the Arizona state count file from:\n"
            "https://www.sciencebase.gov/catalog/item/691cfb53d4be021d1d89b482\n"
            "and place the unzipped Arizona.csv in data/raw/bbs/"
        )

    df = pd.read_csv(path, encoding="utf-8")
    log.info("Loaded %d count rows from %s", len(df), path.name)

    required = [STATE_COL, ROUTE_COL, YEAR_COL, SPECIES_COL, COUNT_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Counts file missing required columns: {missing}\n"
            f"Available: {df.columns.tolist()}"
        )

    return df


def _filter_routes_to_region(routes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter BBS routes to those whose coordinates fall inside the
    eight-county boundary using region.filter_points().

    Parameters
    ----------
    routes_df : DataFrame
        Full BBS routes table with Latitude and Longitude columns.

    Returns
    -------
    Filtered DataFrame containing only routes inside the study area.

    Raises
    ------
    ValueError
        If no routes are found inside the boundary.
    """
    pre = len(routes_df)

    geometry = [
        Point(lon, lat)
        for lon, lat in zip(routes_df[LON_COL], routes_df[LAT_COL], strict=False)
    ]
    gdf = gpd.GeoDataFrame(routes_df, geometry=geometry, crs="EPSG:4326")

    filtered = filter_points(gdf)

    log.info(
        "Route spatial filter: %d total → %d inside eight-county "
        "boundary (removed %d).",
        pre,
        len(filtered),
        pre - len(filtered),
    )

    if len(filtered) == 0:
        raise ValueError(
            "No BBS routes found inside the eight-county study area.\n"
            "Check that Latitude/Longitude columns are in decimal degrees "
            "(EPSG:4326) and that the county shapefile loaded correctly."
        )

    # Log the route names for transparency
    if "RouteName" in filtered.columns:
        log.info(
            "Routes in study area:\n%s",
            "\n".join(
                f"  StateNum={row[STATE_COL]} Route={row[ROUTE_COL]} "
                f"Name={row['RouteName']}"
                for _, row in filtered.iterrows()
            ),
        )

    return filtered.drop(columns=["geometry"]).reset_index(drop=True)


def _aggregate_to_annual(
    counts_df: pd.DataFrame,
    regional_routes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join counts to the regional route list and aggregate to annual totals.

    Uses SpeciesTotal as the count column — this is the sum across all
    50 stops for a given species on a given route in a given year.
    Summing SpeciesTotal across all species and routes gives the total
    annual abundance for the region.

    Aggregations:
        route_count      - unique routes with at least one observation
        total_abundance  - sum of SpeciesTotal across all species/routes
        species_richness - count of unique AOU codes observed

    Parameters
    ----------
    counts_df : DataFrame
        Full Arizona count table.
    regional_routes : DataFrame
        Routes filtered to the eight-county study area.

    Returns
    -------
    DataFrame with columns: year, route_count, total_abundance,
    species_richness.
    """
    # Build set of (StateNum, Route) pairs in the region
    regional_keys = set(
        zip(regional_routes[STATE_COL], regional_routes[ROUTE_COL], strict=False)
    )
    log.info(
        "%d unique (StateNum, Route) pairs in the study area.",
        len(regional_keys),
    )

    # Filter counts to regional routes
    mask = list(zip(counts_df[STATE_COL], counts_df[ROUTE_COL], strict=False))
    counts_df = counts_df[pd.Series(mask).isin(regional_keys).values].copy()

    log.info("Count rows after route filter: %d", len(counts_df))

    if counts_df.empty:
        raise ValueError(
            "No count records matched the regional routes.\n"
            "Check that StateNum and Route columns align between "
            "Routes.csv and Arizona.csv."
        )

    # Coerce year to int
    counts_df[YEAR_COL] = pd.to_numeric(counts_df[YEAR_COL], errors="coerce")
    counts_df = counts_df.dropna(subset=[YEAR_COL])
    counts_df[YEAR_COL] = counts_df[YEAR_COL].astype(int)

    # Filter to project date range [3]
    pre = len(counts_df)
    counts_df = counts_df[
        (counts_df[YEAR_COL] >= START_YEAR) & (counts_df[YEAR_COL] <= END_YEAR)
    ]
    log.info(
        "Date range filter (%d–%d): %d → %d rows.",
        START_YEAR,
        END_YEAR,
        pre,
        len(counts_df),
    )

    # Drop cancelled years [2]
    for yr in BBS_CANCELLED_YEARS:
        cancelled = counts_df[YEAR_COL] == yr
        if cancelled.any():
            log.info(
                "Dropping %d rows for cancelled BBS year %d.",
                cancelled.sum(),
                yr,
            )
            counts_df = counts_df[~cancelled]

    # Coerce count column to numeric
    counts_df[COUNT_COL] = pd.to_numeric(counts_df[COUNT_COL], errors="coerce").fillna(
        0
    )

    # Annual aggregation
    annual = (
        counts_df.groupby(YEAR_COL)
        .agg(
            route_count=(ROUTE_COL, "nunique"),
            total_abundance=(COUNT_COL, "sum"),
            species_richness=(SPECIES_COL, "nunique"),
        )
        .reset_index()
        .rename(columns={YEAR_COL: "year"})
        .sort_values("year")
        .reset_index(drop=True)
    )

    log.info(
        "Annual aggregation complete: %d years (%d–%d), "
        "mean routes/year=%.1f, mean abundance/year=%.0f, "
        "mean species/year=%.0f.",
        len(annual),
        annual["year"].min(),
        annual["year"].max(),
        annual["route_count"].mean(),
        annual["total_abundance"].mean(),
        annual["species_richness"].mean(),
    )

    annual = annual.merge(_route_anomaly_index(counts_df), on="year", how="left")

    return annual


def _route_anomaly_index(counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the effort-corrected target: a per-route log-abundance anomaly, averaged.

    Each route is centered on its *own* long-term mean before averaging, so the index
    does
    not move just because more (or more productive) routes were surveyed. Log space,
    because a route's count is multiplicative — a doubling on a sparse desert route
    should
    count the same as a doubling on a rich riparian one.

    Without this the target is a survey-effort index: route_count correlates +0.86
    with the
    old `abundance_index`. See the module docstring.
    """
    per_route = (
        counts_df.groupby([ROUTE_COL, YEAR_COL])[COUNT_COL]
        .sum()
        .reset_index(name="route_abundance")
    )

    # A route needs a real record before its "long-term mean" means anything.
    n_years = per_route.groupby(ROUTE_COL)[YEAR_COL].transform("size")
    kept = per_route[n_years >= MIN_YEARS_PER_ROUTE].copy()

    dropped = per_route[ROUTE_COL].nunique() - kept[ROUTE_COL].nunique()
    log.info(
        "Anomaly index: %d routes kept, %d dropped for <%d years of record.",
        kept[ROUTE_COL].nunique(),
        dropped,
        MIN_YEARS_PER_ROUTE,
    )

    kept["log_abundance"] = np.log1p(kept["route_abundance"])
    kept["anomaly"] = kept["log_abundance"] - kept.groupby(ROUTE_COL)[
        "log_abundance"
    ].transform("mean")

    return (
        kept.groupby(YEAR_COL)["anomaly"]
        .mean()
        .reset_index()
        .rename(columns={YEAR_COL: "year", "anomaly": "abundance_anomaly"})
    )


def _build_abundance_index(annual: pd.DataFrame) -> pd.DataFrame:
    """
    Build the abundance index from log1p(total_abundance),
    normalized to [0, 1] via min-max scaling.

    log1p applied for consistency with wildfire.py — bird counts
    can be right-skewed by exceptional years or high-activity routes.
    """
    df = annual.copy()

    log_abundance = np.log1p(df["total_abundance"])
    lo, hi = log_abundance.min(), log_abundance.max()

    if hi == lo:
        df["abundance_index"] = 0.5
        log.warning(
            "All log_abundance values are identical. "
            "abundance_index set to 0.5 for all years."
        )
    else:
        df["abundance_index"] = ((log_abundance - lo) / (hi - lo)).round(6)

    return df


def _row_count_check(df: pd.DataFrame) -> None:
    """
    Warn if annual row count is low enough to make cross-validation
    unreliable. Mirrors wildfire.py thresholds [1].
    """
    n = len(df)
    very_low_count = 20
    low_count = 50
    if n < very_low_count:
        log.warning(
            "VERY LOW ROW COUNT: only %d annual rows. "
            "XGBoost cross-validation will be unreliable. "
            "Consider a simpler model for the wildlife target.",
            n,
        )
    elif n < low_count:
        log.warning(
            "LOW ROW COUNT: %d annual rows. "
            "Cross-validation splits may be unstable. "
            "Review before training.",
            n,
        )
    else:
        log.info("Row count acceptable for modeling: %d annual rows.", n)


MAX_EFFORT_CORR = 0.6


def _effort_confound_check(df: pd.DataFrame) -> None:
    """
    Refuse to ship a target that is really a measure of survey effort.

    The old `abundance_index` correlated **+0.86** with `route_count` — it was
    tracking how
    many routes were surveyed, not how many birds there were. This is the check that
    would
    have caught that on day one, and it makes the class of error unrepeatable. It is the
    same guard `wildfire_monthly._validate_seasonality` provides for the fire target.
    """
    target, effort = df["abundance_anomaly"], df["route_count"]
    corr = float(np.corrcoef(target, effort)[0, 1])
    log.info("Effort check: corr(abundance_anomaly, route_count) = %+.3f", corr)

    if abs(corr) > MAX_EFFORT_CORR:
        raise ValueError(
            f"abundance_anomaly correlates {corr:+.3f} with route_count "
            f"(limit ±{MAX_EFFORT_CORR}). The target is tracking survey effort, not "
            "birds. Do not model on it — fix the effort correction first."
        )

    legacy = float(np.corrcoef(df["abundance_index"], effort)[0, 1])
    log.info(
        "  (legacy abundance_index correlates %+.3f with route_count — this is why "
        "it is not the target)",
        legacy,
    )


def main() -> None:
    log.info("=== wildlife_bbs.py start ===")

    routes_raw = _load_routes(ROUTES_FILE)
    regional_routes = _filter_routes_to_region(routes_raw)

    counts_raw = _load_counts(COUNTS_FILE)

    annual = _aggregate_to_annual(counts_raw, regional_routes)

    _row_count_check(annual)

    annual = _build_abundance_index(annual)

    output = annual[
        [
            "year",
            "route_count",
            "total_abundance",
            "species_richness",
            "abundance_anomaly",
            "abundance_index",
        ]
    ].copy()

    _effort_confound_check(output)

    if BBS_CANCELLED_YEARS[0] in output["year"].values:
        log.warning("2020 found in output despite cancellation filter. Removing now.")
        output = output[output["year"] != BBS_CANCELLED_YEARS[0]].reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    log.info("Wrote %d rows to %s", len(output), OUTPUT_FILE)

    log.info("=== wildlife_bbs.py complete ===")


if __name__ == "__main__":
    main()

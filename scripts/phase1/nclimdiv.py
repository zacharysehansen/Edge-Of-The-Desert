"""
nclimdiv.py
-----------
NOAA nClimDiv — monthly drought, temperature and precipitation by climate division,
**1895 to present**, area-weighted to the eight-county study region.

Input  : https://www.ncei.noaa.gov/pub/data/cirs/climdiv/  (free, no authentication)
Output : data/Final/nclimdiv_monthly.csv

Columns in output:
    year_month                   - str, YYYY-MM
    nclimdiv_pdsi                - float, Palmer Drought Severity Index (- dry, + wet)
    nclimdiv_temperature_c       - float, mean temperature, degrees Celsius
    nclimdiv_precipitation_mm_day- float, precipitation, mm/day

**Why this file exists.** Three of the panel's inputs sit on hard floors that no amount
of
re-downloading can move:

    usdm_dsci   2000-01   the U.S. Drought Monitor did not exist before it
    MODIS NDVI  2000-02   Terra launch
    GRACE       2002-04   mission start

and MERRA-2 temperature/precipitation, as ingested here, stop at 2000 only by project
convention (the product itself starts in 1980). That 2000 floor is what caps every
annual model at ~20 rows, because a model can only run as far back as its *shortest*
feature.

nClimDiv breaks the floor. NOAA has published monthly climate-division data
continuously since **1895** — drought (PDSI), temperature and precipitation in the
same series, free and unauthenticated. Substituting it for USDM/MERRA lets the annual
models reach back to the limit of their *target* rather than the limit of their
features. For wildlife that is 1968
(the BBS floor), which is n=56 instead of n=20.

**This does not replace `usdm_dsci`.** Keep the USDM where it exists (2000+): it is a
different and arguably better construct, and the variance decomposition in
PHASE2_REPORT.md names it the single highest-signal input in the panel. PDSI is the
long-record *backbone*, so that pre-2000 rows have a drought signal at all.

Area weighting
--------------
Arizona has 7 climate divisions; the region overlaps 4 of them meaningfully:

    SOUTHEAST      42.3%      SOUTHWEST      12.7%
    SOUTH CENTRAL  33.8%      EAST CENTRAL   11.2%

Weights are the true intersected area of each division with the region boundary,
computed in an equal-area projection (EPSG:5070) — not a county-to-division lookup
table. This mirrors how `water_stress.py` area-weights counties, but geometrically
rather than by hand.

Units
-----
The raw files are in US customary units and are converted here:
    tmpc  degrees Fahrenheit  ->  Celsius
    pcpn  inches per month    ->  mm/day  (divided by that month's actual length)
so the columns are directly comparable to `temperature_monthly.csv` and
`precipitation_monthly.csv`. `_validate_against_merra()` enforces that at write time.
"""

from __future__ import annotations

import calendar
import io
import logging
import re
import socket
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from urllib3.util import connection as urllib3_connection

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import load_county_boundary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "data" / "Final"
OUTPUT_FILE = OUTPUT_DIR / "nclimdiv_monthly.csv"

BASE_URL = "https://www.ncei.noaa.gov/pub/data/cirs/climdiv/"
DIVISIONS_SHP = BASE_URL + "CONUS_CLIMATE_DIVISIONS.shp.zip"

# nClimDiv uses its own state numbering, NOT FIPS. Arizona is 02 here (its FIPS is 04).
AZ_STATE_CODE = "02"

# The three elements we need. Others in the same series: phdi, zndx, sp01..sp24,
# tmax, tmin.
ELEMENTS = {
    "pdsi": "nclimdiv_pdsi",
    "tmpc": "nclimdiv_temperature_c",
    "pcpn": "nclimdiv_precipitation_mm_day",
}

MISSING = -99.99

# Divisions clipping the region boundary by less than this are boundary slivers
# (two AZ divisions touch it at ~3e-5 and ~6e-6 of the area) — drop them.
MIN_AREA_WEIGHT = 0.01

# Equal-area projection for CONUS. Computing area in EPSG:4326 gives wrong weights.
EQUAL_AREA_CRS = "EPSG:5070"

# Sanity floor for the MERRA-2 cross-check (see _validate_against_merra).
MIN_MERRA_CORR = 0.90


def _force_ipv4() -> None:
    """
    Pin urllib3 to IPv4 for this process.

    NCEI advertises an AAAA record whose endpoint resets the connection:

        curl -4 https://www.ncei.noaa.gov/pub/data/cirs/climdiv/   -> HTTP 200
        curl -6 https://www.ncei.noaa.gov/pub/data/cirs/climdiv/   -> connection reset

    On any host that prefers IPv6 (most Linux boxes do), `requests` therefore fails
    with a bare ConnectionError that looks like the server being down. It is not.
    Forcing the address family is the whole fix.
    """
    urllib3_connection.allowed_gai_family = lambda: socket.AF_INET


def _get(url: str, timeout: int = 300) -> requests.Response:
    """GET with the IPv4 pin applied and a clear error if it still fails."""
    _force_ipv4()
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


def _latest_filename(element: str) -> str:
    """
    Resolve the current dated filename for one element.

    The files carry a build date (`climdiv-pdsidv-v1.0.0-20260707`) that changes
    every month, so it cannot be hardcoded. Scrape the index for the newest one.
    """
    index = _get(BASE_URL, timeout=60)

    pattern = rf"climdiv-{element}dv-v\d+\.\d+\.\d+-\d{{8}}"
    matches = sorted(set(re.findall(pattern, index.text)))
    if not matches:
        raise ValueError(
            f"No file matching '{pattern}' at {BASE_URL}. "
            "The nClimDiv naming scheme may have changed."
        )
    return matches[-1]


def _parse_climdiv(text: str, value_name: str) -> pd.DataFrame:
    """
    Parse one fixed-width nClimDiv element file down to Arizona's divisions.

    Record layout, per the nClimDiv README:
        cols 0-1   state code (nClimDiv numbering)
        cols 2-3   climate division
        cols 4-5   element code
        cols 6-9   year
        then 12 right-justified values of width 7, Jan..Dec
    """
    rows = []
    for line in text.splitlines():
        if not line.startswith(AZ_STATE_CODE) or len(line) < 10:  # noqa: PLR2004
            continue
        division = int(line[2:4])
        year = int(line[6:10])
        body = line[10:]
        for month in range(1, 13):
            raw = body[(month - 1) * 7 : month * 7].strip()
            if not raw:
                continue
            value = float(raw)
            if value == MISSING:
                continue
            rows.append(
                {
                    "division": division,
                    "year": year,
                    "month": month,
                    value_name: value,
                }
            )

    df = pd.DataFrame(rows)
    log.info(
        "  %-28s %5d division-months, %d-%d",
        value_name,
        len(df),
        df["year"].min(),
        df["year"].max(),
    )
    return df


def _division_weights() -> pd.Series:
    """
    Area weight of each Arizona climate division inside the eight-county region.

    Computed by intersecting the NCEI division polygons with the study boundary in an
    equal-area projection. Returns a Series indexed by division number, summing to 1.
    """
    log.info("Downloading CONUS climate-division polygons...")
    resp = _get(DIVISIONS_SHP)

    divisions = gpd.read_file(io.BytesIO(resp.content))
    arizona = divisions[divisions["STATE"].str.upper() == "ARIZONA"].copy()
    if arizona.empty:
        raise ValueError("No Arizona polygons in CONUS_CLIMATE_DIVISIONS.")

    region = load_county_boundary()
    boundary = region.to_crs(EQUAL_AREA_CRS)
    boundary = (
        boundary.union_all() if hasattr(boundary, "union_all") else boundary.unary_union
    )

    arizona = arizona.to_crs(EQUAL_AREA_CRS)
    arizona["overlap"] = arizona.geometry.intersection(boundary).area

    hit = arizona[arizona["overlap"] > 0].copy()
    hit["weight"] = hit["overlap"] / hit["overlap"].sum()
    hit = hit[hit["weight"] >= MIN_AREA_WEIGHT]
    hit["weight"] = hit["weight"] / hit["weight"].sum()

    weights = pd.Series(
        hit["weight"].to_numpy(),
        index=hit["CD_NEW"].astype(int).to_numpy(),
        name="weight",
    ).sort_index()

    log.info("Division area weights within the eight-county region:")
    for div, name, w in zip(
        hit["CD_NEW"].astype(int), hit["NAME"], hit["weight"], strict=False
    ):
        log.info("  division %d  %-14s %.4f", div, name, w)

    return weights


def _weighted_regional_mean(
    df: pd.DataFrame, value_name: str, weights: pd.Series
) -> pd.DataFrame:
    """Collapse per-division monthly values into one area-weighted regional series."""
    df = df[df["division"].isin(weights.index)].copy()
    df["w"] = df["division"].map(weights)

    # Renormalize per month, so a month missing one division still weights correctly
    # rather than silently under-counting.
    grouped = df.groupby(["year", "month"])
    regional = grouped.apply(
        lambda g: (g[value_name] * g["w"]).sum() / g["w"].sum(),
        include_groups=False,
    )
    return regional.rename(value_name).reset_index()


def _to_celsius(f: pd.Series) -> pd.Series:
    return (f - 32.0) * 5.0 / 9.0


def _inches_per_month_to_mm_day(inches: pd.Series, frame: pd.DataFrame) -> pd.Series:
    days = frame.apply(
        lambda r: calendar.monthrange(int(r.year), int(r.month))[1], axis=1
    )
    return inches * 25.4 / days


def _validate_against_merra(df: pd.DataFrame) -> None:
    """
    Cross-check the overlap against MERRA-2 before writing.

    nClimDiv and MERRA-2 are independent measurements of the same physical
    quantities, so over 2000-2023 they must agree closely. If they do not, the likely
    cause is a unit bug (the raw files are Fahrenheit and inches) or a bad area
    weighting — exactly the class of error that silently poisons a model. Refuse to
    write rather than ship it.
    """
    checks = [
        ("temperature_monthly.csv", "temperature_2m_c", "nclimdiv_temperature_c"),
        (
            "precipitation_monthly.csv",
            "precipitation_mm_day",
            "nclimdiv_precipitation_mm_day",
        ),
    ]
    for filename, merra_col, ours in checks:
        path = OUTPUT_DIR / filename
        if not path.exists():
            log.warning("  %s not found — skipping cross-check for %s", filename, ours)
            continue

        merra = pd.read_csv(path)
        merged = df.merge(
            merra[["year_month", merra_col]], on="year_month", how="inner"
        )
        merged = merged.dropna(subset=[ours, merra_col])
        if len(merged) < 24:  # noqa: PLR2004
            log.warning("  too little overlap to cross-check %s", ours)
            continue

        corr = merged[ours].corr(merged[merra_col])
        bias = (merged[ours] - merged[merra_col]).mean()
        log.info(
            "  cross-check %-30s vs MERRA-2: r=%+.4f  mean bias=%+.3f  (n=%d)",
            ours,
            corr,
            bias,
            len(merged),
        )
        if corr < MIN_MERRA_CORR:
            raise ValueError(
                f"{ours} correlates only {corr:+.4f} with MERRA-2 "
                f"{merra_col} (floor {MIN_MERRA_CORR}). Suspect a unit conversion "
                "(raw files are Fahrenheit / inches-per-month) or the area weights. "
                "Refusing to write."
            )


def main() -> None:
    log.info("=== nclimdiv.py start ===")

    weights = _division_weights()

    frames = []
    log.info("Downloading nClimDiv element files...")
    for element, value_name in ELEMENTS.items():
        filename = _latest_filename(element)
        log.info("  %s -> %s", element, filename)
        resp = _get(BASE_URL + filename)

        per_division = _parse_climdiv(resp.text, value_name)
        frames.append(_weighted_regional_mean(per_division, value_name, weights))

    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on=["year", "month"], how="outer")

    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    # Convert from the raw US customary units.
    df["nclimdiv_temperature_c"] = _to_celsius(df["nclimdiv_temperature_c"])
    df["nclimdiv_precipitation_mm_day"] = _inches_per_month_to_mm_day(
        df["nclimdiv_precipitation_mm_day"], df
    )

    df["year_month"] = (
        df["year"].astype(int).astype(str)
        + "-"
        + df["month"].astype(int).astype(str).str.zfill(2)
    )

    out = df[["year_month", *ELEMENTS.values()]].copy()
    out = out.dropna(how="all", subset=list(ELEMENTS.values()))

    _validate_against_merra(out)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    log.info(
        "Wrote %d monthly rows (%s to %s) to %s",
        len(out),
        out["year_month"].iloc[0],
        out["year_month"].iloc[-1],
        OUTPUT_FILE,
    )
    log.info("=== nclimdiv.py complete ===")


if __name__ == "__main__":
    main()

"""
region_variants.py
------------------
Rebuild the region-dependent measurements under alternative county sets.

WHY THIS EXISTS
---------------
`scripts/phase1/region.py` declares COUNTIES by name and COUNTY_FIPS by code, and
`load_county_boundary()` filters on the CODE. Two of the codes do not match the names
beside them:

    "04013",  # Graham    -> 04013 is MARICOPA. Graham is 04009.
    "04007",  # La Paz    -> 04007 is GILA.     La Paz is 04012.

So every geospatial product in this project has been built over Pima, Pinal, Santa
Cruz, Cochise, **Maricopa**, Greenlee, Yuma and **Gila** — Phoenix included, Graham and
La Paz absent — while every document says otherwise. `region_acres` (27,779,840)
matches the set actually used, not the set named.

This script does not fix anything. It rebuilds the quantities that can be recomputed
from data already on disk, under three county sets, so the size of the problem can be
seen before anything is re-run:

    as_shipped   what the code actually selects today (includes Maricopa + Gila)
    no_maricopa  the same, minus Maricopa — isolates the Phoenix contamination
    documented   the eight counties the project says it studies

Everything here reads rasters and CSVs already present. Nothing is retrained, no
shipped artifact is touched, and `region.py` is left alone.

Run:
    python scripts/phase3/region_variants.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "phase3"))

import ndvi_endpoints as ne  # noqa: E402
from phase1.region import COUNTY_FIPS, STATE_FIPS  # noqa: E402

TIGER = ROOT / "data" / "raw" / "tiger" / "tl_2023_us_county.shp"
OUTPUT_FILE = ROOT / "model" / "region_variants.json"
SQ_M_PER_ACRE = 4046.8564224
EQUAL_AREA = "EPSG:5070"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("region_variants")

DOCUMENTED = ["Pima", "Pinal", "Santa Cruz", "Cochise", "Graham", "Greenlee", "Yuma", "La Paz"]


def county_table() -> gpd.GeoDataFrame:
    counties = gpd.read_file(TIGER)
    counties = counties[counties.STATEFP == STATE_FIPS].copy()
    counties["acres"] = counties.to_crs(EQUAL_AREA).geometry.area / SQ_M_PER_ACRE
    return counties


def variants(counties: gpd.GeoDataFrame) -> dict[str, list[str]]:
    by_name = dict(zip(counties.NAME, counties.GEOID, strict=False))
    shipped = list(COUNTY_FIPS)
    return {
        "as_shipped": shipped,
        "no_maricopa": [f for f in shipped if f != by_name["Maricopa"]],
        "documented": [by_name[n] for n in DOCUMENTED],
    }


def boundary_for(counties: gpd.GeoDataFrame, fips: list[str]) -> gpd.GeoDataFrame:
    sel = counties[counties.GEOID.isin(fips)].to_crs("EPSG:4326")
    return sel.dissolve().reset_index(drop=True)[["geometry"]]


def population_2020(counties: gpd.GeoDataFrame, fips: list[str]) -> float | None:
    """2020 county populations from the Census PEP vintage-2023 file."""
    cache = ROOT / "data" / "raw" / "co-est2023-alldata.csv"
    if not cache.exists():
        import urllib.request  # noqa: PLC0415

        url = ("https://www2.census.gov/programs-surveys/popest/datasets/"
               "2020-2023/counties/totals/co-est2023-alldata.csv")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=90) as response:
                cache.write_bytes(response.read())
        except Exception as err:  # noqa: BLE001
            log.warning("could not fetch Census county estimates: %s", err)
            return None
    frame = pd.read_csv(cache, encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    frame["GEOID"] = frame.STATE + frame.COUNTY
    column = "POPESTIMATE2020" if "POPESTIMATE2020" in frame.columns else "ESTIMATESBASE2020"
    return float(frame[frame.GEOID.isin(fips)][column].sum())


def raster_stats(boundary: gpd.GeoDataFrame, tag: str) -> dict:
    """Mean NDVI and mean impervious cover inside one boundary, plus the DiD slopes."""
    bounds, nx, ny, cutline = ne.region_grid(boundary, tag=tag)
    granules: dict[str, list[Path]] = {}
    for granule in sorted(ne.MODIS_DIR.glob("*.hdf")):
        granules.setdefault(ne.month_of(granule), []).append(granule)

    late = ne.window_mean_ndvi(granules, ne.LATE_YEARS, bounds, nx, ny, cutline)
    imperv = ne.impervious_for(2020, bounds, nx, ny)

    # impervious_for caches per YEAR, not per boundary, so it must be invalidated
    # between variants or the second one silently reuses the first one's grid.
    for stale in ne.CACHE.glob("imperv_*.npy"):
        stale.unlink()

    # impervious_for warps to the GRID with no cutline, so it covers the bounding
    # box rather than the county polygon — and removing an interior county barely
    # moves the bounding box. Mask it by the NDVI footprint, which IS cutline-clipped.
    # (The DiD fits were always correct: the NDVI NaNs do this masking for them.)
    inside = np.isfinite(late)
    imperv = np.where(inside, imperv, np.nan)

    did = ne.difference_in_differences(granules, bounds, nx, ny, cutline)
    for stale in ne.CACHE.glob("imperv_*.npy"):
        stale.unlink()
    irr = ne.irrigation_did(granules, bounds, nx, ny, cutline, boundary)
    for stale in ne.CACHE.glob("imperv_*.npy"):
        stale.unlink()

    cell_acres = ne.RES**2 / SQ_M_PER_ACRE
    return {
        "grid": [nx, ny],
        "cells_with_ndvi": int(np.isfinite(late).sum()),
        "raster_acres": float(np.isfinite(late).sum() * cell_acres),
        "mean_ndvi": float(np.nanmean(late)),
        "mean_impervious_pct": float(np.nanmean(imperv)),
        "ndvi_impervious_slope": did["slope"],
        "ndvi_impervious_t": did["t"],
        "ndvi_irrigated_slope": irr["slope_baseline_controlled"],
        "ndvi_irrigated_t": irr["t_baseline_controlled"],
    }


def irrigation_stats(boundary: gpd.GeoDataFrame) -> dict:
    """HUC12s and total irrigation withdrawal inside one boundary."""
    from phase1 import irrigation as irr  # noqa: PLC0415

    frame = irr._load_raw(irr.INPUT_FILE)
    all_cols = irr._identify_huc12_columns(frame)
    shapes = gpd.read_file(ROOT / "data" / "raw" / "wbd" / "WBDHU12.shp")
    shapes = shapes.to_crs(boundary.crs)
    inside = shapes[shapes.geometry.representative_point().within(boundary.geometry.iloc[0])]
    columns = [c for c in all_cols if c in set(inside.HUC12)]
    values = frame[columns].replace(list(irr.NODATA_SENTINELS), np.nan)
    return {
        "huc12_count": int(len(columns)),
        "irrigation_mgd": float(values.groupby(frame["Year"]).mean().mean(axis=0).sum()),
    }


def main() -> None:
    counties = county_table()
    sets = variants(counties)
    by_geoid = dict(zip(counties.GEOID, counties.NAME, strict=False))
    acres = dict(zip(counties.GEOID, counties.acres, strict=False))

    print("=" * 78)
    print("REGION VARIANTS — what COUNTY_FIPS actually selects, and what it should")
    print("=" * 78)
    for name, fips in sets.items():
        print(f"\n  {name}  ({len(fips)} counties)")
        print("    " + ", ".join(sorted(by_geoid[f] for f in fips)))

    results = {}
    for name, fips in sets.items():
        log.info("rebuilding %s ...", name)
        boundary = boundary_for(counties, fips)
        entry = {
            "counties": sorted(by_geoid[f] for f in fips),
            "fips": sorted(fips),
            "county_acres": float(sum(acres[f] for f in fips)),
            "population_2020": population_2020(counties, fips),
        }
        entry.update(irrigation_stats(boundary))
        entry.update(raster_stats(boundary, tag=name))
        results[name] = entry

    report(results)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWritten to {OUTPUT_FILE.relative_to(ROOT)}")


def report(results: dict) -> None:
    rows = [
        ("counties", "n", lambda e: len(e["counties"]), "{:.0f}"),
        ("county area", "acres", lambda e: e["county_acres"], "{:,.0f}"),
        ("population 2020", "people", lambda e: e["population_2020"], "{:,.0f}"),
        ("HUC12s with data", "n", lambda e: e["huc12_count"], "{:,.0f}"),
        ("irrigation", "MGD", lambda e: e["irrigation_mgd"], "{:,.0f}"),
        ("MODIS cells", "n", lambda e: e["cells_with_ndvi"], "{:,.0f}"),
        ("mean NDVI", "", lambda e: e["mean_ndvi"], "{:.4f}"),
        ("mean impervious", "%", lambda e: e["mean_impervious_pct"], "{:.3f}"),
        ("ndvi_impervious slope", "", lambda e: e["ndvi_impervious_slope"], "{:+.4f}"),
        ("  its t", "", lambda e: e["ndvi_impervious_t"], "{:+.1f}"),
        ("ndvi_irrigated slope", "", lambda e: e["ndvi_irrigated_slope"], "{:+.4f}"),
        ("  its t", "", lambda e: e["ndvi_irrigated_t"], "{:+.1f}"),
    ]
    names = list(results)
    print("\n" + "-" * 78)
    print(f"{'quantity':24s}{'unit':8s}" + "".join(f"{n:>15s}" for n in names))
    print("-" * 78)
    for label, unit, getter, fmt in rows:
        cells = []
        for n in names:
            try:
                value = getter(results[n])
                cells.append(fmt.format(value) if value is not None else "n/a")
            except (KeyError, TypeError):
                cells.append("n/a")
        print(f"{label:24s}{unit:8s}" + "".join(f"{c:>15s}" for c in cells))


if __name__ == "__main__":
    main()

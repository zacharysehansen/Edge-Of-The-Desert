"""
ndvi_endpoints.py
-----------------
Measure `ndvi_impervious` from the project's own MOD13A3 and NLCD rasters.

PHASE3_PARAMS.md §4b assumed two NDVI endpoints and recorded both as UNTESTED,
noting they were "derivable from data/raw/modis_ndvi/ but not measured — that needs
a pass over the HDFs, which was out of budget". This is that pass. The data was
already on disk: 574 MOD13A3 granules and 30 annual NLCD fractional-impervious
rasters.

WHAT IS ACTUALLY BEING MEASURED, AND WHY IT IS THE RIGHT QUANTITY
-----------------------------------------------------------------
The urbanization lever is

    ΔNDVI = (Δimpervious_points / 100) × (ndvi_impervious − ndvi_natural)

so the number the interface multiplies is not the endpoint on its own, it is the
DIFFERENCE — which is exactly the slope of NDVI on impervious fraction across
pixels. That is directly regressable, and the endpoint comes back out as
`intercept + slope` rather than being assumed. Measuring the slope also means the
regression's intercept is a free falsification test: at zero impervious cover it
must reproduce the region's own natural NDVI, and if it does not, the co-registration
is wrong.

METHOD
------
Both rasters are put on one grid — MODIS sinusoidal at the native 926.6 m, clipped
to the dissolved eight-county boundary from `scripts/phase1/region.py`, so the
boundary is the same one every Phase 1 script uses.

Each month is paired with ITS OWN year's NLCD, because impervious cover is the one
predictor here that genuinely trends: pairing every month against a single year
would put a 24-year land-cover change into the residual.

GDAL's `average` resampling includes nodata cells in the mean, which for this region
matters — NLCD's 250 fill begins at the Mexico border and Santa Cruz, Cochise and
Yuma counties all sit on it. Averaging it in produced "impervious" values up to
178.9%. Neither `srcNodata`, `UNIFIED_SRC_NODATA` nor `INIT_DEST` suppressed it. So
the aggregation is done exactly, with two LUT-remapped VRTs: one mapping fill to 0
to get Σvalue/n, one mapping valid to 1 and fill to 0 to get the valid fraction.
Only pixels that are ~entirely valid are kept, which drops a thin border strip
rather than silently biasing it upward.

UNCERTAINTY
-----------
The per-pixel OLS standard error is not reported and must not be: 130,000 MODIS
pixels in one month are nowhere near independent, and treating them as such would
manufacture a precision this has no claim to. The band is instead the spread of the
slope ACROSS months and years — each month is one quasi-independent realisation, and
the seasonal and secular spread is the honest uncertainty for a coefficient the
interface applies at every month of the year.

Run:
    python scripts/phase3/ndvi_endpoints.py            # all years
    python scripts/phase3/ndvi_endpoints.py --years 2005 2010 2015 2020
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from osgeo import gdal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import load_county_boundary  # noqa: E402

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("ndvi_endpoints")

MODIS_DIR = ROOT / "data" / "raw" / "modis_ndvi"
NLCD_ROOT = ROOT / "data" / "raw" / "NLCD"
OUTPUT_FILE = ROOT / "model" / "ndvi_endpoints.json"
CACHE = ROOT / "data" / "processed" / "ndvi_endpoints_cache"

# MODIS sinusoidal, and MOD13A3's native 1 km cell.
SINU = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
RES = 926.6254330558345

NDVI_SUBDATASET = '"1 km monthly NDVI"'
NDVI_SCALE = 10000.0
NDVI_INVALID_BELOW = -2000  # MOD13A3 fill is -3000; the product's own valid floor.
NLCD_FILL = 250
VALID_FRACTION_MIN = 0.999

# gdal.Warp writes -9999 for "no data here"; anything above this floor is real.
WARP_NODATA_FLOOR = -9998.0

# Above this, a MODIS cell is urban core rather than urban fringe. Reported as a
# separate fit because extrapolating a whole-region slope out to 100% impervious
# leans on cells that mostly do not exist; this one is anchored where they do.
CORE_IMPERVIOUS_MIN = 25.0

# The difference-in-differences windows. Five years each end, so weather averages
# out and what is left is the land-cover change.
EARLY_YEARS = range(2000, 2005)
LATE_YEARS = range(2019, 2024)
DID_EARLY_IMPERVIOUS_YEAR = 2001
DID_LATE_IMPERVIOUS_YEAR = 2021
MIN_MONTHS_PER_WINDOW = 12

# A cell counts as urbanised if it gained this many points of impervious cover.
URBANISED_MIN_POINTS = 10.0


def month_of(path: Path) -> str:
    """MOD13A3.AYYYYDDD... -> 'YYYY-MM'."""
    m = re.search(r"A(\d{4})(\d{3})", path.name)
    if not m:
        raise ValueError(f"No A-date in {path.name}")
    d = pd.Timestamp(year=int(m.group(1)), month=1, day=1) + pd.Timedelta(
        days=int(m.group(2)) - 1
    )
    return d.strftime("%Y-%m")


def region_grid() -> tuple[tuple[float, float, float, float], int, int, Path]:
    """The eight-county boundary in MODIS sinusoidal, plus the grid it implies."""
    CACHE.mkdir(parents=True, exist_ok=True)
    boundary = load_county_boundary().to_crs(SINU)
    cutline = CACHE / "region_sinu.gpkg"
    boundary.to_file(cutline, driver="GPKG")

    xmin, ymin, xmax, ymax = boundary.total_bounds
    # Snap outward to whole cells so every month lands on an identical grid.
    xmin = np.floor(xmin / RES) * RES
    ymin = np.floor(ymin / RES) * RES
    nx = int(np.ceil((xmax - xmin) / RES))
    ny = int(np.ceil((ymax - ymin) / RES))
    return (xmin, ymin, xmin + nx * RES, ymin + ny * RES), nx, ny, cutline


def nlcd_path(year: int) -> Path | None:
    for block in sorted(NLCD_ROOT.glob("Annual_NLCD_FctImp_*")):
        hit = block / f"Annual_NLCD_FctImp_{year}_CU_C1V1.tif"
        if hit.exists():
            return hit
    return None


def _lut_vrt(src: Path, dest: Path, lut: str) -> Path:
    """A VRT over `src` with a lookup table. Only 0-100 and 250 occur in the data,
    so the piecewise-linear LUT is exact on every value actually present."""
    ds = gdal.Translate("", str(src), format="VRT")
    xml = ds.GetMetadata("xml:VRT")[0]
    ds = None
    xml = xml.replace("<SimpleSource>", "<ComplexSource>").replace(
        "</SimpleSource>", f"<LUT>{lut}</LUT></ComplexSource>"
    )
    xml = re.sub(r"<ColorTable>.*?</ColorTable>", "", xml, flags=re.S)
    xml = xml.replace('relativeToVRT="1"', 'relativeToVRT="0"')
    dest.write_text(xml)
    return dest


def _warp(src: str | Path, out: Path, bounds, nx, ny, alg, nodata) -> np.ndarray:
    # Flush and drop the handle before reading the file back. Skipping either step
    # fails at runtime, and so does chaining gdal.Open(..).GetRasterBand(1), which lets
    # the dataset be collected out from under the band.
    ds = gdal.Warp(
        str(out), str(src), dstSRS=SINU, outputBounds=bounds, width=nx, height=ny,
        resampleAlg=alg, outputType=gdal.GDT_Float32, dstNodata=nodata,
    )
    ds.FlushCache()
    ds = None
    d = gdal.Open(str(out))
    arr = d.GetRasterBand(1).ReadAsArray()
    d = None
    return arr


def impervious_for(year: int, bounds, nx, ny) -> np.ndarray | None:
    """Per-MODIS-cell mean impervious percent, exactly, or NaN where contaminated."""
    cached = CACHE / f"imperv_{year}.npy"
    if cached.exists():
        return np.load(cached)
    src = nlcd_path(year)
    if src is None:
        return None
    value = _warp(
        _lut_vrt(src, CACHE / "nlcd_val.vrt", f"0:0,100:100,{NLCD_FILL}:0"),
        CACHE / "_val.tif", bounds, nx, ny, "average", -9999,
    )
    frac = _warp(
        _lut_vrt(src, CACHE / "nlcd_mask.vrt", f"0:1,100:1,{NLCD_FILL}:0"),
        CACHE / "_frac.tif", bounds, nx, ny, "average", -9999,
    )
    good = (value > WARP_NODATA_FLOOR) & (frac > VALID_FRACTION_MIN)
    out = np.where(good, value, np.nan).astype("float32")
    np.save(cached, out)
    log.info("  NLCD %d: %d usable cells (%d dropped as part-fill)",
             year, int(good.sum()), int(((frac > WARP_NODATA_FLOOR) & ~good).sum()))
    return out


def ndvi_for(granules: list[Path], bounds, nx, ny, cutline: Path) -> np.ndarray:
    sds = [
        f'HDF4_EOS:EOS_GRID:"{g}":MOD_Grid_monthly_1km_VI:{NDVI_SUBDATASET}'
        for g in granules
    ]
    ds = gdal.Warp(
        str(CACHE / "_ndvi.tif"), sds, dstSRS=SINU, outputBounds=bounds,
        width=nx, height=ny, resampleAlg="near", cutlineDSName=str(cutline),
        dstNodata=-3000, outputType=gdal.GDT_Int16,
    )
    ds.FlushCache()
    ds = None
    d = gdal.Open(str(CACHE / "_ndvi.tif"))
    arr = d.GetRasterBand(1).ReadAsArray().astype("float64")
    d = None
    return np.where(arr > NDVI_INVALID_BELOW, arr / NDVI_SCALE, np.nan)


def fit(ndvi: np.ndarray, imperv: np.ndarray, floor: float = 0.0) -> dict | None:
    """OLS of NDVI on impervious FRACTION. Slope is the lever's own coefficient."""
    m = np.isfinite(ndvi) & np.isfinite(imperv) & (imperv >= floor)
    if m.sum() < 500:  # noqa: PLR2004
        return None
    y = ndvi[m]
    x = imperv[m] / 100.0
    design = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(m.sum()),
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "endpoint": float(coef[0] + coef[1]),
        "r2": float(1 - ((y - pred) ** 2).sum() / ss_tot) if ss_tot else float("nan"),
        "mean_ndvi": float(y.mean()),
        "mean_impervious_pct": float(x.mean() * 100),
    }


def window_mean_ndvi(granules, years, bounds, nx, ny, cutline) -> np.ndarray:
    """Per-cell mean NDVI across every month of `years`."""
    total = np.zeros((ny, nx))
    count = np.zeros((ny, nx))
    for ym in sorted(granules):
        if int(ym[:4]) not in years:
            continue
        arr = ndvi_for(granules[ym], bounds, nx, ny, cutline)
        seen = np.isfinite(arr)
        total[seen] += arr[seen]
        count[seen] += 1
    return np.where(count >= MIN_MONTHS_PER_WINDOW, total / np.maximum(count, 1), np.nan)


def _ols_robust(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Slope and HC1 standard error."""
    design = np.vstack([np.ones_like(y), x]).T
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    xtx_inv = np.linalg.inv(design.T @ design)
    meat = (design * (resid**2)[:, None]).T @ design
    se = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv)) * np.sqrt(len(y) / (len(y) - 2))
    return float(coef[1]), float(se[1])


def difference_in_differences(granules, bounds, nx, ny, cutline) -> dict:
    """The better-identified estimate: difference each cell against ITSELF.

    The cross-sectional fit compares Tucson to the desert around it, which cannot
    separate "this land is paved" from "this land was always different" — cities sit
    on valley floors, alluvial fans and washes, which were never a random sample of
    the region. Differencing a cell against its own past removes every time-invariant
    characteristic at once: soil, elevation, aspect, drainage, and whatever the
    baseline vegetation was.

    Regional drought and any secular greening are common to every cell, so they land
    in the intercept rather than the slope.
    """
    early = window_mean_ndvi(granules, EARLY_YEARS, bounds, nx, ny, cutline)
    late = window_mean_ndvi(granules, LATE_YEARS, bounds, nx, ny, cutline)
    imp_early = impervious_for(DID_EARLY_IMPERVIOUS_YEAR, bounds, nx, ny)
    imp_late = impervious_for(DID_LATE_IMPERVIOUS_YEAR, bounds, nx, ny)

    d_ndvi = (late - early).ravel()
    d_imp = ((imp_late - imp_early) / 100.0).ravel()
    base = early.ravel()
    keep = np.isfinite(d_ndvi) & np.isfinite(d_imp) & np.isfinite(base)
    d_ndvi, d_imp, base = d_ndvi[keep], d_imp[keep], base[keep]

    slope, se = _ols_robust(d_ndvi, d_imp)

    # What the cell WAS before it was paved turns out to be the whole story, so it is
    # reported rather than averaged away. Strata are quantiles of the 2000-04 mean.
    cuts = np.nanquantile(base, [0.5, 0.8, 0.95])
    strata = {}
    for label, sel in [
        ("dry_desert", base < cuts[0]),
        ("typical", (base >= cuts[0]) & (base < cuts[1])),
        ("green", (base >= cuts[1]) & (base < cuts[2])),
        ("very_green_cropland_riparian", base >= cuts[2]),
    ]:
        if sel.sum() < 500:  # noqa: PLR2004
            continue
        s, e = _ols_robust(d_ndvi[sel], d_imp[sel])
        strata[label] = {
            "n": int(sel.sum()),
            "n_urbanised": int((d_imp[sel] > URBANISED_MIN_POINTS / 100).sum()),
            "slope": s, "se": e, "t": s / e if e else float("nan"),
            "baseline_ndvi_mean": float(base[sel].mean()),
        }

    urbanised = d_imp > URBANISED_MIN_POINTS / 100
    control = d_imp < 0.005  # noqa: PLR2004
    return {
        "early_years": [min(EARLY_YEARS), max(EARLY_YEARS)],
        "late_years": [min(LATE_YEARS), max(LATE_YEARS)],
        "n_cells": int(keep.sum()),
        "slope": slope, "se": se, "t": slope / se if se else float("nan"),
        "regional_drift_intercept": float(d_ndvi[control].mean()),
        "urbanised": {
            "n": int(urbanised.sum()),
            "mean_d_ndvi": float(d_ndvi[urbanised].mean()),
            "mean_d_impervious_pts": float(d_imp[urbanised].mean() * 100),
        },
        "control": {"n": int(control.sum()), "mean_d_ndvi": float(d_ndvi[control].mean())},
        "baseline_quantiles": [float(c) for c in cuts],
        "by_prior_land_cover": strata,
    }

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=None)
    args = ap.parse_args()

    bounds, nx, ny, cutline = region_grid()
    log.info("grid %d x %d at %.2f m, sinusoidal", nx, ny, RES)

    granules: dict[str, list[Path]] = {}
    for g in sorted(MODIS_DIR.glob("*.hdf")):
        granules.setdefault(month_of(g), []).append(g)
    months = sorted(granules)
    if args.years:
        months = [m for m in months if int(m[:4]) in args.years]
    log.info("%d months of MOD13A3", len(months))

    rows = []
    for i, ym in enumerate(months, 1):
        year = int(ym[:4])
        imperv = impervious_for(year, bounds, nx, ny)
        if imperv is None:
            log.warning("no NLCD for %d, skipping %s", year, ym)
            continue
        ndvi = ndvi_for(granules[ym], bounds, nx, ny, cutline)
        whole = fit(ndvi, imperv)
        core = fit(ndvi, imperv, floor=CORE_IMPERVIOUS_MIN)
        if whole is None:
            continue
        rows.append({"month": ym, "year": year, "month_of_year": int(ym[5:]),
                     **whole, "core_slope": core["slope"] if core else None,
                     "core_endpoint": core["endpoint"] if core else None})
        if i % 24 == 0 or i == len(months):
            log.info("  %s  (%d/%d)  slope=%+.4f", ym, i, len(months), whole["slope"])

    frame = pd.DataFrame(rows)
    log.info("difference-in-differences ...")
    did = difference_in_differences(granules, bounds, nx, ny, cutline)
    report(frame, did)


def report(frame: pd.DataFrame, did: dict) -> None:
    natural = json.loads((ROOT / "frontend" / "computed_stats.json").read_text())
    ndvi_natural = natural["OUTPUT_STATS"]["ndvi"]["baseline"]

    slope = frame["slope"]
    endpoint = frame["endpoint"]
    print("\n" + "=" * 78)
    print("NDVI ENDPOINTS  (PHASE3_PARAMS.md §4b)")
    print("=" * 78)
    print(f"\n  months fitted            {len(frame)}  ({frame['month'].iloc[0]}..{frame['month'].iloc[-1]})")
    print(f"  cells per month          {frame['n'].median():,.0f} (median)")
    print(f"  ndvi_natural (in repo)   {ndvi_natural:.4f}")
    print(f"  regression intercept     {frame['intercept'].median():.4f}  <- must reproduce it")

    print("\n  slope = ndvi_impervious - ndvi_natural, across months")
    print(f"    median {slope.median():+.4f}   IQR {slope.quantile(.25):+.4f}..{slope.quantile(.75):+.4f}"
          f"   p5..p95 {slope.quantile(.05):+.4f}..{slope.quantile(.95):+.4f}")
    print(f"    sign stable and negative in {int((slope < 0).sum())}/{len(slope)} months")
    print("\n  ndvi_impervious = intercept + slope")
    print(f"    median {endpoint.median():.4f}   IQR {endpoint.quantile(.25):.4f}..{endpoint.quantile(.75):.4f}")
    print("    ASSUMED in PHASE3_PARAMS §4b: 0.08  (band 0.05-0.12)")

    print("\n  by month of year (is the coefficient seasonal?)")
    by_m = frame.groupby("month_of_year")[["slope", "endpoint", "mean_ndvi"]].median()
    for m, r in by_m.iterrows():
        print(f"    {m:>2d}  slope {r['slope']:+.4f}   endpoint {r['endpoint']:.4f}   mean NDVI {r['mean_ndvi']:.4f}")

    core = frame["core_slope"].dropna()
    if len(core):
        print(f"\n  urban-core-only fit (impervious >= {CORE_IMPERVIOUS_MIN:.0f}%), which is where the")
        print("  extrapolation to 100% is actually anchored:")
        print(f"    slope median {core.median():+.4f}   endpoint median {frame['core_endpoint'].median():.4f}")

    print("\n  DIFFERENCE-IN-DIFFERENCES (the adopted estimate)")
    print(f"    each cell against itself, {did['early_years'][0]}-{did['early_years'][1]}"
          f" vs {did['late_years'][0]}-{did['late_years'][1]}, n={did['n_cells']:,}")
    print(f"    slope {did['slope']:+.4f}  (HC1 SE {did['se']:.4f}, t={did['t']:+.1f})")
    print(f"    cross-sectional slope, for comparison: {frame['slope'].median():+.4f}")
    print(f"    regional drift in unurbanised cells: {did['control']['mean_d_ndvi']:+.5f}")
    print("\n    by what the cell WAS before it was paved:")
    for label, s in did["by_prior_land_cover"].items():
        print(f"      {label:30s} base NDVI {s['baseline_ndvi_mean']:.3f}  "
              f"n={s['n']:>6,} urbanising={s['n_urbanised']:>5,}  "
              f"slope {s['slope']:+.4f} (t={s['t']:+.1f})")

    payload = {
        "source": "PHASE3_PARAMS.md §4b; measured by scripts/phase3/ndvi_endpoints.py",
        "difference_in_differences": did,
        "grid": "MODIS sinusoidal 926.63 m, eight-county cutline",
        "n_months": int(len(frame)),
        "span": [frame["month"].iloc[0], frame["month"].iloc[-1]],
        "ndvi_natural_in_repo": ndvi_natural,
        "intercept_median": float(frame["intercept"].median()),
        "slope": {
            "median": float(slope.median()),
            "iqr": [float(slope.quantile(.25)), float(slope.quantile(.75))],
            "p5_p95": [float(slope.quantile(.05)), float(slope.quantile(.95))],
            "negative_months": int((slope < 0).sum()),
        },
        "ndvi_impervious": {
            "median": float(endpoint.median()),
            "iqr": [float(endpoint.quantile(.25)), float(endpoint.quantile(.75))],
            "assumed_before": 0.08,
            "assumed_band_before": [0.05, 0.12],
        },
        "urban_core_fit": {
            "impervious_floor_pct": CORE_IMPERVIOUS_MIN,
            "slope_median": float(core.median()) if len(core) else None,
            "endpoint_median": float(frame["core_endpoint"].median()) if len(core) else None,
        },
        "by_month_of_year": {int(m): float(v) for m, v in by_m["slope"].items()},
        "monthly": frame.to_dict(orient="records"),
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  wrote {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

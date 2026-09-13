"""
gldas.py
--------
Monthly GLDAS-2.1 Noah land-surface state over the study bounding box, as
FEATURES for the GRACE model (PROBLEMS.md P5 Option C; PHASE3_PLAN.md §16 item 5).

WHY THIS EXISTS
---------------
GRACE measures total water storage. Its monthly change is a fast part — soil
moisture and snow responding to recent weather — plus a slow part — groundwater
responding to pumping and recharge. The panel's precipitation lags approximate the
fast part; GLDAS gives an observationally-forced land-surface model's estimate of it
directly. This is a feature script, not a target: `grace_groundwater.py` builds the
target and is untouched.

Input  : GLDAS_NOAH025_M v2.1 (GES DISC), one global 0.25° netCDF-4 granule per
         month, 2000-01 onward, ~24 MB each. Only the study bounding box is needed,
         so each granule is fetched as a DAP4 subset (~40 KB) from Earthdata's
         cloud OPeNDAP service; the full file is downloaded and subset locally only
         if the subset service refuses. Subsets are cached in data/raw/gldas/
         (gitignored).
Auth   : an Earthdata Login bearer token, read from $EARTHDATA_TOKEN or
         ~/.config/earthdata/token. The GES DISC application must be approved on the
         account (https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ);
         without that every request returns 403 "EULA Acceptance Failure".

Output : data/Final/gldas_monthly.csv
    year_month                 str   YYYY-MM
    gldas_surface_soil_mm      float 0-10 cm soil moisture, kg m-2 (= mm), bbox mean
    gldas_root_zone_mm         float 0-100 cm soil moisture, mm
    gldas_soil_moisture_mm     float 0-200 cm soil moisture (all four layers), mm
    gldas_swe_mm               float snow water equivalent, mm
    gldas_canopy_mm            float canopy interception storage, mm
    gldas_tws_proxy_mm         float soil (0-200 cm) + SWE + canopy, mm — the part of
                                     total water storage the land-surface model sees
    gldas_evap_mm_day          float evapotranspiration, mm/day
    gldas_rain_mm_day          float total precipitation forcing, mm/day (a check
                                     against the MERRA-2 series, not a new input)
    gldas_n_cells              int   land cells averaged

Spatial note. The mean is over the same bounding box the raster inputs use
(region.BBOX), not the county cutline — GLDAS at 0.25° has ~290 cells in the box,
and the county polygon would clip a good fraction of them in half. The GRACE target
is a bbox mean too, so the two are on the same footprint.

Run:
    python -m scripts.phase1.gldas
"""

from __future__ import annotations

import logging
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import BBOX  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw" / "gldas"
OUTPUT_FILE = ROOT / "data" / "Final" / "gldas_monthly.csv"
TOKEN_FILE = Path.home() / ".config" / "earthdata" / "token"

SHORT_NAME = "GLDAS_NOAH025_M"
VERSION = "2.1"
COLLECTION_CONCEPT_ID = "C1342986036-GES_DISC"
START, END = "2000-01", "2023-12"

OPENDAP = "https://opendap.earthdata.nasa.gov/collections/{cid}/granules/{gid}.dap.nc4?dap4.ce={ce}"
DATA_URL = "https://data.gesdisc.earthdata.nasa.gov/data/GLDAS/GLDAS_NOAH025_M.2.1/{year}/{gid}"

VARIABLES = [
    "SoilMoi0_10cm_inst",
    "SoilMoi10_40cm_inst",
    "SoilMoi40_100cm_inst",
    "SoilMoi100_200cm_inst",
    "SWE_inst",
    "CanopInt_inst",
    "Evap_tavg",
    "Rainf_f_tavg",
]

# GLDAS 0.25° grid: lat -59.875..89.875 (600), lon -179.875..179.875 (1440).
LAT0, LON0, RES = -59.875, -179.875, 0.25
SECONDS_PER_DAY = 86400.0


def _token() -> str:
    tok = os.environ.get("EARTHDATA_TOKEN") or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
    if not tok:
        raise RuntimeError(
            "No Earthdata token. Set $EARTHDATA_TOKEN or write it to ~/.config/earthdata/token "
            "(mode 600). Tokens: https://urs.earthdata.nasa.gov/profile → Generate Token."
        )
    return tok


def _index_window() -> tuple[int, int, int, int]:
    """Grid index ranges that cover the bbox, one cell of margin each side."""
    min_lon, min_lat, max_lon, max_lat = BBOX
    i0 = int(np.floor((min_lat - LAT0) / RES)) - 1
    i1 = int(np.ceil((max_lat - LAT0) / RES)) + 1
    j0 = int(np.floor((min_lon - LON0) / RES)) - 1
    j1 = int(np.ceil((max_lon - LON0) / RES)) + 1
    return i0, i1, j0, j1


def _granule_id(year: int, month: int) -> str:
    return f"{SHORT_NAME}.A{year}{month:02d}.021.nc4"


def _subset_url(year: int, month: int) -> str:
    i0, i1, j0, j1 = _index_window()
    gid = f"{SHORT_NAME}.{VERSION}%3A{_granule_id(year, month)}"
    ce = ";".join(f"/{v}[0][{i0}:{i1}][{j0}:{j1}]" for v in VARIABLES)
    ce += f";/lat[{i0}:{i1}];/lon[{j0}:{j1}];/time[0]"
    return OPENDAP.format(cid=COLLECTION_CONCEPT_ID, gid=gid, ce=ce)


def _fetch(session: requests.Session, year: int, month: int) -> Path:
    """Return the cached subset for one month, fetching it if needed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached = RAW_DIR / f"{SHORT_NAME}.A{year}{month:02d}.021.subset.nc4"
    if cached.exists() and cached.stat().st_size > 1000:  # noqa: PLR2004
        return cached
    url = _subset_url(year, month)
    for attempt in range(1, 6):
        try:
            r = session.get(url, timeout=180)
        except requests.exceptions.ConnectionError as e:
            # This sandbox's DNS for *.earthdata.nasa.gov drops out intermittently.
            log.warning("%d-%02d attempt %d: %s", year, month, attempt, type(e).__name__)
            time.sleep(5 * attempt)
            continue
        if r.status_code == 200 and r.content[:8] == b"\x89HDF\r\n\x1a\n":  # noqa: PLR2004
            cached.write_bytes(r.content)
            return cached
        if r.status_code == 403 and b"EULA" in r.content:  # noqa: PLR2004
            raise RuntimeError(
                "403 EULA Acceptance Failure: approve the GES DISC application on this "
                "Earthdata account, then re-run: "
                "https://urs.earthdata.nasa.gov/approve_app?client_id=e2WVk8Pw6weeLUKZYOxvTQ"
            )
        log.warning("%d-%02d subset attempt %d -> HTTP %d (%d bytes)", year, month, attempt, r.status_code, len(r.content))
        time.sleep(3 * attempt)
    # Fallback: whole granule, subset locally, keep only the subset.
    log.warning("%d-%02d: subset service failed, downloading the full granule", year, month)
    r = session.get(DATA_URL.format(year=year, gid=_granule_id(year, month)), timeout=600)
    r.raise_for_status()
    import xarray as xr  # noqa: PLC0415

    i0, i1, j0, j1 = _index_window()
    with xr.open_dataset(BytesIO(r.content), engine="netcdf4") as ds:
        sub = ds[VARIABLES].isel(lat=slice(i0, i1 + 1), lon=slice(j0, j1 + 1)).load()
    sub.to_netcdf(cached, engine="netcdf4")
    return cached


def _summarise(path: Path) -> dict:
    import xarray as xr  # noqa: PLC0415

    min_lon, min_lat, max_lon, max_lat = BBOX
    with xr.open_dataset(path, engine="netcdf4") as ds:
        ds = ds.isel(time=0) if "time" in ds.dims else ds
        inside = (ds.lat >= min_lat) & (ds.lat <= max_lat)
        inside = inside & (ds.lon >= min_lon) & (ds.lon <= max_lon)
        box = ds.where(inside)

        def mean(var: str) -> float:
            return float(box[var].mean(skipna=True))

        n_cells = int(np.isfinite(box["SoilMoi0_10cm_inst"].values).sum())
        s10, s40, s100, s200 = (mean(v) for v in VARIABLES[:4])
        swe, canopy = mean("SWE_inst"), mean("CanopInt_inst")
        soil = s10 + s40 + s100 + s200
        return {
            "gldas_surface_soil_mm": s10,
            "gldas_root_zone_mm": s10 + s40 + s100,
            "gldas_soil_moisture_mm": soil,
            "gldas_swe_mm": swe,
            "gldas_canopy_mm": canopy,
            "gldas_tws_proxy_mm": soil + swe + canopy,
            "gldas_evap_mm_day": mean("Evap_tavg") * SECONDS_PER_DAY,
            "gldas_rain_mm_day": mean("Rainf_f_tavg") * SECONDS_PER_DAY,
            "gldas_n_cells": n_cells,
        }


def _guards(df: pd.DataFrame) -> None:
    expected = len(pd.period_range(START, END, freq="M"))
    if len(df) != expected or df.isna().any().any():
        raise RuntimeError(f"expected {expected} complete months, got {len(df)} with NaN={int(df.isna().sum().sum())}")
    if df.gldas_n_cells.nunique() != 1:
        raise RuntimeError(f"cell count varies month to month: {sorted(df.gldas_n_cells.unique())}")
    sm = df.gldas_soil_moisture_mm
    if not (50 < sm.min() and sm.max() < 800):  # noqa: PLR2004
        raise RuntimeError(f"0-200 cm soil moisture out of range: {sm.min():.0f}..{sm.max():.0f} mm")
    by_month = df.assign(m=df.year_month.str[5:].astype(int)).groupby("m")
    swe_peak = int(by_month.gldas_swe_mm.mean().idxmax())
    if swe_peak not in (1, 2, 3, 12):
        raise RuntimeError(f"SWE peaks in month {swe_peak}; expected winter")
    log.info(
        "guards passed: %d months, %d cells, soil %0.f..%0.f mm, SWE peak month %d, rain peak month %d",
        len(df), int(df.gldas_n_cells.iloc[0]), sm.min(), sm.max(), swe_peak,
        int(by_month.gldas_rain_mm_day.mean().idxmax()),
    )


def main() -> None:
    log.info("=== gldas.py start ===")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {_token()}", "User-Agent": "eotd-gldas"})
    rows = []
    months = pd.period_range(START, END, freq="M")
    for i, p in enumerate(months, 1):
        path = _fetch(session, p.year, p.month)
        rows.append({"year_month": str(p), **_summarise(path)})
        if i % 24 == 0 or i == len(months):
            log.info("  %s  (%d/%d)  soil %.1f mm  swe %.2f mm", p, i, len(months), rows[-1]["gldas_soil_moisture_mm"], rows[-1]["gldas_swe_mm"])
    df = pd.DataFrame(rows)
    _guards(df)
    # The MERRA-2 precipitation series is the model's rain input; GLDAS's forcing
    # should agree with it closely, or the box/units are wrong.
    merra = ROOT / "data" / "Final" / "precipitation_monthly.csv"
    if merra.exists():
        m = pd.read_csv(merra)
        j = df.merge(m, on="year_month")
        col = [c for c in m.columns if "precip" in c][0]
        log.info("rain vs MERRA-2 %s: r = %.3f over %d months", col, np.corrcoef(j.gldas_rain_mm_day, j[col])[0, 1], len(j))
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    log.info("wrote %s (%d rows)", OUTPUT_FILE, len(df))


if __name__ == "__main__":
    main()

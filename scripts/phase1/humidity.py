"""
humidity.py
-----------
Monthly 2-metre humidity and vapour pressure deficit (VPD) over the study bounding
box, from MERRA-2 M2TMNXSLV — for the wildfire and NDVI probes (PHASE3_PLAN.md §33).

WHY THIS EXISTS
---------------
The models see heat and rain but not the dryness of the air. VPD — saturation
vapour pressure at the air temperature minus the actual vapour pressure — is the
single strongest fire-weather variable in the western US and a direct measure of
plant water stress. `data/raw/merra_specific_humidity_2m/` was thought to hold
humidity; it holds a second copy of the temperature statistics files (T2MMAX,
T2MMEAN, T2MMIN, no QV2M). The monthly single-level collection M2TMNXSLV does carry
2-m specific humidity (QV2M), 2-m dew point (T2MDEW), 2-m air temperature (T2M) and
surface pressure (PS), so VPD is computed exactly from T2M and T2MDEW rather than
approximated.

Input  : M2TMNXSLV v5.12.4 (GES DISC), one global 0.5° x 0.625° granule per month,
         ~52 MB each; fetched as DAP4 bounding-box subsets (~20 KB) from Earthdata's
         cloud OPeNDAP service, cached in data/raw/merra_humidity/ (gitignored).
Auth   : Earthdata bearer token, $EARTHDATA_TOKEN or ~/.config/earthdata/token, with
         the GES DISC application approved — the same as gldas.py.

Output : data/Final/humidity_monthly.csv
    year_month              str
    specific_humidity_2m    float kg/kg, bbox mean
    dewpoint_2m_c           float °C
    air_temperature_2m_c    float °C (M2TMNXSLV's T2M; the model's temperature input is
                                  M2SMNXSLV's T2MMEAN — the two agree to ~0.1 °C)
    surface_pressure_kpa    float
    vpd_kpa                 float es(T2M) − es(T2MDEW), Tetens/FAO-56 form, kPa
    vpd_proxy_kpa           float the FAO-56 fallback when humidity is missing —
                                  (es(Tmax)+es(Tmin))/2 − es(Tmin) from the temperature
                                  files already on disk — kept beside the real one as
                                  its validation

Run:
    python -m scripts.phase1.humidity
"""

from __future__ import annotations

import glob
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from phase1.region import BBOX  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

RAW_DIR = ROOT / "data" / "raw" / "merra_humidity"
TEMP_RAW_DIR = ROOT / "data" / "raw" / "merra_temperature_2m"
OUTPUT_FILE = ROOT / "data" / "Final" / "humidity_monthly.csv"
TOKEN_FILE = Path.home() / ".config" / "earthdata" / "token"

SHORT_NAME, VERSION = "M2TMNXSLV", "5.12.4"
COLLECTION_CONCEPT_ID = "C1276812859-GES_DISC"
START, END = "2000-01", "2023-12"
OPENDAP = "https://opendap.earthdata.nasa.gov/collections/{cid}/granules/{gid}.dap.nc4?dap4.ce={ce}"
VARIABLES = ["QV2M", "T2MDEW", "T2M", "PS"]

# MERRA-2 grid: lat -90..90 step 0.5 (361), lon -180..179.375 step 0.625 (576).
LAT0, DLAT, LON0, DLON = -90.0, 0.5, -180.0, 0.625


def _token() -> str:
    tok = os.environ.get("EARTHDATA_TOKEN") or (TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else "")
    if not tok:
        raise RuntimeError("No Earthdata token: set $EARTHDATA_TOKEN or write ~/.config/earthdata/token")
    return tok


def _index_window() -> tuple[int, int, int, int]:
    min_lon, min_lat, max_lon, max_lat = BBOX
    i0 = int(np.floor((min_lat - LAT0) / DLAT)) - 1
    i1 = int(np.ceil((max_lat - LAT0) / DLAT)) + 1
    j0 = int(np.floor((min_lon - LON0) / DLON)) - 1
    j1 = int(np.ceil((max_lon - LON0) / DLON)) + 1
    return i0, i1, j0, j1


def granule_names() -> dict[str, str]:
    """year_month -> granule name, from CMR. The stream number (200/300/400/401)
    changes with the year, so the names cannot be assumed."""
    import earthaccess  # noqa: PLC0415

    out: dict[str, str] = {}
    res = earthaccess.search_data(short_name=SHORT_NAME, version=VERSION, temporal=(f"{START}-01", f"{END}-31"))
    for g in res:
        name = g["umm"]["GranuleUR"].split(":")[-1]
        m = re.search(r"\.(\d{4})(\d{2})\.nc4$", name)
        if m:
            out[f"{m.group(1)}-{m.group(2)}"] = name
    return out


def _fetch(session: requests.Session, ym: str, gname: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cached = RAW_DIR / f"{gname}.subset.nc4"
    if cached.exists() and cached.stat().st_size > 500:  # noqa: PLR2004
        return cached
    i0, i1, j0, j1 = _index_window()
    ce = ";".join(f"/{v}[0][{i0}:{i1}][{j0}:{j1}]" for v in VARIABLES) + f";/lat[{i0}:{i1}];/lon[{j0}:{j1}];/time[0]"
    url = OPENDAP.format(cid=COLLECTION_CONCEPT_ID, gid=f"{SHORT_NAME}.{VERSION}%3A{gname}", ce=ce)
    for attempt in range(1, 6):
        try:
            r = session.get(url, timeout=180)
        except requests.exceptions.ConnectionError as e:
            log.warning("%s attempt %d: %s", ym, attempt, type(e).__name__)
            time.sleep(5 * attempt)
            continue
        if r.status_code == 200 and r.content[:8] == b"\x89HDF\r\n\x1a\n":  # noqa: PLR2004
            cached.write_bytes(r.content)
            return cached
        if r.status_code == 403 and b"EULA" in r.content:  # noqa: PLR2004
            raise RuntimeError("403 EULA Acceptance Failure — approve the GES DISC application on this Earthdata account")
        log.warning("%s attempt %d -> HTTP %d", ym, attempt, r.status_code)
        time.sleep(3 * attempt)
    raise RuntimeError(f"could not fetch {gname}")


def es_kpa(t_c: float) -> float:
    """Saturation vapour pressure, Tetens (FAO-56 eq. 11)."""
    return 0.6108 * float(np.exp(17.27 * t_c / (t_c + 237.3)))


def _summarise(path: Path) -> dict:
    import xarray as xr  # noqa: PLC0415

    min_lon, min_lat, max_lon, max_lat = BBOX
    with xr.open_dataset(path, engine="netcdf4") as ds:
        ds = ds.isel(time=0) if "time" in ds.dims else ds
        inside = (ds.lat >= min_lat) & (ds.lat <= max_lat) & (ds.lon >= min_lon) & (ds.lon <= max_lon)
        box = ds.where(inside)
        q = float(box["QV2M"].mean(skipna=True))
        tdew = float(box["T2MDEW"].mean(skipna=True)) - 273.15
        t2m = float(box["T2M"].mean(skipna=True)) - 273.15
        ps = float(box["PS"].mean(skipna=True)) / 1000.0
    return {
        "specific_humidity_2m": q,
        "dewpoint_2m_c": tdew,
        "air_temperature_2m_c": t2m,
        "surface_pressure_kpa": ps,
        "vpd_kpa": es_kpa(t2m) - es_kpa(tdew),
    }


def vpd_proxy() -> pd.Series:
    """FAO-56 fallback from the temperature statistics files on disk."""
    import xarray as xr  # noqa: PLC0415

    rows = {}
    for f in sorted(glob.glob(str(TEMP_RAW_DIR / "*.nc4"))):
        m = re.search(r"\.(\d{4})(\d{2})\.nc4$", f)
        if not m:
            continue
        with xr.open_dataset(f, engine="netcdf4") as ds:
            box = ds.sel(lat=slice(BBOX[1], BBOX[3]), lon=slice(BBOX[0], BBOX[2]))
            tmin = float(box["T2MMIN"].mean()) - 273.15
            tmax = float(box["T2MMAX"].mean()) - 273.15
        rows[f"{m.group(1)}-{m.group(2)}"] = (es_kpa(tmax) + es_kpa(tmin)) / 2 - es_kpa(tmin)
    return pd.Series(rows, name="vpd_proxy_kpa")


def _guards(df: pd.DataFrame) -> None:
    expected = len(pd.period_range(START, END, freq="M"))
    if len(df) != expected or df.drop(columns=["vpd_proxy_kpa"]).isna().any().any():
        raise RuntimeError(f"expected {expected} complete months, got {len(df)}")
    if not (0.1 < df.vpd_kpa.min() and df.vpd_kpa.max() < 5):  # noqa: PLR2004
        raise RuntimeError(f"VPD out of range: {df.vpd_kpa.min():.2f}..{df.vpd_kpa.max():.2f} kPa")
    by_month = df.assign(m=df.year_month.str[5:].astype(int)).groupby("m").vpd_kpa.mean()
    if int(by_month.idxmax()) not in (5, 6, 7):
        raise RuntimeError(f"VPD peaks in month {int(by_month.idxmax())}; expected early summer")
    r = df.vpd_kpa.corr(df.vpd_proxy_kpa)
    if r < 0.85:  # noqa: PLR2004
        raise RuntimeError(f"VPD vs Tmin proxy r = {r:.3f}; the two should agree closely")
    log.info("guards passed: %d months, VPD %.2f..%.2f kPa, peak month %d, r vs Tmin proxy %.3f",
             len(df), df.vpd_kpa.min(), df.vpd_kpa.max(), int(by_month.idxmax()), r)


def main() -> None:
    log.info("=== humidity.py start ===")
    names = granule_names()
    months = [str(p) for p in pd.period_range(START, END, freq="M")]
    missing = [m for m in months if m not in names]
    if missing:
        raise RuntimeError(f"CMR returned no granule for {len(missing)} months, e.g. {missing[:3]}")
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {_token()}", "User-Agent": "eotd-humidity"})
    rows = []
    for i, ym in enumerate(months, 1):
        rows.append({"year_month": ym, **_summarise(_fetch(session, ym, names[ym]))})
        if i % 24 == 0 or i == len(months):
            log.info("  %s (%d/%d)  VPD %.2f kPa", ym, i, len(months), rows[-1]["vpd_kpa"])
    df = pd.DataFrame(rows)
    df = df.merge(vpd_proxy().rename_axis("year_month").reset_index(), on="year_month", how="left")
    _guards(df)
    temp = ROOT / "data" / "Final" / "temperature_monthly.csv"
    if temp.exists():
        t = pd.read_csv(temp).merge(df, on="year_month")
        log.info("T2M vs the model's T2MMEAN: mean |diff| %.3f °C", (t.air_temperature_2m_c - t.temperature_2m_c).abs().mean())
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    log.info("wrote %s (%d rows)", OUTPUT_FILE, len(df))


if __name__ == "__main__":
    main()

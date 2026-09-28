"""Ticket 02: fetch and assemble the DEM and imagery for the superset box.

Each remote piece is saved on its own first (skipped if already on disk, written
via a .part file so an interrupted run never leaves a half file), then the
pieces are merged and reprojected to UTM 12N in one local step.
"""

import json
import math
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from osgeo import gdal

from terrain.dem import UTM_12N, utm_crop_box

gdal.UseExceptions()
FETCH_MARGIN_DEG = 0.01


@dataclass(frozen=True)
class SourcesConfig:
    west: float
    east: float
    south: float
    north: float
    dem_tile_url: str
    dem_tiles_dir: Path
    dem_path: Path
    dem_resolution_m: float
    stac_url: str
    token_url: str
    imagery_items_dir: Path
    imagery_path: Path
    imagery_resolution_m: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.west, self.south, self.east, self.north


def load_sources_config(path: str | Path) -> SourcesConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    box, dem, img = raw["superset"], raw["dem"], raw["imagery"]
    return SourcesConfig(
        west=box["west"], east=box["east"], south=box["south"], north=box["north"],
        dem_tile_url=dem["tile_url"],
        dem_tiles_dir=Path(dem["tiles_dir"]),
        dem_path=Path(dem["merged_path"]),
        dem_resolution_m=dem["resolution_m"],
        stac_url=img["stac_url"],
        token_url=img["token_url"],
        imagery_items_dir=Path(img["items_dir"]),
        imagery_path=Path(img["merged_path"]),
        imagery_resolution_m=img["resolution_m"],
    )


# --- DEM --------------------------------------------------------------------------

def dem_tile_names(west: float, south: float, east: float, north: float) -> list[str]:
    """3DEP 1x1 degree tile names covering the box, e.g. 'n33w111' spans 32-33 N, 111-110 W."""
    names = []
    for top in range(math.floor(south) + 1, math.ceil(north) + 1):
        for left in range(math.floor(west), math.ceil(east)):
            names.append(f"n{top:02d}w{-left:03d}")
    return names


def _remote(url: str) -> str:
    return url if url.startswith("/vsi") or not url.startswith("http") else f"/vsicurl/{url}"


def _write_atomic(dest: Path, write) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    # A .part left by an interrupted run must go: gdal.Warp writes into an existing file.
    part.unlink(missing_ok=True)
    write(str(part))
    part.rename(dest)


def fetch_dem_tiles(cfg: SourcesConfig, log=print) -> list[Path]:
    """Save the part of each 3DEP tile that overlaps the box (plus a margin)."""
    w, s, e, n = (cfg.west - FETCH_MARGIN_DEG, cfg.south - FETCH_MARGIN_DEG,
                  cfg.east + FETCH_MARGIN_DEG, cfg.north + FETCH_MARGIN_DEG)
    paths = []
    for name in dem_tile_names(w, s, e, n):
        dest = cfg.dem_tiles_dir / f"USGS_13_{name}_crop.tif"
        paths.append(dest)
        if dest.exists():
            log(f"  {name}: already on disk")
            continue
        top, left = int(name[1:3]), -int(name[4:7])
        win = [max(w, left), min(n, top), min(e, left + 1), max(s, top - 1)]  # ulx, uly, lrx, lry
        log(f"  {name}: downloading {win}")

        def write(part, src=_remote(cfg.dem_tile_url.format(name=name)), win=win):
            ds = gdal.Translate(part, src, projWin=win, format="GTiff",
                                creationOptions=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES"])
            ds.FlushCache()
            ds = None

        _write_atomic(dest, write)
    return paths


def _warp_to_box(sources: list[Path], dest: Path, cfg: SourcesConfig, resolution_m: float,
                 creation_options: list[str], bands: list[int] | None = None, **warp_options) -> None:
    """Mosaic the pieces straight into the box. Warping from the list (not a VRT) honours
    each piece's nodata or alpha, so one piece's empty edge never blanks its neighbour."""
    xmin, ymin, xmax, ymax = utm_crop_box(*cfg.bbox)

    def write(part):
        common = dict(dstSRS=UTM_12N, outputBounds=(xmin, ymin, xmax, ymax), xRes=resolution_m,
                      yRes=resolution_m, targetAlignedPixels=True, **warp_options)
        if bands is None:
            ds = gdal.Warp(part, [str(p) for p in sources], format="GTiff", multithread=True,
                           creationOptions=creation_options, **common)
        else:
            # gdal.Warp carries an alpha band through to its output (and can't warp several
            # sources to a VRT), so warp to a temporary GeoTIFF and copy out just the wanted bands.
            tmp = part + ".full.tif"
            tds = gdal.Warp(tmp, [str(p) for p in sources], format="GTiff", multithread=True,
                            creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"], **common)
            tds.FlushCache()
            tds = None
            ds = gdal.Translate(part, tmp, format="GTiff", bandList=bands, creationOptions=creation_options)
            Path(tmp).unlink()
        ds.FlushCache()
        ds = None

    if dest.exists():
        dest.unlink()
    _write_atomic(dest, write)


def merge_dem(cfg: SourcesConfig, tiles: list[Path]) -> None:
    _warp_to_box(tiles, cfg.dem_path, cfg, cfg.dem_resolution_m,
                 ["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES", "BIGTIFF=IF_SAFER"],
                 resampleAlg="bilinear", dstNodata=-9999.0)


# --- Imagery ----------------------------------------------------------------------

def _get_json(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def search_naip(cfg: SourcesConfig) -> list[dict]:
    """All NAIP STAC items intersecting the box, following pagination."""
    body = {"collections": ["naip"], "bbox": list(cfg.bbox), "limit": 500}
    features, url = [], cfg.stac_url
    while True:
        page = _get_json(url, body)
        features += page["features"]
        nxt = next((link for link in page.get("links", []) if link.get("rel") == "next"), None)
        if not nxt:
            return features
        url, body = nxt["href"], nxt.get("body", body)


def latest_year_items(features: list[dict]) -> tuple[int, list[dict]]:
    """Keep only the most recent NAIP year, so the mosaic has one flight season's colours."""
    def year(f):
        return int((f["properties"].get("naip:year") or f["properties"]["datetime"][:4]))

    if not features:
        raise ValueError("no NAIP imagery found for the box")
    latest = max(year(f) for f in features)
    return latest, sorted((f for f in features if year(f) == latest), key=lambda f: f["id"])


def fetch_imagery_items(cfg: SourcesConfig, log=print) -> list[Path]:
    """Save each NAIP item's RGB bands, reprojected to UTM 12N at the target resolution.

    Reading at a coarse resolution lets GDAL use the COGs' overviews, so only a
    small fraction of each full-resolution image is downloaded.
    """
    year, items = latest_year_items(search_naip(cfg))
    log(f"  NAIP {year}: {len(items)} images cover the box")
    token = _get_json(cfg.token_url)["token"]
    paths = []
    for i, item in enumerate(items, 1):
        dest = cfg.imagery_items_dir / f"{item['id']}.tif"
        paths.append(dest)
        if dest.exists():
            continue
        log(f"  [{i}/{len(items)}] {item['id']}")
        href = f"/vsicurl/{item['assets']['image']['href']}?{token}"

        def write(part, href=href):
            ds = gdal.Warp(part, href, format="GTiff", dstSRS=UTM_12N,
                           xRes=cfg.imagery_resolution_m, yRes=cfg.imagery_resolution_m,
                           targetAlignedPixels=True, resampleAlg="average", srcBands=[1, 2, 3],
                           dstAlpha=True, creationOptions=["COMPRESS=DEFLATE", "TILED=YES"])
            ds.FlushCache()
            ds = None

        _write_atomic(dest, write)
    return paths


def merge_imagery(cfg: SourcesConfig, items: list[Path]) -> None:
    # The pieces' alpha band masks their empty edges; the mosaic itself is plain RGB.
    _warp_to_box(items, cfg.imagery_path, cfg, cfg.imagery_resolution_m,
                 ["COMPRESS=JPEG", "PHOTOMETRIC=YCBCR", "JPEG_QUALITY=90", "TILED=YES", "BIGTIFF=IF_SAFER"],
                 bands=[1, 2, 3], resampleAlg="average", srcAlpha=True)


# --- Verification -----------------------------------------------------------------

def verify(cfg: SourcesConfig) -> list[str]:
    """One line per product: size, resolution, and any gaps inside the box. Raises on gaps."""
    lines, gaps = [], []
    for label, path in [("DEM", cfg.dem_path), ("Imagery", cfg.imagery_path)]:
        if not path.exists():
            gaps.append(f"{label}: {path} is missing")
            continue
        ds = gdal.Open(str(path))
        gt = ds.GetGeoTransform()
        if label == "DEM":
            band = ds.GetRasterBand(1)
            data = band.ReadAsArray()
            holes = int((data == band.GetNoDataValue()).sum())
            extra = f"elevation {data[data != band.GetNoDataValue()].min():.0f}-{data.max():.0f} m"
        else:
            brightest = ds.GetRasterBand(1).ReadAsArray()
            for b in (2, 3):  # one band at a time: the full mosaic is several hundred MB per band
                np.maximum(brightest, ds.GetRasterBand(b).ReadAsArray(), out=brightest)
            holes = int((brightest == 0).sum())
            extra = f"{ds.RasterCount} bands"
        size_mb = path.stat().st_size / 1e6
        lines.append(f"{label}: {ds.RasterXSize} x {ds.RasterYSize} px at {gt[1]:g} m, "
                     f"{ds.RasterXSize * gt[1] / 1000:.1f} x {ds.RasterYSize * -gt[5] / 1000:.1f} km, "
                     f"{extra}, {size_mb:.0f} MB, {holes} empty pixels")
        if holes:
            gaps.append(f"{label}: {holes} empty pixels inside the box")
        ds = None
    if gaps:
        raise ValueError("\n".join(lines + gaps))
    return lines

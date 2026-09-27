"""Fetch a DEM patch and resample it onto a UTM 12N grid at print resolution."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

UTM_12N = "EPSG:26912"  # NAD83 / UTM zone 12N, matching 3DEP's NAD83 datum
FETCH_MARGIN_DEG = 0.01  # extra DEM around the patch so reprojection never runs off the edge


@dataclass(frozen=True)
class Heightfield:
    """Elevations in metres, row 0 = north, sampled every `spacing_m` in UTM 12N."""

    elevation_m: np.ndarray
    spacing_m: float
    west_m: float
    north_m: float

    @property
    def width_m(self) -> float:
        return (self.elevation_m.shape[1] - 1) * self.spacing_m

    @property
    def height_m(self) -> float:
        return (self.elevation_m.shape[0] - 1) * self.spacing_m


def fetch_patch(url: str, dest: Path, west: float, south: float, east: float, north: float) -> None:
    """Read just the patch (plus a margin) from a remote cloud-optimized GeoTIFF and save it."""
    src_path = url if url.startswith("/vsi") else f"/vsicurl/{url}"
    with rasterio.open(src_path) as src:
        window = from_bounds(
            west - FETCH_MARGIN_DEG,
            south - FETCH_MARGIN_DEG,
            east + FETCH_MARGIN_DEG,
            north + FETCH_MARGIN_DEG,
            transform=src.transform,
        ).round_offsets().round_lengths()
        data = src.read(1, window=window)
        profile = src.profile | {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "transform": src.window_transform(window),
            "compress": "deflate",
        }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(data, 1)


def utm_crop_box(west: float, south: float, east: float, north: float) -> tuple[float, float, float, float]:
    """The largest north-up UTM rectangle inside the lon/lat box, as (xmin, ymin, xmax, ymax)."""
    to_utm = Transformer.from_crs("EPSG:4269", UTM_12N, always_xy=True)
    xs, ys = to_utm.transform([west, east, east, west], [south, south, north, north])
    sw, se, ne, nw = zip(xs, ys)
    return max(sw[0], nw[0]), max(sw[1], se[1]), min(se[0], ne[0]), min(nw[1], ne[1])


def load_heightfield(dem_path: Path, west: float, south: float, east: float, north: float,
                     spacing_m: float) -> Heightfield:
    """Resample the DEM onto a north-up UTM grid of point samples `spacing_m` apart."""
    xmin, ymin, xmax, ymax = utm_crop_box(west, south, east, north)
    ncols = int((xmax - xmin) // spacing_m) + 1
    nrows = int((ymax - ymin) // spacing_m) + 1
    # Samples sit on pixel centres, so the grid's outer edge is half a spacing beyond them.
    dst_transform = from_origin(xmin - spacing_m / 2, ymax + spacing_m / 2, spacing_m, spacing_m)
    elevation = np.full((nrows, ncols), np.nan, dtype=np.float64)
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=elevation,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=UTM_12N,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    if np.isnan(elevation).any():
        missing = int(np.isnan(elevation).sum())
        raise ValueError(f"{missing} samples have no elevation data; the DEM does not cover the patch")
    return Heightfield(elevation, spacing_m, xmin, ymax)

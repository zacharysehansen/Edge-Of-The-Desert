"""Ticket 03: table layout. The screen shows a chosen patch of valley floor; the scale
follows from fitting that patch to the screen, and the table's footprint from the scale."""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from pyproj import Transformer

from terrain.dem import UTM_12N, utm_crop_box

Box = tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax), UTM metres


@dataclass(frozen=True)
class Layout:
    name: str
    title: str
    extent: Box  # lon/lat (west, south, east, north): everything the table covers
    valley: Box  # lon/lat: the valley floor the screen must show in full
    screen_diagonal_in: float
    screen_aspect: tuple[int, int]
    orientation: str  # "ew": long side runs east-west; "ns": long side runs north-south

    @property
    def screen_mm(self) -> tuple[float, float]:
        """Display area as (east-west, north-south) millimetres on the table."""
        a, b = self.screen_aspect
        diag_mm = self.screen_diagonal_in * 25.4
        long_mm, short_mm = diag_mm * a / math.hypot(a, b), diag_mm * b / math.hypot(a, b)
        return (long_mm, short_mm) if self.orientation == "ew" else (short_mm, long_mm)

    @property
    def valley_utm(self) -> Box:
        return utm_crop_box(*self.valley)

    @property
    def table_utm(self) -> Box:
        return utm_crop_box(*self.extent)

    @property
    def scale(self) -> float:
        """Scale denominator at which the whole valley patch just fits on the screen."""
        xmin, ymin, xmax, ymax = self.valley_utm
        sx, sy = self.screen_mm
        return max((xmax - xmin) * 1000 / sx, (ymax - ymin) * 1000 / sy)

    @property
    def screen_utm(self) -> Box:
        """Ground shown on the screen: the screen's size at scale, centred on the valley patch."""
        xmin, ymin, xmax, ymax = self.valley_utm
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        half_w, half_h = (d * self.scale / 1000 / 2 for d in self.screen_mm)
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h

    @property
    def table_mm(self) -> tuple[float, float]:
        xmin, ymin, xmax, ymax = self.table_utm
        return (xmax - xmin) * 1000 / self.scale, (ymax - ymin) * 1000 / self.scale

    def problems(self) -> list[str]:
        """Layout errors that make the design impossible, not just ugly."""
        sx0, sy0, sx1, sy1 = self.screen_utm
        tx0, ty0, tx1, ty1 = self.table_utm
        if sx0 < tx0 or sy0 < ty0 or sx1 > tx1 or sy1 > ty1:
            return ["the screen runs past the table's extent; widen the extent or shrink the valley patch"]
        return []

    def summary(self) -> list[str]:
        sw, sh = self.screen_mm
        tw, th = self.table_mm
        gx0, gy0, gx1, gy1 = self.screen_utm
        tx0, ty0, tx1, ty1 = self.table_utm
        return [
            f"scale 1:{self.scale:,.0f}",
            f'screen {self.screen_diagonal_in:g}" ({sw / 1000:.2f} x {sh / 1000:.2f} m) shows '
            f"{(gx1 - gx0) / 1000:.1f} x {(gy1 - gy0) / 1000:.1f} km",
            f"table {tw / 1000:.2f} x {th / 1000:.2f} m covers {(tx1 - tx0) / 1000:.1f} x {(ty1 - ty0) / 1000:.1f} km",
        ]


def load_layout(path: str | Path) -> Layout:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    ext, val, scr = raw["extent"], raw["valley"], raw["screen"]
    orientation = scr["orientation"]
    if orientation not in ("ew", "ns"):
        raise ValueError('screen.orientation must be "ew" or "ns"')
    a, b = (int(v) for v in scr["aspect"].split(":"))
    return Layout(
        name=raw["name"],
        title=raw["title"],
        extent=(ext["west"], ext["south"], ext["east"], ext["north"]),
        valley=(val["west"], val["south"], val["east"], val["north"]),
        screen_diagonal_in=scr["diagonal_in"],
        screen_aspect=(max(a, b), min(a, b)),
        orientation=orientation,
    )


def to_utm(lon: float, lat: float) -> tuple[float, float]:
    return Transformer.from_crs("EPSG:4269", UTM_12N, always_xy=True).transform(lon, lat)

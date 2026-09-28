"""Ticket 04: the terrain ring around the screen, with the lip that hides the bezel.

Coordinates are table millimetres: x east and y north from the table's south-west
corner, z up from the plywood. Zones, from the centre outward:

  opening   the display area inset by the lip overhang; no terrain here
  lip       overhang + bezel: printed terrain resting on the glass, then over the bezel
  plywood   everything beyond the screen's outer edge
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from terrain.dem import load_heightfield
from terrain.layout import Layout
from terrain.mesh import Mesh, solid_from_fields
from terrain.trim import Outline

Rect = tuple[float, float, float, float]  # (x0, y0, x1, y1) mm
SEAM_SMOOTHING_MM = 20.0  # how much the ground along the seam is smoothed before it sets the datum


@dataclass(frozen=True)
class BuildConfig:
    vertical_exaggeration: float
    base_thickness_mm: float
    mesh_resolution_mm: float
    bed_mm: float
    bezel_mm: float
    bezel_raise_mm: float
    glass_above_plywood_mm: float
    lip_overhang_mm: float
    lip_edge_thickness_mm: float
    blend_mm: float
    datum_blend_mm: float
    test_corner: str
    test_arm_mm: float
    test_depth_mm: float


def load_build(path: str | Path) -> BuildConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    p, s, lip, t = raw["print"], raw["screen"], raw["lip"], raw["test_piece"]
    if t["corner"] not in ("sw", "se", "nw", "ne"):
        raise ValueError('test_piece.corner must be one of "sw", "se", "nw", "ne"')
    return BuildConfig(
        vertical_exaggeration=p["vertical_exaggeration"], base_thickness_mm=p["base_thickness_mm"],
        mesh_resolution_mm=p["mesh_resolution_mm"], bed_mm=p["bed_mm"],
        bezel_mm=s["bezel_mm"], bezel_raise_mm=s["bezel_raise_mm"],
        glass_above_plywood_mm=s["glass_above_plywood_mm"],
        lip_overhang_mm=lip["overhang_mm"], lip_edge_thickness_mm=lip["edge_thickness_mm"],
        blend_mm=lip["blend_mm"], datum_blend_mm=lip["datum_blend_mm"],
        test_corner=t["corner"], test_arm_mm=t["arm_mm"], test_depth_mm=t["depth_mm"],
    )


def _inset(r: Rect, d: float) -> Rect:
    return r[0] + d, r[1] + d, r[2] - d, r[3] - d


def _distance_outside(xs: np.ndarray, ys: np.ndarray, r: Rect) -> np.ndarray:
    """Distance (mm) from each grid node to rectangle r; 0 inside it."""
    dx = np.maximum.reduce([r[0] - xs, np.zeros_like(xs), xs - r[2]])
    dy = np.maximum.reduce([r[1] - ys, np.zeros_like(ys), ys - r[3]])
    return np.hypot(dx[None, :], dy[:, None])


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """0 at t<=0, 1 at t>=1, level at both ends so there is no ledge."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _inside(xs: np.ndarray, ys: np.ndarray, r: Rect) -> np.ndarray:
    return ((xs >= r[0]) & (xs <= r[2]))[None, :] & ((ys >= r[1]) & (ys <= r[3]))[:, None]


class Ring:
    def __init__(self, layout: Layout, build: BuildConfig, dem_path: Path, trimmed: bool = True):
        self.layout, self.build, self.dem_path = layout, build, dem_path
        tx0, ty0, _, _ = layout.table_utm
        sx0, sy0, sx1, sy1 = layout.screen_utm
        k = 1000 / layout.scale  # ground metres -> table mm
        self.table: Rect = (0.0, 0.0, *layout.table_mm)
        self.display: Rect = ((sx0 - tx0) * k, (sy0 - ty0) * k, (sx1 - tx0) * k, (sy1 - ty0) * k)
        self.opening: Rect = _inset(self.display, build.lip_overhang_mm)
        self.screen_outer: Rect = _inset(self.display, -build.bezel_mm)
        self._elevation = None
        self._smoothed = None
        self.outline = Outline(self) if trimmed else None

    # --- fields ------------------------------------------------------------------

    def _elevation_grid(self):
        """Elevation over the whole table at the print resolution, row 0 = north. Loaded once."""
        if self._elevation is None:
            lay = self.layout
            spacing_m = self.build.mesh_resolution_mm * lay.scale / 1000
            w, s, e, n = lay.extent
            self._elevation = load_heightfield(self.dem_path, w, s, e, n, spacing_m)
        return self._elevation

    def _smoothed_grid(self) -> np.ndarray:
        """Elevation with washes and small ridges smoothed out (row 0 = north)."""
        if self._smoothed is None:
            cells = SEAM_SMOOTHING_MM / self.build.mesh_resolution_mm
            self._smoothed = gaussian_filter(self._elevation_grid().elevation_m, cells, mode="nearest")
        return self._smoothed

    def elevation_at(self, xs: np.ndarray, ys: np.ndarray, smoothed: bool = False) -> np.ndarray:
        """Bilinear elevation (m) at table-mm grid nodes; ys increasing northward."""
        hf = self._elevation_grid()
        grid = (self._smoothed_grid() if smoothed else hf.elevation_m)[::-1]  # row 0 = south, to match ys
        step_mm = hf.spacing_m * 1000 / self.layout.scale
        fx = np.clip(xs / step_mm, 0, grid.shape[1] - 1)
        fy = np.clip(ys / step_mm, 0, grid.shape[0] - 1)
        x0 = np.minimum(fx.astype(int), grid.shape[1] - 2)
        y0 = np.minimum(fy.astype(int), grid.shape[0] - 2)
        tx, ty = (fx - x0)[None, :], (fy - y0)[:, None]
        g00 = grid[np.ix_(y0, x0)]
        g01 = grid[np.ix_(y0, x0 + 1)]
        g10 = grid[np.ix_(y0 + 1, x0)]
        g11 = grid[np.ix_(y0 + 1, x0 + 1)]
        return (g00 * (1 - tx) + g01 * tx) * (1 - ty) + (g10 * (1 - tx) + g11 * tx) * ty

    def fields(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Top and underside heights (mm above the plywood) at the grid nodes."""
        b = self.build
        glass = b.glass_above_plywood_mm
        bottom = np.zeros((len(ys), len(xs)))
        bottom[_inside(xs, ys, self.screen_outer)] = glass + b.bezel_raise_mm
        bottom[_inside(xs, ys, self.display)] = glass  # the lip's tip rests on the glass

        k = b.vertical_exaggeration / self.layout.scale * 1000  # ground metres -> model mm
        elevation = self.elevation_at(xs, ys)
        d = _distance_outside(xs, ys, self.opening)

        # Vertical datum. Far from the screen, heights count up from the table's lowest
        # point. The screen is flat but the valley is not, so near the screen they count
        # up from the (smoothed) ground at the nearest point of the seam instead, which
        # puts the terrain level with the glass all the way round.
        # Clamping each axis to the opening gives the nearest point on its edge.
        o = self.opening
        seam = self.elevation_at(np.clip(xs, o[0], o[2]), np.clip(ys, o[1], o[3]), smoothed=True)
        low = self._elevation_grid().elevation_m.min()
        w = 1 - _smoothstep(d / b.datum_blend_mm)
        datum = w * seam + (1 - w) * low

        # Thickness under the terrain grows from the lip's tip to the full base, and the
        # local ups and downs of the ground fade in over the same distance.
        ease = _smoothstep(d / b.blend_mm)
        # The level the terrain stands on eases from the glass up to whichever is highest
        # of glass, raised bezel and plywood, so the top surface never steps.
        floor = max(glass, glass + b.bezel_raise_mm, 0.0)
        level = glass + (floor - glass) * ease
        thickness = b.lip_edge_thickness_mm + (b.base_thickness_mm - b.lip_edge_thickness_mm) * ease
        top = level + thickness + ease * k * (elevation - datum)
        top = np.maximum(top, bottom + b.lip_edge_thickness_mm)
        return top, bottom

    # --- meshes ------------------------------------------------------------------

    def grid(self, region: Rect, resolution_mm: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Node coordinates covering `region`, with every zone edge on a grid line exactly."""
        res = resolution_mm or self.build.mesh_resolution_mm
        axes = []
        for lo, hi, edges in [
            (region[0], region[2], [self.opening[0], self.opening[2], self.display[0], self.display[2],
                                    self.screen_outer[0], self.screen_outer[2]]),
            (region[1], region[3], [self.opening[1], self.opening[3], self.display[1], self.display[3],
                                    self.screen_outer[1], self.screen_outer[3]]),
        ]:
            n = max(int(np.ceil((hi - lo) / res)), 1)
            exact = np.array([lo, hi] + [e for e in edges if lo < e < hi])
            regular = np.linspace(lo, hi, n + 1)
            # Drop regular ticks that would leave a sliver next to an exact edge.
            near = np.abs(regular[:, None] - exact[None, :]).min(axis=1) < res / 4
            axes.append(np.unique(np.concatenate([regular[~near], exact])))
        return axes[0], axes[1]

    def cells(self, xs: np.ndarray, ys: np.ndarray, region: Rect, shape: Rect | None = None) -> np.ndarray:
        """Kept cells: inside the table (and `shape`, if given), outside the opening."""
        cx, cy = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
        keep = _inside(cx, cy, self.table) & ~_inside(cx, cy, self.opening)
        if self.outline is not None:
            keep &= self.outline.keep(cx, cy)
        if shape is not None:
            keep &= _inside(cx, cy, shape)
        return keep

    def mesh(self, region: Rect, resolution_mm: float | None = None, shape: Rect | None = None) -> Mesh:
        xs, ys = self.grid(region, resolution_mm)
        top, bottom = self.fields(xs, ys)
        return solid_from_fields(xs, ys, top, bottom, self.cells(xs, ys, region, shape))

    def test_piece_region(self) -> tuple[Rect, list[Rect]]:
        """The L-shaped lip fit-test piece at one corner of the opening.

        Returns its bounding box and the two arm rectangles; the union of the arms
        (minus the opening) is the piece.
        """
        b = self.build
        o = self.opening
        ew, ns = b.test_corner[1], b.test_corner[0]
        cx = o[0] if ew == "w" else o[2]
        cy = o[1] if ns == "s" else o[3]
        out_x = -b.test_depth_mm if ew == "w" else b.test_depth_mm
        out_y = -b.test_depth_mm if ns == "s" else b.test_depth_mm
        in_x = b.test_arm_mm if ew == "w" else -b.test_arm_mm
        in_y = b.test_arm_mm if ns == "s" else -b.test_arm_mm

        def rect(x_a, x_b, y_a, y_b):
            return min(x_a, x_b), min(y_a, y_b), max(x_a, x_b), max(y_a, y_b)

        # Each arm runs along one side of the opening, from the outer edge of the piece inward.
        arm_x = rect(cx + out_x, cx + in_x, cy + out_y, cy)  # along the south/north side
        arm_y = rect(cx + out_x, cx, cy + out_y, cy + in_y)  # along the west/east side
        box = rect(cx + out_x, cx + in_x, cy + out_y, cy + in_y)
        return box, [arm_x, arm_y]

    def test_piece(self) -> Mesh:
        box, arms = self.test_piece_region()
        xs, ys = self.grid(box)
        top, bottom = self.fields(xs, ys)
        cx, cy = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
        in_arms = _inside(cx, cy, arms[0]) | _inside(cx, cy, arms[1])
        cells = self.cells(xs, ys, box) & in_arms
        return solid_from_fields(xs, ys, top, bottom, cells)

"""Ticket 05: where the printed terrain ends.

On a crest side the diorama keeps the slopes whose water drains out of the table
through the same outlet as the screen's valley (for Tucson, the Santa Cruz leaving
to the north-west), plus a margin beyond the crest. The far side of a range drains
somewhere else (the San Pedro, Avra Valley, Sonoita Creek), so the drainage divide
is the crest line, even where a range curves.
"""

import heapq

import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter, label

from terrain.layout import to_utm

ANALYSIS_RES_M = 100.0  # ground spacing of the drainage analysis
MAIN_OUTLET_SHARE = 0.05  # an outlet counts as the valley's if this share of the screen drains to it


def drainage_roots(elevation: np.ndarray) -> np.ndarray:
    """For each cell, the flat index of the edge cell its water leaves the grid through.

    Priority-flood: grow inward from the edges, lowest first; each cell drains to the
    neighbour that reached it. Pits are filled implicitly, so every cell has a path out.
    """
    h, w = elevation.shape
    flat = elevation.ravel().tolist()
    root = [-1] * (h * w)
    heap = []
    for r in range(h):
        for c in ((0, w - 1) if 0 < r < h - 1 else range(w)):
            i = r * w + c
            root[i] = i
            heap.append((flat[i], i))
    heapq.heapify(heap)
    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        level, i = heapq.heappop(heap)
        r, c = divmod(i, w)
        for dr, dc in neighbours:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                j = rr * w + cc
                if root[j] < 0:
                    root[j] = root[i]
                    heapq.heappush(heap, (max(flat[j], level), j))
    return np.array(root).reshape(h, w)


class Outline:
    """Which parts of the table get printed terrain. Coordinates are table mm."""

    def __init__(self, ring):
        self.ring = ring
        lay = ring.layout
        self.trim = lay.trim
        self.step_mm = ANALYSIS_RES_M * 1000 / lay.scale
        self._crest = None
        tx0, ty0, _, _ = lay.table_utm

        def to_mm(poly):
            pts = [to_utm(lon, lat) for lon, lat in poly]
            return PolygonPath([((x - tx0) * 1000 / lay.scale, (y - ty0) * 1000 / lay.scale) for x, y in pts])

        self.keep_polys = [to_mm(p) for p in self.trim.keep]
        self.cut_polys = [to_mm(p) for p in self.trim.cut]

    # --- the crest field -----------------------------------------------------------

    def crest_field(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(xs, ys, F): F >= 0.5 where a crest side keeps terrain. Computed once."""
        if self._crest is None:
            r = self.ring
            xs = np.arange(0, r.table[2] + self.step_mm, self.step_mm)
            ys = np.arange(0, r.table[3] + self.step_mm, self.step_mm)
            roots = drainage_roots(r.elevation_at(xs, ys))

            o = r.opening
            on_screen = ((xs > o[0]) & (xs < o[2]))[None, :] & ((ys > o[1]) & (ys < o[3]))[:, None]
            outlets, counts = np.unique(roots[on_screen], return_counts=True)
            main = outlets[counts >= MAIN_OUTLET_SHARE * on_screen.sum()]
            basin = np.isin(roots, main) | on_screen

            # One piece around the screen, with no pockets inside it.
            labels, _ = label(basin)
            basin = labels == labels[on_screen].max()
            basin = binary_fill_holes(basin)

            margin_mm = self.trim.margin_m * 1000 / r.layout.scale
            kept = distance_transform_edt(~basin) * self.step_mm <= margin_mm
            field = gaussian_filter(kept.astype(float), self.trim.smooth_mm / self.step_mm, mode="nearest")
            self._crest = (xs, ys, field)
        return self._crest

    def _crest_keep(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        gx, gy, field = self.crest_field()
        fx = np.clip(xs / self.step_mm, 0, len(gx) - 1)
        fy = np.clip(ys / self.step_mm, 0, len(gy) - 1)
        x0 = np.minimum(fx.astype(int), len(gx) - 2)
        y0 = np.minimum(fy.astype(int), len(gy) - 2)
        tx, ty = (fx - x0)[None, :], (fy - y0)[:, None]
        v = (field[np.ix_(y0, x0)] * (1 - tx) + field[np.ix_(y0, x0 + 1)] * tx) * (1 - ty) \
            + (field[np.ix_(y0 + 1, x0)] * (1 - tx) + field[np.ix_(y0 + 1, x0 + 1)] * tx) * ty
        return v >= 0.5

    # --- the outline -----------------------------------------------------------------

    def sector(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Which side of the screen each point lies beyond: 0 north, 1 east, 2 south, 3 west.

        The four sectors meet along the screen's diagonals.
        """
        o = self.ring.opening
        cx, cy = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
        hw, hh = (o[2] - o[0]) / 2, (o[3] - o[1]) / 2
        dx, dy = ((xs - cx) / hw)[None, :], ((ys - cy) / hh)[:, None]
        east_west = np.abs(dx) > np.abs(dy)
        return np.where(east_west, np.where(dx > 0, 1, 3), np.where(dy > 0, 0, 2))

    def keep(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Boolean grid (len(ys), len(xs)): True where terrain is printed (ignoring the opening)."""
        r = self.ring
        sector = self.sector(xs, ys)
        rules = [self.trim.sides[s] for s in ("north", "east", "south", "west")]
        out = np.zeros(sector.shape, bool)
        if "crest" in rules:
            crest = self._crest_keep(xs, ys)
        if "strip" in rules:
            o = r.opening
            dx = np.maximum.reduce([o[0] - xs, np.zeros_like(xs), xs - o[2]])[None, :]
            dy = np.maximum.reduce([o[1] - ys, np.zeros_like(ys), ys - o[3]])[:, None]
            strip = np.maximum(dx, dy) <= self.trim.strip_mm
        for i, rule in enumerate(rules):
            here = sector == i
            if rule == "crest":
                out |= here & crest
            elif rule == "strip":
                out |= here & strip
            elif rule == "full":
                out |= here
        if self.keep_polys or self.cut_polys:
            pts = np.column_stack([np.repeat(xs[None, :], len(ys), 0).ravel(),
                                   np.repeat(ys[:, None], len(xs), 1).ravel()])
            for p in self.keep_polys:
                out |= p.contains_points(pts).reshape(out.shape)
            for p in self.cut_polys:
                out &= ~p.contains_points(pts).reshape(out.shape)
        inside_table = ((xs >= 0) & (xs <= r.table[2]))[None, :] & ((ys >= 0) & (ys <= r.table[3]))[:, None]
        return out & inside_table

    def area_m2(self, resolution_mm: float = 2.0) -> tuple[float, float]:
        """Printed area (m^2) without and with the trim, both excluding the opening."""
        r = self.ring
        xs = np.arange(resolution_mm / 2, r.table[2], resolution_mm)
        ys = np.arange(resolution_mm / 2, r.table[3], resolution_mm)
        o = r.opening
        opening = ((xs > o[0]) & (xs < o[2]))[None, :] & ((ys > o[1]) & (ys < o[3]))[:, None]
        cell = (resolution_mm / 1000) ** 2
        return float((~opening).sum() * cell), float((self.keep(xs, ys) & ~opening).sum() * cell)

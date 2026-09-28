"""Ticket 06: split the ring into transport sections and bed-sized print tiles.

Sections: the ring is cut from the screen's edge out to the rim along K paths that
prefer washes (where a seam hides best). One Dijkstra from the rim gives every
cell its cheapest path outward, so trying many sets of cut positions is cheap.

Tiles: a square grid, bed-sized, offset to leave as few slivers as possible. A
square crossed by a section boundary becomes one tile per section. Neighbouring
tiles share exact grid lines, so their edges match.
"""

import heapq
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, label

from terrain.checks import signed_volume
from terrain.mesh import Mesh, solid_from_fields
from terrain.ring import Rect, Ring

FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
WASH_PREFERENCE = 4.0  # crossing open ground costs up to this much more than following a wash
OFFSET_TRIES = 12  # sets of cut positions tried per section count
SMALL_FRAGMENT = 0.03  # ring pieces smaller than this share are folded into a neighbour
CRUMB_SHARE = 0.01  # tile pieces smaller than this share of a square, with nothing to join, are dropped


@dataclass(frozen=True)
class TileSettings:
    tile_mm: float
    max_height_mm: float
    min_tile_share: float
    section_count: int
    door_mm: float
    max_length_mm: float
    wall_mm: float
    infill: float
    flow_mm3_per_s: float


def load_tile_settings(path: str | Path) -> TileSettings:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    t, s, pt = raw["tiles"], raw["sections"], raw["print_time"]
    return TileSettings(
        tile_mm=raw["print"]["bed_mm"] - t["clearance_mm"], max_height_mm=t["max_height_mm"],
        min_tile_share=t["min_tile_share"], section_count=s["count"], door_mm=s["door_mm"],
        max_length_mm=s["max_length_mm"], wall_mm=pt["wall_mm"], infill=pt["infill"],
        flow_mm3_per_s=pt["flow_mm3_per_s"],
    )


# --- sections ----------------------------------------------------------------------

def _dijkstra_from(sources: np.ndarray, allowed: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Predecessor (flat index) of each allowed cell on its cheapest path to any source; -1 at sources."""
    h, w = allowed.shape
    allowed_f = allowed.ravel().tolist()
    cost_f = cost.ravel().tolist()
    dist = [math.inf] * (h * w)
    pred = [-1] * (h * w)
    heap = []
    for i in np.flatnonzero(sources.ravel()).tolist():
        dist[i] = 0.0
        heap.append((0.0, i))
    heapq.heapify(heap)
    steps = [(dr, dc, math.hypot(dr, dc)) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i]:
            continue
        r, c = divmod(i, w)
        for dr, dc, step in steps:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                j = rr * w + cc
                if allowed_f[j]:
                    nd = d + step * cost_f[j]
                    if nd < dist[j]:
                        dist[j] = nd
                        pred[j] = i
                        heapq.heappush(heap, (nd, j))
    return np.array(pred)


class Sections:
    """Transport sections over the ring, found on the drainage-analysis grid."""

    def __init__(self, ring: Ring, settings: TileSettings):
        self.ring, self.settings = ring, settings
        outline = ring.outline
        self.xs, self.ys, _ = outline.crest_field()
        self.step = outline.step_mm
        xs, ys, o = self.xs, self.ys, ring.opening
        self.opening = ((xs > o[0]) & (xs < o[2]))[None, :] & ((ys > o[1]) & (ys < o[3]))[:, None]
        self.ring_mask = outline.keep(xs, ys) & ~self.opening
        acc = outline.accumulation
        ref = np.percentile(acc[self.ring_mask], 99)
        wash = np.clip(np.log1p(acc) / np.log1p(ref), 0, 1)
        cost = 1 + WASH_PREFERENCE * (1 - wash)

        outside = ~self.ring_mask & ~self.opening
        rim = self.ring_mask & binary_dilation(outside, FOUR)
        inner = self.ring_mask & binary_dilation(self.opening, FOUR)
        self.pred = _dijkstra_from(rim, self.ring_mask, cost)
        self.inner_cells, self.inner_pos = self._perimeter_order(inner)
        self.labels, self.count = self._choose()

    def _perimeter_order(self, inner: np.ndarray):
        """Inner-edge cells and their position (mm) along the opening's perimeter, clockwise from NW."""
        o = self.ring.opening
        rows, cols = np.nonzero(inner)
        x, y = self.xs[cols], self.ys[rows]
        w, h = o[2] - o[0], o[3] - o[1]
        px, py = np.clip(x, o[0], o[2]), np.clip(y, o[1], o[3])
        # Nearest side of the rectangle, then distance along the perimeter.
        d = np.stack([o[3] - py, o[2] - px, py - o[1], px - o[0]])  # to north, east, south, west sides
        side = np.argmin(np.stack([np.abs(y - o[3]) + (x < o[0]) * 1e9 + (x > o[2]) * 1e9,
                                   np.abs(x - o[2]) + (y < o[1]) * 1e9 + (y > o[3]) * 1e9,
                                   np.abs(y - o[1]) + (x < o[0]) * 1e9 + (x > o[2]) * 1e9,
                                   np.abs(x - o[0]) + (y < o[1]) * 1e9 + (y > o[3]) * 1e9]), axis=0)
        corner = (x < o[0]) | (x > o[2])
        corner &= (y < o[1]) | (y > o[3])
        pos = np.select([side == 0, side == 1, side == 2, side == 3],
                        [px - o[0], w + (o[3] - py), w + h + (o[2] - px), 2 * w + h + (py - o[1])])
        del d, corner
        return np.flatnonzero(inner.ravel()), pos

    def _cut(self, starts: list[int]) -> np.ndarray:
        cut = np.zeros(self.ring_mask.size, bool)
        for i in starts:
            while i >= 0:
                cut[i] = True
                i = self.pred[i]
        return cut.reshape(self.ring_mask.shape)

    def _split(self, starts: list[int]) -> np.ndarray:
        """Section labels 1..n on the analysis grid (0 off the ring)."""
        cut = self._cut(starts)
        labels, n = label(self.ring_mask & ~cut, FOUR)
        sizes = np.bincount(labels.ravel(), minlength=n + 1)
        big = [k for k in range(1, n + 1) if sizes[k] >= SMALL_FRAGMENT * self.ring_mask.sum()]
        keep = np.isin(labels, big)
        # Cut cells and small fragments join the nearest big section.
        _, (ri, ci) = distance_transform_edt(~keep, return_indices=True)
        filled = labels[ri, ci]
        filled[~self.ring_mask] = 0
        remap = np.zeros(n + 1, int)
        for new, old in enumerate(big, 1):
            remap[old] = new
        return remap[filled]

    def _bbox_mm(self, labels: np.ndarray, k: int) -> tuple[float, float, float, float]:
        rows, cols = np.nonzero(labels == k)
        return (self.xs[cols.min()], self.ys[rows.min()], self.xs[cols.max()] + self.step,
                self.ys[rows.max()] + self.step)

    def _score(self, labels: np.ndarray, n: int):
        """(constraint violation in mm, largest section's area): lower is better."""
        violation, largest = 0.0, 0
        for k in range(1, n + 1):
            x0, y0, x1, y1 = self._bbox_mm(labels, k)
            short, long_ = sorted((x1 - x0, y1 - y0))
            violation += max(0, short - self.settings.door_mm) + max(0, long_ - self.settings.max_length_mm)
            largest = max(largest, int((labels == k).sum()))
        return violation, largest

    def _choose(self):
        perimeter = 2 * ((self.ring.opening[2] - self.ring.opening[0]) + (self.ring.opening[3] - self.ring.opening[1]))
        best = None
        for count in range(self.settings.section_count, self.settings.section_count + 3):
            for t in range(OFFSET_TRIES):
                offset = perimeter / count * t / OFFSET_TRIES
                targets = [(offset + i * perimeter / count) % perimeter for i in range(count)]
                starts = [int(self.inner_cells[np.argmin(np.abs(self.inner_pos - p))]) for p in targets]
                labels = self._split(starts)
                n = int(labels.max())
                violation, largest = self._score(labels, n)
                key = (violation > 0, n != count, violation, largest)
                if best is None or key < best[0]:
                    best = (key, labels, n)
            if best[0][0] is False:  # this many sections satisfies the door and length limits
                break
        _, labels, n = best
        return self._order(labels, n), n

    def _order(self, labels: np.ndarray, n: int) -> np.ndarray:
        """Renumber sections clockwise from north by the angle of their centroid."""
        o = self.ring.opening
        cx, cy = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
        angles = []
        for k in range(1, n + 1):
            rows, cols = np.nonzero(labels == k)
            a = math.atan2(self.xs[cols].mean() - cx, self.ys[rows].mean() - cy)  # 0 = north, clockwise
            angles.append((a % (2 * math.pi), k))
        remap = np.zeros(n + 1, int)
        for new, (_, old) in enumerate(sorted(angles), 1):
            remap[old] = new
        return remap[labels]

    def smoothed(self):
        """Per-section indicator fields, smoothed so boundaries are curves, not stairs."""
        if not hasattr(self, "_smooth"):
            self._smooth = np.stack([gaussian_filter((self.labels == k).astype(float), 1.0)
                                     for k in range(1, self.count + 1)])
        return self._smooth

    def label_at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Section number (1..n) at each grid node; meaningful only where the ring is kept."""
        stack = self.smoothed()
        fx = np.clip((xs - self.xs[0]) / self.step, 0, len(self.xs) - 1)
        fy = np.clip((ys - self.ys[0]) / self.step, 0, len(self.ys) - 1)
        x0 = np.minimum(fx.astype(int), len(self.xs) - 2)
        y0 = np.minimum(fy.astype(int), len(self.ys) - 2)
        tx, ty = (fx - x0)[None, None, :], (fy - y0)[None, :, None]
        v = (stack[:, y0][:, :, x0] * (1 - tx) + stack[:, y0][:, :, x0 + 1] * tx) * (1 - ty) \
            + (stack[:, y0 + 1][:, :, x0] * (1 - tx) + stack[:, y0 + 1][:, :, x0 + 1] * tx) * ty
        return np.argmax(v, axis=0) + 1


# --- tiles -------------------------------------------------------------------------

@dataclass
class Tile:
    name: str
    section: int
    square: Rect
    share: float  # printed area as a share of a full square
    folded: tuple = ()  # sections whose small pieces of this square were folded into this tile


class TilePlan:
    """The tile grid, and which section every point of every tile belongs to."""

    COARSE_MM = 2.0

    def __init__(self, ring: Ring, sections: Sections, settings: TileSettings):
        self.ring, self.sections, self.settings = ring, sections, settings
        self.size = settings.tile_mm
        xs, ys = self._coarse()
        kept = ring.cells(np.append(xs - self.COARSE_MM / 2, xs[-1] + self.COARSE_MM / 2),
                          np.append(ys - self.COARSE_MM / 2, ys[-1] + self.COARSE_MM / 2), ring.table)
        self._kept = kept
        self._labels = np.where(kept, sections.label_at(xs, ys), 0)
        self.origin = self._best_offset()
        self.tiles, self.fold = self._make_tiles()

    def _coarse(self):
        r = self.ring
        return (np.arange(self.COARSE_MM / 2, r.table[2], self.COARSE_MM),
                np.arange(self.COARSE_MM / 2, r.table[3], self.COARSE_MM))

    def _pieces(self, ox: float, oy: float):
        """(keys, counts): coarse cells per (square column, square row, section)."""
        xs, ys = self._coarse()
        if not hasattr(self, "_nz"):
            self._nz = np.nonzero(self._labels)
        rows, cols = self._nz
        ci = np.floor((xs - ox) / self.size).astype(int)[cols] + 1  # offsets are negative: +1 keeps >= 0
        ri = np.floor((ys - oy) / self.size).astype(int)[rows] + 1
        sec = self._labels[rows, cols]
        ncol, nrow, nsec = ci.max() + 1, ri.max() + 1, sec.max() + 1
        counts = np.bincount((ci * nrow + ri) * nsec + sec, minlength=ncol * nrow * nsec)
        keys = np.flatnonzero(counts)
        c, rest = np.divmod(keys, nrow * nsec)
        r, sc = np.divmod(rest, nsec)
        return np.stack([c - 1, r - 1, sc], axis=1), counts[keys]

    def _best_offset(self) -> tuple[float, float]:
        full = (self.size / self.COARSE_MM) ** 2
        best = None
        for ox in np.arange(-self.size, 0, 10.0):
            for oy in np.arange(-self.size, 0, 10.0):
                _, counts = self._pieces(ox, oy)
                small = int((counts < self.settings.min_tile_share * full).sum())
                key = (small, len(counts))
                if best is None or key < best[0]:
                    best = (key, (float(ox), float(oy)))
        return best[1]

    def _make_tiles(self):
        full = (self.size / self.COARSE_MM) ** 2
        uniq, counts = self._pieces(*self.origin)
        fold = {}  # (col, row, small section) -> section it is printed with (0: dropped crumb)
        by_square = {}
        for (c, r, s), n in zip(uniq.tolist(), counts.tolist()):
            by_square.setdefault((c, r), []).append((n, s))
        tiles = []
        ox, oy = self.origin
        self.dropped = []
        for (c, r), pieces in sorted(by_square.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
            pieces.sort(reverse=True)
            # A crumb alone in its square, too small to print, is left off the table.
            crumbs = [(n, s) for n, s in pieces if n < CRUMB_SHARE * full]
            if len(crumbs) == len(pieces):
                for _, s in crumbs:
                    fold[(c, r, s)] = 0
                self.dropped.append(sum(n for n, _ in crumbs) * self.COARSE_MM ** 2)
                continue
            main_n, main_s = pieces[0]
            keep = [(n, s) for n, s in pieces if n >= self.settings.min_tile_share * full or s == main_s]
            folded = [s for n, s in pieces if (n, s) not in keep]
            for s in folded:  # a sliver of another section rides along with the square's main tile
                fold[(c, r, s)] = main_s
            square = (ox + c * self.size, oy + r * self.size, ox + (c + 1) * self.size, oy + (r + 1) * self.size)
            for n, s in keep:
                extra = sum(m for m, t in pieces if t in folded) if s == main_s else 0
                tiles.append(Tile("", s, square, (n + extra) / full, tuple(folded) if s == main_s else ()))
        # Name tiles by section, then row (north first) and column.
        per_section = {}
        for t in sorted(tiles, key=lambda t: (t.section, -t.square[1], t.square[0])):
            per_section[t.section] = per_section.get(t.section, 0) + 1
            t.name = f"S{t.section}-{per_section[t.section]:02d}"
        return sorted(tiles, key=lambda t: t.name), fold

    def owner_at(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Section that prints each point of a (len(ys), len(xs)) grid; 0 where nothing is printed.

        Applies the folds and dropped crumbs, so plywood templates and tiles agree.
        """
        r = self.ring
        o = r.opening
        kept = r.outline.keep(xs, ys) & ~(((xs > o[0]) & (xs < o[2]))[None, :] & ((ys > o[1]) & (ys < o[3]))[:, None])
        owner = np.where(kept, self.sections.label_at(xs, ys), 0)
        ox, oy = self.origin
        ci = np.floor((xs - ox) / self.size).astype(int)[None, :].repeat(len(ys), 0)
        ri = np.floor((ys - oy) / self.size).astype(int)[:, None].repeat(len(xs), 1)
        for (c, rr, sec), into in self.fold.items():
            owner = np.where((ci == c) & (ri == rr) & (owner == sec), into, owner)
        return owner

    def cell_mask(self, tile: Tile, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Cells of this tile's square that belong to it."""
        cx, cy = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
        inside = ((cx >= tile.square[0]) & (cx <= tile.square[2]))[None, :] \
            & ((cy >= tile.square[1]) & (cy <= tile.square[3]))[:, None]
        table = ((cx >= 0) & (cx <= self.ring.table[2]))[None, :] & ((cy >= 0) & (cy <= self.ring.table[3]))[:, None]
        return inside & table & (self.owner_at(cx, cy) == tile.section)

    def mesh(self, tile: Tile, resolution_mm: float | None = None) -> Mesh:
        xs, ys = self.ring.grid(tile.square, resolution_mm)
        top, bottom = self.ring.fields(xs, ys)
        return solid_from_fields(xs, ys, top, bottom, self.cell_mask(tile, xs, ys))

    def print_hours(self, mesh: Mesh) -> float:
        """Rough: a shell of wall_mm over the whole surface plus sparse infill inside."""
        v = mesh.vertices[mesh.faces]
        area = float(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2)
        volume = signed_volume(mesh)
        shell = min(volume, area * self.settings.wall_mm)
        return (shell + (volume - shell) * self.settings.infill) / self.settings.flow_mm3_per_s / 3600

"""Turn a heightfield into a closed, printable triangle mesh (millimetres)."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (n, 3) float, mm
    faces: np.ndarray  # (m, 3) int, counter-clockwise seen from outside


def heightfield_to_mesh(elevation_m: np.ndarray, spacing_mm: float, scale: float,
                        vertical_exaggeration: float, base_thickness_mm: float) -> Mesh:
    """A solid block: terrain on top, flat base at z=0, vertical side walls.

    Row 0 of `elevation_m` is north; x runs east and y runs north. The lowest
    point of the terrain sits `base_thickness_mm` above the base.
    """
    nrows, ncols = elevation_m.shape
    if nrows < 2 or ncols < 2:
        raise ValueError("heightfield needs at least 2x2 samples")

    relief_mm = (elevation_m - elevation_m.min()) * vertical_exaggeration / scale * 1000.0
    rows, cols = np.mgrid[0:nrows, 0:ncols]
    top = np.column_stack([
        (cols * spacing_mm).ravel(),
        ((nrows - 1 - rows) * spacing_mm).ravel(),
        (base_thickness_mm + relief_mm).ravel(),
    ])

    def idx(r, c):
        return r * ncols + c

    # Top surface: two triangles per cell, counter-clockwise seen from above.
    r, c = np.mgrid[0:nrows - 1, 0:ncols - 1]
    nw, ne, sw, se = idx(r, c), idx(r, c + 1), idx(r + 1, c), idx(r + 1, c + 1)
    top_faces = np.concatenate([
        np.stack([sw, se, ne], axis=-1).reshape(-1, 3),
        np.stack([sw, ne, nw], axis=-1).reshape(-1, 3),
    ])

    # Perimeter of the top surface, counter-clockwise seen from above.
    south = [idx(nrows - 1, c) for c in range(ncols)]
    east = [idx(r, ncols - 1) for r in range(nrows - 2, -1, -1)]
    north = [idx(0, c) for c in range(ncols - 2, -1, -1)]
    west = [idx(r, 0) for r in range(1, nrows - 1)]
    loop = np.array(south + east + north + west)

    # Base vertices directly under the perimeter, plus one centre vertex for the base fan.
    n_top = len(top)
    bottom = top[loop].copy()
    bottom[:, 2] = 0.0
    centre = np.array([[top[:, 0].max() / 2, top[:, 1].max() / 2, 0.0]])
    vertices = np.vstack([top, bottom, centre])

    a_top, b_top = loop, np.roll(loop, -1)
    a_bot = n_top + np.arange(len(loop))
    b_bot = np.roll(a_bot, -1)
    wall_faces = np.concatenate([
        np.column_stack([a_bot, b_bot, b_top]),
        np.column_stack([a_bot, b_top, a_top]),
    ])
    centre_idx = len(vertices) - 1
    base_faces = np.column_stack([np.full(len(loop), centre_idx), b_bot, a_bot])

    faces = np.concatenate([top_faces, wall_faces, base_faces]).astype(np.int64)
    return Mesh(vertices, faces)


def flush_edge_top(elevation_m: np.ndarray, spacing_mm: float, scale: float,
                   vertical_exaggeration: float, base_thickness_mm: float, edge: str,
                   edge_thickness_mm: float, blend_mm: float, datum_blend_mm: float,
                   smoothing_mm: float) -> np.ndarray:
    """Top heights (mm) for a tile whose `edge` ("n", "s", "e" or "w") meets the screen glass.

    The same seam the ring uses (terrain/ring.py), along one straight edge: at the edge the
    tile is `edge_thickness_mm` thick and level with the glass, over `blend_mm` it thickens
    to the full base and the ground's bumps fade in, and over `datum_blend_mm` heights ease
    from "above the smoothed ground at the edge" to "above the tile's lowest point".
    Rows and columns match `elevation_m` (row 0 = north).
    """
    turns = {"s": 0, "w": 1, "n": 2, "e": 3}[edge]  # quarter turns that bring the edge to the bottom row
    elev = np.rot90(elevation_m, turns)
    smoothed = gaussian_filter(elev, smoothing_mm / spacing_mm, mode="nearest")
    d = (elev.shape[0] - 1 - np.arange(elev.shape[0]))[:, None] * spacing_mm  # distance from the edge

    def smoothstep(t):
        t = np.clip(t, 0, 1)
        return t * t * (3 - 2 * t)

    w = 1 - smoothstep(d / datum_blend_mm)
    datum = w * smoothed[-1][None, :] + (1 - w) * elev.min()
    ease = smoothstep(d / blend_mm)
    thickness = edge_thickness_mm + (base_thickness_mm - edge_thickness_mm) * ease
    k = vertical_exaggeration / scale * 1000
    top = np.maximum(thickness + ease * k * (elev - datum), edge_thickness_mm)
    return np.rot90(top, -turns)


def top_to_mesh(top_mm: np.ndarray, spacing_mm: float) -> Mesh:
    """A solid with the given top heights (row 0 = north) over a flat base at z=0."""
    nrows, ncols = top_mm.shape
    xs, ys = np.arange(ncols) * spacing_mm, np.arange(nrows) * spacing_mm
    return solid_from_fields(xs, ys, top_mm[::-1], np.zeros_like(top_mm),
                             np.ones((nrows - 1, ncols - 1), dtype=bool))


def solid_from_fields(xs: np.ndarray, ys: np.ndarray, top: np.ndarray, bottom: np.ndarray,
                      cells: np.ndarray) -> Mesh:
    """A closed solid over a rectilinear grid, possibly with holes.

    `xs` (east) and `ys` (north) are increasing node coordinates in mm. `top` and
    `bottom` are (len(ys), len(xs)) surface heights at the nodes, with top above
    bottom wherever a kept cell touches. `cells` is a (len(ys)-1, len(xs)-1) mask
    of the grid cells that belong to the solid; walls are built wherever a kept
    cell borders a dropped cell or the grid's edge.
    """
    ny, nx = len(ys), len(xs)
    if top.shape != (ny, nx) or bottom.shape != (ny, nx) or cells.shape != (ny - 1, nx - 1):
        raise ValueError("field shapes do not match the grid")
    gx, gy = np.meshgrid(xs, ys)
    n = ny * nx
    vertices = np.concatenate([
        np.column_stack([gx.ravel(), gy.ravel(), top.ravel()]),
        np.column_stack([gx.ravel(), gy.ravel(), bottom.ravel()]),
    ])

    i, j = np.nonzero(cells)
    p00, p01 = i * nx + j, i * nx + j + 1  # south-west, south-east
    p10, p11 = (i + 1) * nx + j, (i + 1) * nx + j + 1  # north-west, north-east
    faces = [
        np.column_stack([p00, p01, p11]), np.column_stack([p00, p11, p10]),  # top, facing up
        np.column_stack([p00, p11, p01]) + n, np.column_stack([p00, p10, p11]) + n,  # bottom, facing down
    ]

    # Walls: a cell edge is on the boundary when the cell across it is not kept.
    padded = np.pad(cells, 1, constant_values=False)
    kept = padded[1:-1, 1:-1]
    for across, (a, b) in [
        (padded[:-2, 1:-1], (p00, p01)),  # south edge, travelling east
        (padded[1:-1, 2:], (p01, p11)),  # east edge, travelling north
        (padded[2:, 1:-1], (p11, p10)),  # north edge, travelling west
        (padded[1:-1, :-2], (p10, p00)),  # west edge, travelling south
    ]:
        edge = ~across[kept]
        a, b = a[edge], b[edge]
        # Travelling counter-clockwise around the solid, the outside is on the right.
        faces += [np.column_stack([a + n, b + n, b]), np.column_stack([a + n, b, a])]

    return Mesh(vertices, np.concatenate(faces).astype(np.int64))

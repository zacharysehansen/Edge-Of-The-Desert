"""Turn a heightfield into a closed, printable triangle mesh (millimetres)."""

from dataclasses import dataclass

import numpy as np


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

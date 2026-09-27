"""Automatic checks that a mesh is printable."""

import numpy as np

from terrain.mesh import Mesh


def watertight_problems(mesh: Mesh) -> list[str]:
    """Empty if the mesh is closed, manifold and consistently oriented outward."""
    problems = []
    f = mesh.faces
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    unique_directed, counts = np.unique(directed, axis=0, return_counts=True)
    if (counts > 1).any():
        problems.append(f"{int((counts > 1).sum())} edges are used twice in the same direction "
                        "(non-manifold or flipped faces)")
    forward = {tuple(e) for e in unique_directed}
    unmatched = sum(1 for a, b in forward if (b, a) not in forward)
    if unmatched:
        problems.append(f"{unmatched} edges have no matching opposite edge (holes)")

    v = mesh.vertices[f]
    cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    if (np.linalg.norm(cross, axis=1) == 0).any():
        problems.append("mesh has zero-area triangles")
    if signed_volume(mesh) <= 0:
        problems.append("mesh is inside out (negative volume)")
    return problems


def signed_volume(mesh: Mesh) -> float:
    v = mesh.vertices[mesh.faces]
    return float(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum() / 6.0)

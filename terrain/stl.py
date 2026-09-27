"""Binary STL export."""

from pathlib import Path

import numpy as np

from terrain.mesh import Mesh


def write_stl(mesh: Mesh, path: Path, header: str = "") -> None:
    v = mesh.vertices[mesh.faces].astype(np.float32)
    normals = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    record = np.zeros(len(v), dtype=[("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    record["normal"] = normals
    record["v"] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii", "replace")[:80].ljust(80, b" "))
        f.write(np.uint32(len(v)).tobytes())
        f.write(record.tobytes())

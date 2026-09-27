"""Ticket 01: config -> DEM -> watertight STL, on a synthetic DEM (no network)."""

import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from terrain.checks import watertight_problems
from terrain.config import load_config
from terrain.dem import load_heightfield
from terrain.mesh import Mesh, heightfield_to_mesh
from terrain.stl import write_stl

REPO = Path(__file__).resolve().parent.parent
PATCH = dict(west=-110.840, south=32.290, east=-110.776, north=32.344)


def synthetic_dem(path: Path, low=800.0, high=1800.0) -> None:
    """A 1/3 arc-second DEM in NAD83 lon/lat that rises linearly northward from `low` to `high`."""
    res = 1 / 10800
    west, north = PATCH["west"] - 0.02, PATCH["north"] + 0.02
    n = int(0.104 / res)
    lat = north - (np.arange(n) + 0.5) * res
    frac = (lat - PATCH["south"]) / (PATCH["north"] - PATCH["south"])
    elev = np.repeat((low + np.clip(frac, 0, 1) * (high - low))[:, None], n, axis=1).astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                       crs="EPSG:4269", transform=from_origin(west, north, res, res), nodata=-999999) as dst:
        dst.write(elev, 1)


def test_mesh_is_watertight_and_sized():
    rng = np.random.default_rng(0)
    elev = 800 + rng.random((40, 60)) * 1000
    mesh = heightfield_to_mesh(elev, spacing_mm=0.5, scale=30000, vertical_exaggeration=2.5, base_thickness_mm=5)
    assert watertight_problems(mesh) == []
    size = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    assert size[0] == pytest.approx(59 * 0.5)
    assert size[1] == pytest.approx(39 * 0.5)
    expected_relief = (elev.max() - elev.min()) * 2.5 / 30000 * 1000
    assert size[2] - 5 == pytest.approx(expected_relief)


def test_check_catches_a_hole_and_a_flip():
    mesh = heightfield_to_mesh(np.zeros((5, 5)) + np.arange(5), 1.0, 1000, 1, 2)
    holed = Mesh(mesh.vertices, mesh.faces[1:])
    assert any("holes" in p for p in watertight_problems(holed))
    flipped = Mesh(mesh.vertices, mesh.faces[:, ::-1])
    assert any("inside out" in p for p in watertight_problems(flipped))


def test_stl_has_one_record_per_face(tmp_path):
    mesh = heightfield_to_mesh(np.ones((4, 4)), 1.0, 1000, 1, 2)
    path = tmp_path / "t.stl"
    write_stl(mesh, path)
    data = path.read_bytes()
    (count,) = struct.unpack("<I", data[80:84])
    assert count == len(mesh.faces)
    assert len(data) == 84 + 50 * count


def test_heightfield_reprojects_to_utm_grid(tmp_path):
    dem = tmp_path / "dem.tif"
    synthetic_dem(dem)
    hf = load_heightfield(dem, **PATCH, spacing_m=15.0)
    assert not np.isnan(hf.elevation_m).any()
    # About 6 km each way, sampled every 15 m.
    assert 5.5e3 < hf.width_m < 6.2e3 and 5.5e3 < hf.height_m < 6.2e3
    # North is row 0, so elevation should fall from the first row to the last.
    assert hf.elevation_m[0].mean() > hf.elevation_m[-1].mean()


def test_heightfield_rejects_dem_that_misses_the_patch(tmp_path):
    dem = tmp_path / "dem.tif"
    synthetic_dem(dem)
    with pytest.raises(ValueError, match="no elevation data"):
        load_heightfield(dem, west=-111.5, south=32.290, east=-111.4, north=32.344, spacing_m=15.0)


def test_cli_end_to_end(tmp_path):
    dem = tmp_path / "dem.tif"
    synthetic_dem(dem, low=800, high=1800)
    config = (REPO / "config" / "test_tile.toml").read_text()
    config = config.replace('dem_path = "data/raw/dem/catalina_front_sabino.tif"', f'dem_path = "{dem}"')
    config = config.replace('dir = "outputs/terrain"', f'dir = "{tmp_path / "out"}"')
    cfg_path = tmp_path / "tile.toml"
    cfg_path.write_text(config)

    result = subprocess.run([sys.executable, "-m", "terrain", "tile", str(cfg_path)],
                            cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "watertight" in result.stdout and "fits a 220 mm bed" in result.stdout

    cfg = load_config(cfg_path)
    assert cfg.stl_path.exists()
    vertices = np.frombuffer(cfg.stl_path.read_bytes()[84:],
                             dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])["v"].reshape(-1, 3)
    size = vertices.max(axis=0) - vertices.min(axis=0)
    hf = load_heightfield(dem, **PATCH, spacing_m=cfg.ground_spacing_m)
    # Footprint and relief both match the configured scale within 1 mm.
    assert abs(size[0] - hf.width_m * 1000 / cfg.scale) < 1.0
    assert abs(size[1] - hf.height_m * 1000 / cfg.scale) < 1.0
    expected_relief = (hf.elevation_m.max() - hf.elevation_m.min()) * cfg.vertical_exaggeration / cfg.scale * 1000
    assert abs(size[2] - cfg.base_thickness_mm - expected_relief) < 1.0


def test_cli_explains_missing_dem(tmp_path):
    config = (REPO / "config" / "test_tile.toml").read_text()
    config = config.replace("data/raw/dem/catalina_front_sabino.tif", str(tmp_path / "missing.tif"))
    cfg_path = tmp_path / "tile.toml"
    cfg_path.write_text(config)
    result = subprocess.run([sys.executable, "-m", "terrain", "tile", str(cfg_path)],
                            cwd=REPO, capture_output=True, text=True)
    assert result.returncode != 0
    assert "python -m terrain fetch" in result.stderr

"""Ticket 05: valley-facing trim, on a synthetic basin ringed by ridges (no network)."""

from dataclasses import replace

import numpy as np
import pytest
from osgeo import gdal, osr

from terrain.checks import watertight_problems
from terrain.layout import Layout, Trim
from terrain.ring import Ring
from terrain.trim import drainage_roots
from test_ring import BUILD, LAYOUT


def test_drainage_follows_the_divide():
    # Two valleys split by a north-south ridge; each drains out of its own edge.
    x = np.linspace(-1, 1, 41)
    elev = np.repeat((1000 - 400 * np.abs(x))[None, :], 21, axis=0)  # one ridge down the middle
    elev[:, 0] -= 50  # west valley's outlet
    elev[:, -1] -= 50  # east valley's outlet
    roots = drainage_roots(elev)
    rows, cols = np.divmod(roots, elev.shape[1])
    # Every edge cell is an outlet, so compare sides of the ridge, not exact outlets.
    assert (cols[:, :19] < 20).all()  # west of the ridge drains out on the west side
    assert (cols[:, 22:] > 20).all()  # east of the ridge drains out on the east side


@pytest.fixture(scope="module")
def basin_dem(tmp_path_factory):
    """A bowl draining out of its north-west corner, walled by a ridge ring; outside the
    ring the ground falls away, so the ring's crest is the divide."""
    x0, y0, x1, y1 = LAYOUT.table_utm
    res, pad = 100.0, 3000.0
    w, h = int((x1 - x0 + 2 * pad) / res), int((y1 - y0 + 2 * pad) / res)
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)  # -1..1, v down = south
    r = np.hypot(u, v)
    elev = 900 + 1200 * np.exp(-((r - 0.75) ** 2) / 0.01) - 300 * np.clip(r - 0.75, 0, None)
    # Cut a gap in the ring at the north-west so the bowl has one outlet.
    gap = (u < -0.4) & (v < -0.4)
    elev = np.where(gap, np.minimum(elev, 800 + 50 * r), elev)
    path = tmp_path_factory.mktemp("basin") / "dem.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((x0 - pad, res, 0, y1 + pad, 0, -res))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26912)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(elev)
    ds = None
    return path


def ring_with(trim: Trim, dem) -> Ring:
    return Ring(replace(LAYOUT, trim=trim), replace(BUILD, mesh_resolution_mm=4.0), dem)


def test_crest_trim_keeps_the_bowl_and_cuts_the_far_side(basin_dem):
    r = ring_with(Trim(margin_m=0.0), basin_dem)
    w, h = r.table[2], r.table[3]
    cx, cy = w / 2, h / 2
    keep = r.outline.keep(np.array([cx, 0.97 * w, 0.03 * w]), np.array([cy, 0.97 * h]))
    assert keep[0, 0]  # the middle of the bowl
    assert not keep[1, 1]  # north-east corner, beyond the ring
    before, after = r.outline.area_m2()
    assert after < 0.9 * before


def test_margin_extends_past_the_crest(basin_dem):
    tight = ring_with(Trim(margin_m=0.0), basin_dem).outline.area_m2()[1]
    loose = ring_with(Trim(margin_m=2000.0), basin_dem).outline.area_m2()[1]
    assert loose > tight


def test_side_rules(basin_dem):
    sides = {"north": "full", "east": "none", "south": "strip", "west": "crest"}
    r = ring_with(Trim(sides=sides, strip_mm=60.0), basin_dem)
    o = r.opening
    mx, my = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
    k = r.outline.keep
    assert k(np.array([mx]), np.array([r.table[3] - 1]))[0, 0]  # north: full, so the far edge is kept
    assert not k(np.array([o[2] + 5]), np.array([my]))[0, 0]  # east: none
    assert k(np.array([mx]), np.array([o[1] - 50]))[0, 0]  # south: inside the strip
    assert not k(np.array([mx]), np.array([o[1] - 70]))[0, 0]  # south: beyond the strip


def test_cut_polygon_overrides_the_crest(basin_dem):
    x0, y0, x1, y1 = LAYOUT.extent
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # Cut a patch just north of the screen, which the crest rule would keep.
    patch = ((cx - 0.02, cy + 0.08), (cx + 0.02, cy + 0.08), (cx + 0.02, cy + 0.11), (cx - 0.02, cy + 0.11))
    plain = ring_with(Trim(), basin_dem).outline.area_m2()[1]
    cut = ring_with(Trim(cut=(patch,)), basin_dem).outline.area_m2()[1]
    assert cut < plain


def test_trimmed_ring_is_watertight(basin_dem):
    r = ring_with(Trim(sides={"north": "crest", "east": "crest", "south": "strip", "west": "crest"}), basin_dem)
    assert watertight_problems(r.mesh(r.table)) == []

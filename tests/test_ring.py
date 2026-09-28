"""Ticket 04: screen opening and lip, on a synthetic tilted valley (no network)."""

from dataclasses import replace

import numpy as np
import pytest
from osgeo import gdal, osr

from terrain.checks import watertight_problems
from terrain.layout import Layout
from terrain.ring import BuildConfig, Ring

LAYOUT = Layout("t", "Test", (-111.18, 32.08, -110.50, 32.45), (-111.06, 32.15, -110.72, 32.33),
                55, (16, 9), "ew")
BUILD = BuildConfig(
    vertical_exaggeration=2.5, base_thickness_mm=5.0, mesh_resolution_mm=2.0, bed_mm=220.0,
    bezel_mm=15.0, bezel_raise_mm=0.0, glass_above_plywood_mm=0.0,
    lip_overhang_mm=3.0, lip_edge_thickness_mm=1.2, blend_mm=15.0, datum_blend_mm=200.0,
    test_corner="sw", test_arm_mm=150.0, test_depth_mm=50.0)


@pytest.fixture(scope="module")
def dem(tmp_path_factory):
    """A valley that tilts 300 m across the table, plus a mountain in the north-east."""
    x0, y0, x1, y1 = LAYOUT.table_utm
    res, pad = 60.0, 3000.0
    w, h = int((x1 - x0 + 2 * pad) / res), int((y1 - y0 + 2 * pad) / res)
    yy, xx = np.mgrid[0:h, 0:w]
    elev = 700 + 300 * xx / w + 1800 * np.exp(-((xx - 0.6 * w) ** 2 + (yy - 0.1 * h) ** 2) / (0.01 * w * w))
    path = tmp_path_factory.mktemp("dem") / "dem.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(path), w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((x0 - pad, res, 0, y1 + pad, 0, -res))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26912)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(elev)
    ds = None
    return path


def seam_nodes(ring: Ring, n=200):
    """Grid nodes exactly on the opening's edge, all four sides."""
    o = ring.opening
    xs, ys = np.linspace(o[0], o[2], n), np.linspace(o[1], o[3], n)
    return xs, ys


def test_opening_is_the_display_less_the_overhang(dem):
    r = Ring(LAYOUT, BUILD, dem)
    d, o = r.display, r.opening
    assert np.allclose([o[0] - d[0], o[1] - d[1], d[2] - o[2], d[3] - o[3]], 3.0)
    assert r.screen_outer[0] == pytest.approx(d[0] - 15.0)
    # The opening's edges are grid lines exactly, so the printed hole has the exact size.
    xs, ys = r.grid(r.table)
    assert all(np.isclose(xs, e).any() for e in (o[0], o[2]))
    assert all(np.isclose(ys, e).any() for e in (o[1], o[3]))


def test_lip_tip_sits_on_the_glass_all_round(dem):
    r = Ring(LAYOUT, BUILD, dem)
    xs, ys = seam_nodes(r)
    o = r.opening
    for sx, sy in [(xs, np.array([o[1]])), (xs, np.array([o[3]])), (np.array([o[0]]), ys), (np.array([o[2]]), ys)]:
        top, bottom = r.fields(sx, sy)
        assert np.abs(bottom - BUILD.glass_above_plywood_mm).max() < 0.2
        assert np.abs(top - (BUILD.glass_above_plywood_mm + BUILD.lip_edge_thickness_mm)).max() < 0.2


def test_terrain_level_with_the_glass_despite_the_tilted_valley(dem):
    """Without the seam datum, the 300 m tilt would put one side ~25 mm above the glass."""
    r = Ring(LAYOUT, BUILD, dem)
    o = r.opening
    xs = np.linspace(o[0], o[0] + 400, 50)  # west half of the south side, away from the mountain
    top, _ = r.fields(xs, np.array([o[1] - BUILD.blend_mm]))
    assert np.abs(top - (BUILD.glass_above_plywood_mm + BUILD.base_thickness_mm)).max() < 1.0


def test_no_step_between_lip_and_terrain(dem):
    r = Ring(LAYOUT, BUILD, dem)
    o = r.opening
    ys = np.arange(o[1] - 250, o[1] + 0.01, 0.25)
    top, _ = r.fields(np.array([(o[0] + o[2]) / 2 - 300]), ys)
    slope = np.abs(np.diff(top[:, 0])) / 0.25
    assert slope.max() < 1.0  # under 45 degrees anywhere on the way out from the lip


@pytest.mark.parametrize("glass, raise_", [(0.0, 0.0), (4.0, 0.0), (0.0, 6.0), (-3.0, 2.0)])
def test_ring_is_watertight_and_thick_enough(dem, glass, raise_):
    build = replace(BUILD, glass_above_plywood_mm=glass, bezel_raise_mm=raise_, mesh_resolution_mm=4.0)
    r = Ring(LAYOUT, build, dem)
    mesh = r.mesh(r.table)
    assert watertight_problems(mesh) == []
    xs, ys = r.grid(r.table)
    top, bottom = r.fields(xs, ys)
    keep = r.cells(xs, ys, r.table)
    used = np.zeros_like(top, bool)
    used[:-1, :-1] |= keep
    used[1:, 1:] |= keep
    assert (top - bottom)[used].min() >= build.lip_edge_thickness_mm - 1e-6
    # The underside over the display area rests on the glass.
    d = r.display
    inside = ((xs > d[0]) & (xs < d[2]))[None, :] & ((ys > d[1]) & (ys < d[3]))[:, None]
    assert np.allclose(bottom[inside & used], glass)


def test_lip_test_piece_is_an_l_that_fits_the_bed(dem):
    r = Ring(LAYOUT, BUILD, dem)
    piece = r.test_piece()
    assert watertight_problems(piece) == []
    used = piece.vertices[np.unique(piece.faces)]
    size = used.max(axis=0) - used.min(axis=0)
    assert size[0] <= BUILD.bed_mm and size[1] <= BUILD.bed_mm
    # Nothing printed inside the opening.
    o = r.opening
    inside = (used[:, 0] > o[0] + 1e-6) & (used[:, 1] > o[1] + 1e-6)
    assert not inside.any()

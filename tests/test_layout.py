"""Ticket 03: layout maths and preview rendering (synthetic DEM, no network)."""

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal, osr

from terrain.layout import Layout, load_layout
from terrain.preview import render_preview

REPO = Path(__file__).resolve().parent.parent


def layout(orientation="ew", valley=(-111.06, 32.15, -110.72, 32.33),
           extent=(-111.18, 32.08, -110.50, 32.45)) -> Layout:
    return Layout("t", "Test", extent, valley, 55, (16, 9), orientation)


def test_screen_size_follows_diagonal_and_orientation():
    w, h = layout("ew").screen_mm
    assert np.hypot(w, h) == pytest.approx(55 * 25.4)
    assert w / h == pytest.approx(16 / 9)
    assert layout("ns").screen_mm == pytest.approx((h, w))


def test_valley_fits_the_screen_and_fills_one_side():
    lay = layout()
    vx0, vy0, vx1, vy1 = lay.valley_utm
    sx0, sy0, sx1, sy1 = lay.screen_utm
    assert sx0 <= vx0 + 1e-6 and sy0 <= vy0 + 1e-6 and sx1 >= vx1 - 1e-6 and sy1 >= vy1 - 1e-6
    # The limiting side is filled exactly.
    assert min(abs((sx1 - sx0) - (vx1 - vx0)), abs((sy1 - sy0) - (vy1 - vy0))) < 1e-6
    # Screen ground size / physical size = scale.
    assert (sx1 - sx0) * 1000 / lay.screen_mm[0] == pytest.approx(lay.scale)


def test_table_footprint_at_scale():
    lay = layout()
    tx0, ty0, tx1, ty1 = lay.table_utm
    assert lay.table_mm[0] == pytest.approx((tx1 - tx0) * 1000 / lay.scale)
    assert lay.problems() == []


def test_screen_past_the_table_is_a_problem():
    lay = layout(extent=(-111.0, 32.2, -110.8, 32.3))
    assert lay.problems()


@pytest.mark.parametrize("path", sorted((REPO / "config" / "layouts").glob("*.toml")))
def test_shipped_layouts_are_valid(path):
    lay = load_layout(path)
    assert lay.problems() == []
    assert 10_000 < lay.scale < 100_000


def test_preview_renders_from_a_dem(tmp_path):
    lay = layout()
    x0, y0, x1, y1 = lay.table_utm
    res, pad = 100.0, 6000.0
    w, h = int((x1 - x0 + 2 * pad) / res), int((y1 - y0 + 2 * pad) / res)
    dem = tmp_path / "dem.tif"
    ds = gdal.GetDriverByName("GTiff").Create(str(dem), w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((x0 - pad, res, 0, y1 + pad, 0, -res))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26912)
    ds.SetProjection(srs.ExportToWkt())
    yy, xx = np.mgrid[0:h, 0:w]
    ds.GetRasterBand(1).WriteArray(700 + 2000 * np.exp(-((xx - w / 2) ** 2 + (yy - h / 4) ** 2) / 500))
    ds = None
    out = tmp_path / "p.png"
    render_preview(lay, dem, None, out)
    assert out.exists() and out.stat().st_size > 10_000

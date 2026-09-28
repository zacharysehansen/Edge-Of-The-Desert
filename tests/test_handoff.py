"""Ticket 07: screen-area handoff and seam colours, on synthetic DEM and imagery (no network)."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import rasterio
from osgeo import gdal, osr

from terrain.handoff import (HandoffSettings, perimeter_points, sample_seam_colors, screen_pixels,
                             srgb_to_lab, window_mean, write_handoff)
from terrain.ring import Ring
from test_ring import BUILD, LAYOUT, dem  # noqa: F401  (dem is a fixture)

SETTINGS = HandoffSettings(screen_px=(1920, 1080), spacing_mm=100.0, window_mm=4.0,
                           lower_slope_mm=100.0, fixed_band_mm=25.0, feather_mm=20.0)
RES = 20.0  # synthetic imagery pixel size, metres


def _raster(path, x0, y1, bands, dtype):
    ds = gdal.GetDriverByName("GTiff").Create(str(path), bands[0].shape[1], bands[0].shape[0], len(bands), dtype)
    ds.SetGeoTransform((x0, RES, 0, y1, 0, -RES))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(26912)
    ds.SetProjection(srs.ExportToWkt())
    for i, b in enumerate(bands, 1):
        ds.GetRasterBand(i).WriteArray(b)
    ds = None


@pytest.fixture(scope="module")
def imagery(tmp_path_factory):
    """Red rises west to east and green south to north, so any pixel says where it came from."""
    x0, y0, x1, y1 = LAYOUT.table_utm
    pad = 3000.0
    w, h = int((x1 - x0 + 2 * pad) / RES), int((y1 - y0 + 2 * pad) / RES)
    yy, xx = np.mgrid[0:h, 0:w]
    r = (20 + 215 * xx / (w - 1)).astype(np.uint8)
    g = (235 - 215 * yy / (h - 1)).astype(np.uint8)
    b = np.full((h, w), 90, np.uint8)
    path = tmp_path_factory.mktemp("img") / "img.tif"
    _raster(path, x0 - pad, y1 + pad, [r, g, b], gdal.GDT_Byte)
    return path, (x0 - pad, y1 + pad, w, h)


@pytest.fixture(scope="module")
def ring(dem):  # noqa: F811
    return Ring(LAYOUT, replace(BUILD, mesh_resolution_mm=4.0), dem)


def test_lab_matches_published_values():
    assert np.allclose(srgb_to_lab(np.array([255, 255, 255])), [100, 0, 0], atol=0.01)
    assert np.allclose(srgb_to_lab(np.array([0, 0, 0])), [0, 0, 0], atol=0.01)
    assert np.allclose(srgb_to_lab(np.array([255, 0, 0])), [53.24, 80.09, 67.20], atol=0.05)
    assert np.allclose(srgb_to_lab(np.array([119, 119, 119])), [50.03, 0, 0], atol=0.05)


def test_window_mean_ignores_a_single_bright_pixel():
    block = np.full((21, 21, 3), [150, 120, 90], np.uint8)
    block[10, 10] = [255, 255, 255]
    assert np.abs(window_mean(block) - [150, 120, 90]).max() <= 2


def test_perimeter_covers_all_four_sides_at_the_spacing():
    r = (100.0, 50.0, 1318.0, 735.0)
    pts = perimeter_points(r, 100.0)
    assert {p[0] for p in pts} == {"north", "east", "south", "west"}
    xy = np.array([p[1:] for p in pts])
    on_edge = np.isclose(xy[:, 0], r[0]) | np.isclose(xy[:, 0], r[2]) | np.isclose(xy[:, 1], r[1]) | np.isclose(xy[:, 1], r[3])
    assert on_edge.all()
    gaps = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
    assert gaps.max() <= 100.0 + 1e-9


def test_screen_resolution_must_match_the_display_aspect(ring):
    assert screen_pixels(ring, SETTINGS) == (1920, 1080)
    with pytest.raises(ValueError, match="not 16:9"):
        screen_pixels(ring, replace(SETTINGS, screen_px=(1920, 1200)))
    ns = Ring(replace(LAYOUT, orientation="ns"), ring.build, ring.dem_path, trimmed=False)
    assert screen_pixels(ns, SETTINGS) == (1080, 1920)


@pytest.fixture(scope="module")
def handoff(ring, dem, imagery, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("handoff")
    info, swatches = write_handoff(ring, SETTINGS, dem, imagery[0], out)
    return out, info, swatches


def test_screen_image_covers_exactly_the_display_area(handoff):
    out, info, _ = handoff
    sx0, sy0, sx1, sy1 = LAYOUT.screen_utm
    for name in ("screen_imagery.tif", "screen_dem.tif"):
        with rasterio.open(out / name) as src:
            assert (src.width, src.height) == (1920, 1080)
            assert np.allclose(src.bounds, (sx0, sy0, sx1, sy1), atol=1e-3)
    assert (out / "screen_imagery.png").stat().st_size > 0
    saved = json.loads((out / "screen.json").read_text())
    assert saved["pixels"] == [1920, 1080]
    assert np.allclose(list(saved["bounds_utm"].values()), (sx0, sy0, sx1, sy1), atol=0.01)
    assert set(saved["corners_lonlat_nad83"]) == {"nw", "ne", "se", "sw"}
    # Display 1218 mm over 1920 px; 3 mm of lip hides about 4.7 px.
    assert info["hidden_under_lip_px"] == pytest.approx(3.0 / (LAYOUT.screen_mm[0] / 1920), abs=0.1)


def test_screen_image_pixels_come_from_the_right_ground(handoff, imagery):
    """Each corner pixel carries the colour of the ground under its centre, within one source pixel."""
    out, _, _ = handoff
    _, (ix0, iy1, w, h) = imagery
    with rasterio.open(out / "screen_imagery.tif") as src:
        img = src.read()
        for row, col in [(0, 0), (0, src.width - 1), (src.height - 1, 0), (src.height - 1, src.width - 1)]:
            e, n = src.xy(row, col)
            want_r = 20 + 215 * ((e - ix0) / RES - 0.5) / (w - 1)
            want_g = 235 - 215 * ((iy1 - n) / RES - 0.5) / (h - 1)
            step = 215 / (w - 1) + 1  # one source pixel of gradient, plus rounding
            assert abs(img[0, row, col] - want_r) <= step
            assert abs(img[1, row, col] - want_g) <= step


def test_seam_colours_sampled_all_round_and_located(handoff, imagery, ring):
    out, _, swatches = handoff
    seam = [s for s in swatches if s.row == "seam"]
    assert {s.side for s in seam} == {"north", "east", "south", "west"}
    o = ring.opening
    assert len(seam) >= 2 * ((o[2] - o[0]) + (o[3] - o[1])) / SETTINGS.spacing_mm
    # Lower-slope samples are only where terrain is printed.
    for s in swatches:
        if s.row != "seam":
            assert ring.outline.keep(np.array([s.x_mm]), np.array([s.y_mm]))[0, 0]
    # A swatch's colour is the imagery at its own ground position.
    _, (ix0, iy1, w, h) = imagery
    for s in seam[::7]:
        assert abs(s.rgb[0] - (20 + 215 * ((s.easting - ix0) / RES - 0.5) / (w - 1))) <= 2
        assert abs(s.rgb[1] - (235 - 215 * ((iy1 - s.northing) / RES - 0.5) / (h - 1))) <= 2
    rows = list(csv.DictReader(open(out / "seam_colors.csv")))
    assert len(rows) == len(swatches)
    assert rows[0]["hex"] == swatches[0].hex and rows[0]["L_d65"]


def test_swatch_sheet_and_note_are_written(handoff):
    out, _, swatches = handoff
    pdf = (out / "seam_colors.pdf").read_bytes()
    assert pdf.startswith(b"%PDF") and len(pdf) > 10_000
    note = (out / "HANDOFF.md").read_text()
    for must in ("Fixed band", "Feathering", "Edge-pixel calibration", f"{len(swatches)} swatches"):
        assert must in note

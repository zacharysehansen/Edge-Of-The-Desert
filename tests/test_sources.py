"""Ticket 02: superset data assembly, on synthetic local files (no network)."""

import numpy as np
import pytest
from osgeo import gdal, osr

from terrain import sources
from terrain.sources import SourcesConfig, dem_tile_names, latest_year_items

BOX = dict(west=-110.84, south=32.29, east=-110.776, north=32.344)


def config(tmp_path, tile_url) -> SourcesConfig:
    return SourcesConfig(
        **BOX, dem_tile_url=tile_url, dem_tiles_dir=tmp_path / "tiles", dem_path=tmp_path / "dem.tif",
        dem_resolution_m=10.0, stac_url="", token_url="", imagery_items_dir=tmp_path / "items",
        imagery_path=tmp_path / "naip.tif", imagery_resolution_m=5.0)


def write_raster(path, data, west, north, res, epsg, nodata=None, alpha=False):
    bands = data.shape[0]
    dtype = gdal.GDT_Byte if data.dtype == np.uint8 else gdal.GDT_Float32
    ds = gdal.GetDriverByName("GTiff").Create(str(path), data.shape[2], data.shape[1], bands, dtype)
    ds.SetGeoTransform((west, res, 0, north, 0, -res))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    for b in range(bands):
        ds.GetRasterBand(b + 1).WriteArray(data[b])
        if nodata is not None:
            ds.GetRasterBand(b + 1).SetNoDataValue(nodata)
    if alpha:
        ds.GetRasterBand(bands).SetColorInterpretation(gdal.GCI_AlphaBand)
    ds = None


def test_tile_names_cover_the_superset():
    names = dem_tile_names(-111.25, 31.62, -110.45, 32.52)
    assert sorted(names) == ["n32w111", "n32w112", "n33w111", "n33w112"]
    assert dem_tile_names(**BOX) == ["n33w111"]


def test_latest_year_only():
    feats = [{"id": f"a{y}", "properties": {"naip:year": str(y), "datetime": f"{y}-06-01"}}
             for y in (2019, 2021, 2023, 2023)]
    year, items = latest_year_items(feats)
    assert year == 2023 and len(items) == 2
    with pytest.raises(ValueError):
        latest_year_items([])


def test_dem_fetch_merge_verify_and_resume(tmp_path):
    # A stand-in for the remote 3DEP tile n33w111, in NAD83 lon/lat.
    res = 1 / 10800
    rows, cols = int(0.15 / res), int(0.2 / res)  # 32.25-32.40 N, 110.9-110.7 W covers the box
    elev = np.linspace(800, 1600, rows * cols, dtype="float32").reshape(1, rows, cols)
    tile = tmp_path / "USGS_13_n33w111.tif"
    write_raster(tile, elev, -110.9, 32.4, res, 4269, nodata=-999999)
    cfg = config(tmp_path, str(tmp_path / "USGS_13_{name}.tif"))

    logs = []
    tiles = sources.fetch_dem_tiles(cfg, log=logs.append)
    assert [p.name for p in tiles] == ["USGS_13_n33w111_crop.tif"] and tiles[0].exists()
    assert not list(cfg.dem_tiles_dir.glob("*.part"))
    sources.fetch_dem_tiles(cfg, log=logs.append)
    assert "already on disk" in logs[-1]

    sources.merge_dem(cfg, tiles)
    ds = gdal.Open(str(cfg.dem_path))
    assert "UTM zone 12N" in ds.GetProjection() and ds.GetGeoTransform()[1] == 10.0
    ds = None
    # verify() raises on gaps, so the image being in the report means full coverage
    # (and the missing imagery is the only gap).
    with pytest.raises(ValueError) as err:
        sources.verify(cfg)
    assert "DEM:" in str(err.value) and "0 empty pixels" in str(err.value)
    assert "Imagery:" in str(err.value) and "is missing" in str(err.value)


def test_imagery_mosaic_respects_alpha_and_reports_holes(tmp_path):
    cfg = config(tmp_path, "")
    xmin, ymin, xmax, ymax = sources.utm_crop_box(*cfg.bbox)
    midx = (xmin + xmax) / 2
    h = int((ymax - ymin) / 5) + 40
    w = int((midx - xmin) / 5) + 40
    # Two overlapping pieces; each has a transparent strip where the other has data.
    for i, west in enumerate([xmin - 100, midx - 100]):
        rgba = np.full((4, h, w), 60 + 100 * i, dtype=np.uint8)
        rgba[3] = 255
        if i == 1:
            rgba[:, :, :30] = 0  # transparent left edge overlapping piece 0
        write_raster(tmp_path / f"item{i}.tif", rgba, west, ymax + 100, 5.0, 26912, alpha=True)
    items = [tmp_path / "item0.tif", tmp_path / "item1.tif"]
    sources.merge_imagery(cfg, items)
    ds = gdal.Open(str(cfg.imagery_path))
    assert ds.RasterCount == 3
    rgb = np.stack([ds.GetRasterBand(b).ReadAsArray() for b in (1, 2, 3)])
    ds = None
    assert (rgb.max(axis=0) > 0).all()  # piece 1's transparent edge did not blank piece 0

    # Drop the second piece: the east half is now empty and verify must say so.
    sources.merge_imagery(cfg, items[:1])
    with pytest.raises(ValueError, match="Imagery: .* empty pixels inside the box"):
        sources.verify(cfg)


def test_token_is_renewed_when_old_or_rejected(monkeypatch):
    issued = iter(["t1", "t2", "t3"])
    monkeypatch.setattr(sources, "_get_json", lambda url: {"token": next(issued)})
    clock = [0.0]
    monkeypatch.setattr(sources.time, "monotonic", lambda: clock[0])
    cache = sources._TokenCache("x")
    assert cache.get() == "t1"
    clock[0] = 10 * 60
    assert cache.get() == "t1"
    clock[0] = 45 * 60
    assert cache.get() == "t2"
    assert cache.get(fresh=True) == "t3"


def test_write_that_produces_nothing_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="no data was written"):
        sources._write_atomic(tmp_path / "x.tif", lambda part: None)
    assert not (tmp_path / "x.tif").exists()

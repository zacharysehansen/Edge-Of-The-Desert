"""
local_region.py
---------------
Single source of truth for the Tucson local-tier domain, nested inside
the eight-county regional boundary defined by region.py.

The local tier is built from four HUC8 watersheds (Upper Santa Cruz,
Rillito, Lower Santa Cruz, Brawley Wash) clipped to Pima County.  The
clip removes Nogales / Mexico to the south and Pinal agriculture to the
north, keeping the domain tightly around the Tucson basin where
land-cover changes (urbanization, farming) are ~12x more visible than
at the regional scale.

Exposes:
    LOCAL_COUNTIES      - ["Pima"]
    LOCAL_COUNTY_FIPS   - ["04019"]
    LOCAL_HUC8_CODES    - the four HUC8 codes
    LOCAL_BBOX          - (min_lon, min_lat, max_lon, max_lat) tuple
    LOCAL_BOUNDARY      - dissolved GeoDataFrame of the local domain
    load_local_boundary()   - returns the dissolved boundary GeoDataFrame
    filter_points_local()   - filters points to the local boundary
    clip_raster_local()     - clips a rasterio dataset to the local boundary
"""

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
from rasterio.mask import mask as rio_mask
from shapely.geometry import box


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCAL_COUNTIES: list[str] = ["Pima"]
LOCAL_COUNTY_FIPS: list[str] = ["04019"]

LOCAL_HUC8_CODES: list[str] = [
    "15050301",  # Upper Santa Cruz
    "15050302",  # Rillito
    "15050303",  # Lower Santa Cruz
    "15050304",  # Brawley Wash
]

LOCAL_HUC8_NAMES: list[str] = [
    "Upper Santa Cruz",
    "Rillito",
    "Lower Santa Cruz",
    "Brawley Wash",
]

WGS84_CRS: str = "EPSG:4326"
WGS84_EPSG: int = 4326
UTM12N_CRS: str = "EPSG:32612"  # for area calculations in southern AZ

# Default paths assume the standard data/raw/ layout
_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_WBD_GDB = (
    _REPO_ROOT / "data" / "raw" / "wbd" / "WBD_15_HU2_GDB.gdb"
)

_DEFAULT_TIGER_SHAPEFILE = (
    _REPO_ROOT / "data" / "raw" / "tiger" / "tl_2023_us_county.shp"
)


# ---------------------------------------------------------------------------
# Boundary loader
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_local_boundary(
    wbd_path: str | Path | None = None,
    tiger_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """
    Build the Tucson local-tier boundary: four HUC8 watersheds dissolved
    and clipped to Pima County.

    The result is cached after the first call so the GDB and shapefile
    are only read once per process.

    Parameters
    ----------
    wbd_path : str or Path, optional
        Path to the WBD HU2 GDB (region 15).
        Defaults to data/raw/wbd/WBD_15_HU2_GDB.gdb.
    tiger_path : str or Path, optional
        Path to the TIGER/Line county shapefile.
        Defaults to data/raw/tiger/tl_2023_us_county.shp.

    Returns
    -------
    GeoDataFrame
        Single-row GeoDataFrame with the dissolved and clipped local
        boundary, CRS = EPSG:4326.

    Raises
    ------
    FileNotFoundError
        If the WBD GDB or TIGER shapefile is missing.
    ValueError
        If the expected HUC8 codes or Pima County FIPS are not found.
    """
    gdb = Path(wbd_path) if wbd_path else _DEFAULT_WBD_GDB
    shp = Path(tiger_path) if tiger_path else _DEFAULT_TIGER_SHAPEFILE

    if not gdb.exists():
        raise FileNotFoundError(
            f"WBD GDB not found at {gdb}.\n"
            "Download WBD_15_HU2_GDB from "
            "https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/WBD/HU2/GDB/ "
            "and extract it to data/raw/wbd/"
        )

    if not shp.exists():
        raise FileNotFoundError(
            f"TIGER county shapefile not found at {shp}.\n"
            "Download from https://www.census.gov/cgi-bin/geo/shapefiles/"
            "index.php?year=2023&layergroup=Counties+%28and+equivalent%29"
        )

    # --- Load the four HUC8 watersheds ---
    wbd_hu8 = gpd.read_file(gdb, layer="WBDHU8")
    selected = wbd_hu8[wbd_hu8["huc8"].isin(LOCAL_HUC8_CODES)].copy()

    if len(selected) != len(LOCAL_HUC8_CODES):
        matched = set(selected["huc8"].tolist())
        missing = set(LOCAL_HUC8_CODES) - matched
        raise ValueError(
            f"Expected {len(LOCAL_HUC8_CODES)} HUC8 watersheds but matched "
            f"{len(selected)}. Missing codes: {missing}."
        )

    # Verify names match expectations
    name_of = dict(zip(selected["huc8"], selected["name"], strict=False))
    for code, expected_name in zip(LOCAL_HUC8_CODES, LOCAL_HUC8_NAMES, strict=True):
        actual = name_of[code]
        if actual != expected_name:
            raise ValueError(
                f"HUC8 {code}: expected {expected_name!r}, got {actual!r}. "
                "Check that LOCAL_HUC8_CODES and LOCAL_HUC8_NAMES agree."
            )

    # Dissolve the four basins into one polygon
    huc8_dissolved = selected.dissolve().reset_index(drop=True)

    # --- Load Pima County ---
    counties_all = gpd.read_file(shp)
    pima = counties_all[counties_all["GEOID"] == LOCAL_COUNTY_FIPS[0]].copy()

    if pima.empty:
        raise ValueError(
            f"Pima County (FIPS {LOCAL_COUNTY_FIPS[0]}) not found in the "
            "TIGER shapefile. Check that the GEOID column and FIPS code match."
        )

    # Align CRS (WBD is typically NAD83 = EPSG:4269)
    if pima.crs != huc8_dissolved.crs:
        pima = pima.to_crs(huc8_dissolved.crs)

    # --- Clip HUC8 to Pima County ---
    clipped = gpd.clip(huc8_dissolved, pima)

    # Dissolve again in case the clip produced multiple pieces
    clipped = clipped.dissolve().reset_index(drop=True)

    # Reproject to WGS84
    if clipped.crs is None:
        clipped = clipped.set_crs(WGS84_CRS)
    elif clipped.crs.to_epsg() != WGS84_EPSG:
        clipped = clipped.to_crs(WGS84_CRS)

    return clipped


def _compute_bbox(boundary: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Extract (min_lon, min_lat, max_lon, max_lat) from a GeoDataFrame."""
    min_lon, min_lat, max_lon, max_lat = boundary.total_bounds
    return (float(min_lon), float(min_lat), float(max_lon), float(max_lat))


# ---------------------------------------------------------------------------
# Module-level singletons (lazy via __getattr__)
# ---------------------------------------------------------------------------
#
# LOCAL_BOUNDARY and LOCAL_BBOX are resolved on first access using the
# module-level __getattr__ hook (PEP 562).  This avoids reading the WBD
# GDB and TIGER shapefile at import time, which would break CI and any
# script that only needs the constants.

_LOCAL_BOUNDARY: gpd.GeoDataFrame | None = None
_LOCAL_BBOX: tuple[float, float, float, float] | None = None


def __getattr__(name: str):
    global _LOCAL_BOUNDARY, _LOCAL_BBOX

    if name == "LOCAL_BOUNDARY":
        if _LOCAL_BOUNDARY is None:
            _LOCAL_BOUNDARY = load_local_boundary()
        return _LOCAL_BOUNDARY

    if name == "LOCAL_BBOX":
        if _LOCAL_BBOX is None:
            if _LOCAL_BOUNDARY is None:
                _LOCAL_BOUNDARY = load_local_boundary()
            _LOCAL_BBOX = _compute_bbox(_LOCAL_BOUNDARY)
        return _LOCAL_BBOX

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Spatial utilities (mirror region.py's interface)
# ---------------------------------------------------------------------------

def filter_points_local(
    gdf: gpd.GeoDataFrame,
    wbd_path: str | Path | None = None,
    tiger_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """
    Filter a GeoDataFrame of point geometries to only those that fall
    inside the local Tucson-basin boundary.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input points. Must have a geometry column. CRS will be
        reprojected to EPSG:4326 automatically if it differs.
    wbd_path, tiger_path : str or Path, optional
        Passed through to load_local_boundary().

    Returns
    -------
    GeoDataFrame
        Subset containing only points inside the local boundary.

    Raises
    ------
    ValueError
        If the GeoDataFrame has no geometry or is empty.
    """
    if gdf.geometry is None or gdf.empty:
        raise ValueError("Input GeoDataFrame has no geometry or is empty.")

    original_crs = gdf.crs
    boundary = load_local_boundary(wbd_path, tiger_path)

    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84_CRS)
    elif gdf.crs.to_epsg() != WGS84_EPSG:
        gdf = gdf.to_crs(WGS84_CRS)

    filtered = gpd.sjoin(
        gdf, boundary[["geometry"]], how="inner", predicate="within"
    )
    filtered = filtered.drop(columns=["index_right"], errors="ignore")

    if original_crs is not None and original_crs.to_epsg() != WGS84_EPSG:
        filtered = filtered.to_crs(original_crs)

    return filtered.reset_index(drop=True)


def clip_raster_local(
    raster_dataset,  # noqa: ANN001
    wbd_path: str | Path | None = None,
    tiger_path: str | Path | None = None,
    all_touched: bool = False,
) -> tuple:
    """
    Clip an open rasterio dataset to the local Tucson-basin boundary.

    Parameters
    ----------
    raster_dataset : rasterio.io.DatasetReader
        An already-opened rasterio dataset. The caller is responsible
        for opening and closing the file.
    wbd_path, tiger_path : str or Path, optional
        Passed through to load_local_boundary().
    all_touched : bool, optional
        Passed to rasterio.mask.mask(). If True, pixels touched by the
        boundary are included. Default False.

    Returns
    -------
    clipped_array : numpy.ndarray
        The clipped raster array, shape (bands, rows, cols).
    clipped_transform : affine.Affine
        The affine transform for the clipped array.
    clipped_meta : dict
        Updated rasterio metadata reflecting the clipped dimensions,
        suitable for passing directly to rasterio.open() in write mode.

    Raises
    ------
    ImportError
        If rasterio is not installed.
    ValueError
        If the raster CRS cannot be determined.
    """
    if importlib.util.find_spec("rasterio") is None:
        raise ImportError(
            "rasterio is required for clip_raster_local(). "
            "Install it with: pip install rasterio"
        )

    boundary = load_local_boundary(wbd_path, tiger_path)

    raster_crs = raster_dataset.crs
    if raster_crs is None:
        raise ValueError(
            "The raster dataset has no CRS. Cannot reproject boundary to match."
        )

    boundary_reprojected = boundary.to_crs(raster_crs.to_epsg())
    shapes = [geom.__geo_interface__ for geom in boundary_reprojected.geometry]

    clipped_array, clipped_transform = rio_mask(
        raster_dataset,
        shapes,
        crop=True,
        all_touched=all_touched,
    )

    clipped_meta = raster_dataset.meta.copy()
    clipped_meta.update(
        {
            "driver": "GTiff",
            "height": clipped_array.shape[1],
            "width": clipped_array.shape[2],
            "transform": clipped_transform,
        }
    )

    return clipped_array, clipped_transform, clipped_meta


def bbox_geometry_local() -> box:
    """
    Return the local bounding box as a shapely Polygon in EPSG:4326.
    Useful for quick pre-filtering large datasets before the more
    expensive point-in-polygon check.
    """
    global _LOCAL_BBOX, _LOCAL_BOUNDARY
    if _LOCAL_BBOX is None:
        if _LOCAL_BOUNDARY is None:
            _LOCAL_BOUNDARY = load_local_boundary()
        _LOCAL_BBOX = _compute_bbox(_LOCAL_BOUNDARY)
    min_lon, min_lat, max_lon, max_lat = _LOCAL_BBOX
    return box(min_lon, min_lat, max_lon, max_lat)


# ---------------------------------------------------------------------------
# Verification (python scripts/phase1/local_region.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Add the scripts/ directory to sys.path so we can import region.py
    # the same way other phase1 scripts do.
    _scripts_dir = str(Path(__file__).resolve().parents[1])
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)

    print("Tucson local-tier domain")
    print(f"  Counties : {', '.join(LOCAL_COUNTIES)}")
    print(f"  FIPS     : {', '.join(LOCAL_COUNTY_FIPS)}")
    print(f"  HUC8     : {', '.join(LOCAL_HUC8_CODES)}")
    print(f"  Basins   : {', '.join(LOCAL_HUC8_NAMES)}")
    print()

    try:
        boundary = load_local_boundary()
        print("Local boundary loaded successfully.")
        print(f"  CRS      : {boundary.crs}")

        local_bbox = _compute_bbox(boundary)
        print(f"  BBox     : {local_bbox}")
        print(f"  Bounds   : {boundary.total_bounds}")
        print()

        # Area in projected CRS
        proj = boundary.to_crs(UTM12N_CRS)
        area_km2 = proj.geometry.area.sum() / 1e6
        area_mi2 = area_km2 * 0.386102
        print(f"  Area     : {area_km2:,.1f} km2  /  {area_mi2:,.1f} mi2")

        # Percentage of the eight-county region
        try:
            from phase1.region import load_county_boundary

            regional = load_county_boundary()
            regional_proj = regional.to_crs(UTM12N_CRS)
            regional_km2 = regional_proj.geometry.area.sum() / 1e6
            pct = (area_km2 / regional_km2) * 100
            print(f"  Regional : {regional_km2:,.1f} km2")
            print(f"  Local/Reg: {pct:.2f}%")
        except Exception as e:
            print(f"  (Could not compute regional percentage: {e})")

        # Sanity check against target
        target_km2 = 9124.0
        deviation = abs(area_km2 - target_km2) / target_km2 * 100
        status = "PASS" if deviation < 5.0 else "FAIL"
        print()
        print(f"  Target   : {target_km2:,.1f} km2")
        print(f"  Deviation: {deviation:.2f}%")
        print(f"  Status   : {status}")

    except FileNotFoundError as e:
        print(f"Data not found (expected during CI): {e}")
        sys.exit(1)

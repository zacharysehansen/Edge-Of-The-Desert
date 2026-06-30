"""
region.py
---------
Single source of truth for the eight-county southern Arizona study area.

Every other script in phase1/ imports from here instead of repeating
boundary logic. Exposes:
    COUNTIES            - list of county names
    COUNTY_FIPS         - list of five-digit FIPS strings
    BBOX                - (min_lon, min_lat, max_lon, max_lat) tuple
    load_county_boundary() - returns a single-row GeoDataFrame of the
                             dissolved eight-county boundary
    filter_points()     - filters a GeoDataFrame of points to those
                          inside the boundary
    clip_raster()       - clips an open rasterio dataset to the boundary
                          and returns the clipped array + transform
"""

import importlib.util
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
from rasterio.mask import mask as rio_mask
from shapely.geometry import box


STATE_FIPS: str = "04"
STATE_ABBR: str = "AZ"

COUNTIES: list[str] = [
    "Pima",
    "Pinal",
    "Santa Cruz",
    "Cochise",
    "Graham",
    "Greenlee",
    "Yuma",
    "La Paz",
]

COUNTY_FIPS: list[str] = [
    "04019",  # Pima
    "04021",  # Pinal
    "04023",  # Santa Cruz
    "04003",  # Cochise
    "04013",  # Graham
    "04011",  # Greenlee
    "04027",  # Yuma
    "04007",  # La Paz
]

# (min_lon, min_lat, max_lon, max_lat) — from bbox block in config [3]
BBOX: tuple[float, float, float, float] = (-114.81, 31.33, -109.05, 34.5)

WGS84_CRS: str = "EPSG:4326"
WGS84_EPSG: int = 4326

# Shapefile is sourced from Census TIGER/Line [1].
# Default path assumes the file has been downloaded into data/raw/tiger/
_DEFAULT_SHAPEFILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "tiger"
    / "tl_2023_us_county.shp"
)


@lru_cache(maxsize=1)
def load_county_boundary(shapefile_path: str | Path | None = None) -> gpd.GeoDataFrame:
    """
    Load the Census TIGER/Line county shapefile, filter to the eight study
    counties by FIPS code, dissolve into a single boundary polygon, and
    return a one-row GeoDataFrame in EPSG:4326.

    The result is cached after the first call so the shapefile is only
    read from disk once per process.

    Parameters
    ----------
    shapefile_path : str or Path, optional
        Path to the TIGER/Line county shapefile. Defaults to
        data/raw/tiger/tl_2023_us_county.shp relative to the repo root.

    Returns
    -------
    GeoDataFrame
        Single-row GeoDataFrame with the dissolved eight-county polygon,
        CRS = EPSG:4326.

    Raises
    ------
    FileNotFoundError
        If the shapefile does not exist at the given or default path.
    ValueError
        If fewer than eight counties are matched — indicates a FIPS or
        shapefile mismatch that would silently corrupt every downstream
        spatial filter.
    """
    path = Path(shapefile_path) if shapefile_path else _DEFAULT_SHAPEFILE

    if not path.exists():
        raise FileNotFoundError(
            f"County shapefile not found at {path}.\n"
            "Download the Census TIGER/Line county shapefile and place it at "
            "that path, or pass the correct path to load_county_boundary().\n"
            "Download: https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2023&layergroup=Counties+%28and+equivalent%29"
        )

    counties_all = gpd.read_file(path)

    # TIGER/Line uses GEOID for the five-digit FIPS code
    if "GEOID" not in counties_all.columns:
        raise ValueError(
            "Expected a 'GEOID' column in the shapefile but did not find one. "
            "Confirm this is a Census TIGER/Line county file."
        )

    study_counties = counties_all[counties_all["GEOID"].isin(COUNTY_FIPS)].copy()

    if len(study_counties) != len(COUNTY_FIPS):
        matched = set(study_counties["GEOID"].tolist())
        missing = set(COUNTY_FIPS) - matched
        raise ValueError(
            f"Expected {len(COUNTY_FIPS)} counties but matched only "
            f"{len(study_counties)}. Missing FIPS: {missing}. "
            "Check that COUNTY_FIPS values match the GEOID column in the shapefile."
        )

    # Reproject to WGS84 if needed
    if study_counties.crs is None:
        study_counties = study_counties.set_crs(WGS84_CRS)
    elif study_counties.crs.to_epsg() != WGS84_EPSG:
        study_counties = study_counties.to_crs(WGS84_CRS)

    # Dissolve to a single polygon so callers get one clean boundary
    boundary = study_counties.dissolve().reset_index(drop=True)

    return boundary


def filter_points(
    gdf: gpd.GeoDataFrame,
    shapefile_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """
    Filter a GeoDataFrame of point geometries to only those that fall
    inside the eight-county boundary.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input points. Must have a geometry column. CRS will be
        reprojected to EPSG:4326 automatically if it differs.
    shapefile_path : str or Path, optional
        Passed through to load_county_boundary().

    Returns
    -------
    GeoDataFrame
        Subset of the input containing only points inside the boundary,
        with the original CRS restored.

    Raises
    ------
    ValueError
        If the GeoDataFrame has no geometry column.
    """
    if gdf.geometry is None or gdf.empty:
        raise ValueError("Input GeoDataFrame has no geometry or is empty.")

    original_crs = gdf.crs
    boundary = load_county_boundary(shapefile_path)

    # Align CRS before spatial join
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84_CRS)
    elif gdf.crs.to_epsg() != WGS84_EPSG:
        gdf = gdf.to_crs(WGS84_CRS)

    filtered = gpd.sjoin(gdf, boundary[["geometry"]], how="inner", predicate="within")

    # Drop the index column added by sjoin
    filtered = filtered.drop(columns=["index_right"], errors="ignore")

    # Restore the original CRS if it was different
    if original_crs is not None and original_crs.to_epsg() != WGS84_EPSG:
        filtered = filtered.to_crs(original_crs)

    return filtered.reset_index(drop=True)


def clip_raster(
    raster_dataset,  # noqa: ANN001
    shapefile_path: str | Path | None = None,
    all_touched: bool = False,
) -> tuple:
    """
    Clip an open rasterio dataset to the eight-county boundary.

    Parameters
    ----------
    raster_dataset : rasterio.io.DatasetReader
        An already-opened rasterio dataset (i.e. the result of
        rasterio.open()). The caller is responsible for opening and
        closing the file.
    shapefile_path : str or Path, optional
        Passed through to load_county_boundary().
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

    Notes
    -----
    rasterio is imported inside this function so that scripts that only
    need filter_points() do not require rasterio to be installed.
    """
    if importlib.util.find_spec("rasterio") is None:
        raise ImportError(
            "rasterio is required for clip_raster(). "
            "Install it with: pip install rasterio"
        )

    boundary = load_county_boundary(shapefile_path)

    # Reproject boundary to match raster CRS
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


def bbox_geometry() -> box:
    """
    Return the study area bounding box as a shapely Polygon in EPSG:4326.
    Useful for quick pre-filtering large datasets before the more
    expensive point-in-polygon check.
    """
    min_lon, min_lat, max_lon, max_lat = BBOX
    return box(min_lon, min_lat, max_lon, max_lat)


if __name__ == "__main__":
    print("Eight-county study area")
    print(f"  Counties : {', '.join(COUNTIES)}")
    print(f"  FIPS     : {', '.join(COUNTY_FIPS)}")
    print(f"  BBox     : {BBOX}")
    print()

    try:
        boundary = load_county_boundary()
        print("Boundary loaded successfully.")
        print(f"  CRS      : {boundary.crs}")
        print(f"  Area     : {boundary.geometry.area.values[0]:.4f} sq degrees")
        print(f"  Bounds   : {boundary.total_bounds}")
    except FileNotFoundError as e:
        print(f"Shapefile not found (expected during CI): {e}")

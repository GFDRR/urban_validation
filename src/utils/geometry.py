"""
Geometry and CRS helpers.

Invalid-geometry repair, GeoDataFrame loading from any vector format,
projected-CRS selection (UTM zone), and Earth Engine geometry conversion.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import pyproj
from shapely import make_valid as _make_valid

log = logging.getLogger(__name__)


def _read_gdf(path: Union[str, Path]) -> gpd.GeoDataFrame:
    """Read a vector file as a GeoDataFrame, dispatching by suffix."""
    path = Path(path)
    if path.suffix.lower() in [".parquet", ".geoparquet"]:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def validate_aoi_geometry(gdf: gpd.GeoDataFrame, label: str = "") -> gpd.GeoDataFrame:
    """
    Fix invalid geometries (self-intersections, rings, etc.) and drop empty
    geometries. Uses shapely.make_valid where possible; falls back to buffer(0).
    """
    gdf = gdf.copy()

    # Drop missing/empty early
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    invalid = ~gdf.geometry.is_valid
    n_invalid = int(invalid.sum())

    if n_invalid == 0:
        return gdf

    print(f"[{label}] fixing {n_invalid} invalid geometries...")

    try:
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(_make_valid)
    except Exception:
        # Fallback (less robust, but often works)
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # Drop anything that became empty after fixing
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    still_invalid = int((~gdf.geometry.is_valid).sum())
    if still_invalid:
        print(f"[{label}] warning: {still_invalid} geometries are still invalid after fixing.")

    return gdf


def get_projected_crs(gdf: gpd.GeoDataFrame) -> str:
    """Pick a UTM CRS appropriate for the GeoDataFrame's centroid."""
    if gdf.crs and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)
    bounds = gdf.total_bounds
    lon = (bounds[0] + bounds[2]) / 2
    lat = (bounds[1] + bounds[3]) / 2
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon, south_lat_degree=lat,
            east_lon_degree=lon, north_lat_degree=lat,
        ),
    )
    if utm_crs_list:
        return f"EPSG:{utm_crs_list[0].code}"
    zone_number = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone_number}" if lat >= 0 else f"EPSG:{32700 + zone_number}"


def _shapely_to_geojson_dict(geom) -> dict:
    """Round-trip a shapely geometry to a GeoJSON dict via GeoSeries."""
    return json.loads(
        gpd.GeoSeries([geom], crs="EPSG:4326").to_json()
    )["features"][0]["geometry"]


def aoi_gdf_to_ee_geometry(gdf):
    """Build an ee.Geometry from the dissolved AOI GeoDataFrame."""
    import ee
    return ee.Geometry(_shapely_to_geojson_dict(gdf.union_all()))

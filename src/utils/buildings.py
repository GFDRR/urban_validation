"""
Building footprint and AOI loaders.

load_buildings: read footprints, reproject, compute area (chunked when
UTM-fallback is needed), filter by minimum area.
load_aoi: read an AOI vector, optionally buffer in metric CRS, dissolve,
and reproject to the configured output CRS.
"""
from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pyproj

from src.utils.geometry import _read_gdf, validate_aoi_geometry

log = logging.getLogger(__name__)

# Rows per chunk when computing UTM areas
_AREA_CHUNK_SIZE = int(os.environ.get("AREA_CHUNK_SIZE", 50_000))


def load_buildings(
    path: Union[str, Path],
    *,
    crs_work: str,
    min_area_m2: float,
    fix_invalid_geoms: bool = False,
    compute_area_mode: str = "auto",
    logger=None,
) -> gpd.GeoDataFrame:
    """
    Load building footprints, reproject, compute area, filter by min area.
    """
    path = Path(path)
    gdf = (gpd.read_parquet(path) if path.suffix.lower() in {".parquet", ".geoparquet"}
           else gpd.read_file(path))

    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS defined.")

    n_before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    n_dropped = n_before - len(gdf)
    if n_dropped > 0:
        msg = f"[{path.name}] Dropped {n_dropped} null/empty geometries ({n_before} → {len(gdf)})"
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    if gdf.empty:
        gdf = gdf.to_crs(crs_work)
        gdf["area_m2"] = np.float64()
        return gdf

    gdf = gdf.to_crs(crs_work)

    if fix_invalid_geoms:
        gdf = validate_aoi_geometry(gdf, label=path.name)

    if compute_area_mode == "auto":
        crs_obj = pyproj.CRS(crs_work)
        if crs_obj.is_projected:
            compute_area_mode = "work_crs"
        else:
            compute_area_mode = "utm"

    if compute_area_mode == "work_crs":
        gdf["area_m2"] = gdf.geometry.area

    elif compute_area_mode == "utm":
        valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty
        if not valid_mask.all():
            gdf = gdf[valid_mask].copy()

        metric_crs = gdf.estimate_utm_crs()
        n = len(gdf)
        areas = np.empty(n, dtype=np.float64)

        chunk = _AREA_CHUNK_SIZE
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            chunk_gdf = gdf.iloc[start:end].to_crs(metric_crs)
            areas[start:end] = chunk_gdf.geometry.area.values
            del chunk_gdf

        del metric_crs
        gc.collect()

        gdf["area_m2"] = areas
        del areas
    else:
        raise ValueError(f"Unknown compute_area_mode={compute_area_mode!r}")

    gdf = gdf[gdf["area_m2"] >= float(min_area_m2)].copy()
    gdf.reset_index(drop=True, inplace=True)

    if logger:
        logger.info("Loaded buildings | n=%d | path=%s", len(gdf), path)

    return gdf


def load_aoi(
    path: Union[str, Path],
    *,
    crs_out: str = "EPSG:4326",
    buffer_meters: float = 0.0,
    dissolve: bool = False,
    logger=None,
) -> gpd.GeoDataFrame:
    """Load an AOI vector, optionally buffer (in metric CRS), dissolve, and reproject."""
    path = Path(path)
    aoi = _read_gdf(path)

    if aoi.crs is None:
        if logger:
            logger.warning("AOI CRS missing; assuming %s", crs_out)
        aoi = aoi.set_crs("EPSG:4326")

    if dissolve and len(aoi) > 1:
        aoi = aoi.dissolve().reset_index(drop=True)

    buffer_meters = float(buffer_meters or 0.0)
    if buffer_meters > 0:
        if not aoi.crs.is_projected:
            if logger:
                logger.warning("AOI CRS is geographic; reprojecting to EPSG:3857 for buffering")
            aoi_metric = aoi.to_crs("EPSG:3857")
            aoi_metric["geometry"] = aoi_metric.geometry.buffer(buffer_meters)
            aoi = aoi_metric.to_crs(aoi.crs)
        else:
            aoi["geometry"] = aoi.geometry.buffer(buffer_meters)

    if str(aoi.crs) != str(crs_out):
        if logger:
            logger.info("Reprojecting AOI | %s -> %s", aoi.crs, crs_out)
        aoi = aoi.to_crs(crs_out)

    if logger:
        logger.info("Loaded AOI | rows=%d | crs=%s | path=%s", len(aoi), aoi.crs, path)

    return aoi
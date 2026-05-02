"""
Tile-grid construction, spatial-index subsetting, and output-path resolution.
"""
from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import box

log = logging.getLogger(__name__)


def make_tiles(
    aoi: gpd.GeoDataFrame,
    tile_size_m: float,
    *,
    clip_to_aoi: bool = False,
    snap_origin: bool = False,
) -> gpd.GeoDataFrame:
    """Build a square tile grid covering the AOI (already in a metric CRS)."""
    if aoi.empty:
        return gpd.GeoDataFrame({"tile_id": [], "geometry": []}, crs=aoi.crs)

    aoi_union = aoi.geometry.union_all()
    minx, miny, maxx, maxy = aoi_union.bounds
    tile = float(tile_size_m)

    if snap_origin:
        minx = np.floor(minx / tile) * tile
        miny = np.floor(miny / tile) * tile
        maxx = np.ceil(maxx / tile) * tile
        maxy = np.ceil(maxy / tile) * tile

    nx = int(np.ceil((maxx - minx) / tile))
    ny = int(np.ceil((maxy - miny) / tile))

    tiles = []
    for ix in range(nx):
        x0 = minx + ix * tile
        x1 = x0 + tile
        for iy in range(ny):
            y0 = miny + iy * tile
            y1 = y0 + tile
            poly = box(x0, y0, x1, y1)

            if not poly.intersects(aoi_union):
                continue

            tiles.append(poly.intersection(aoi_union) if clip_to_aoi else poly)

    tiles_gdf = gpd.GeoDataFrame({"geometry": tiles}, crs=aoi.crs)
    tiles_gdf.reset_index(drop=True, inplace=True)
    tiles_gdf["tile_id"] = tiles_gdf.index.astype(int)
    return tiles_gdf[["tile_id", "geometry"]]


def subset_by_tile(
    buildings: gpd.GeoDataFrame,
    sindex,
    tile_geom,
):
    """Subset a buildings GeoDataFrame to those intersecting a tile geometry."""
    possible_idx = list(sindex.intersection(tile_geom.bounds))
    if not possible_idx:
        return buildings.iloc[[]].copy()

    subset = buildings.iloc[possible_idx]
    subset = subset[subset.intersects(tile_geom)].copy()
    return subset


def resolve_out_root(config, dataset_id: str, subdir: str = "vector") -> Path:
    """Resolve the output directory for a dataset."""
    use_base = config.output.use_base_dir_for_output
    if use_base:
        p = Path(config.aoi.base_dir) / dataset_id / subdir
    else:
        root = config.output.root_dir or "data/outputs"
        p = Path(root) / dataset_id / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p

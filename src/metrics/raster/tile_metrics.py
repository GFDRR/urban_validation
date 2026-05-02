"""
Tile-level raster metric orchestrator.

For each candidate raster, iterates tile × evaluation-grid, reads and
reprojects the tile, rasterizes reference polygons, computes binary
+ area-based metrics + Pontius disagreement, and emits one row per
(tile, grid).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from src.metrics.raster.binarize import (
    pred_bin_from_pred_area,
    predicted_area_from_raster,
)
from src.metrics.raster.disagreement import _compute_quantity_allocation_disagreement
from src.metrics.raster.grids import (
    _empty_raster_tile_row,
    _filter_evaluation_grids_by_native_resolution,
    _normalize_evaluation_grids,
)
from src.metrics.raster.io import (
    _open_raster_in_crs,
    _read_reprojected_tile,
)
from src.metrics.raster.rasterize import (
    _aoi_mask_for_window,
    _make_grid_aligned_transform,
    _pixel_area_from_transform,
    _rasterize_ref_fraction,
)
from src.utils import subset_by_tile

logger = logging.getLogger("Validation_Metrics")


def compute_raster_tile_metrics(
    raster_path: Path,
    cand_cfg: dict,
    ref_all: gpd.GeoDataFrame,
    ref_sindex,
    aoi_union,
    tiles: gpd.GeoDataFrame,
    tau_frac: float,
    default_oversample: int = 4,
    default_all_touched: bool = False,
    evaluation_grids: Optional[List[dict]] = None,
    native_guard_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Tile-level raster validation for one candidate dataset, evaluated at one
    or more target grid resolutions.

    For each tile and for each evaluation grid:
      1. Read/reproject candidate raster onto the target grid.
      2. Rasterize reference polygons onto the same target grid.
      3. Compute binary and area-based metrics.

    Returns one row per (tile, grid).
    """
    bin_spec = cand_cfg.get("binarize", {"method": "fraction"})
    rast_over = cand_cfg.get("rasterization", {}) or {}

    base_oversample = int(rast_over.get("oversample_factor", default_oversample))
    base_all_touched = bool(rast_over.get("all_touched", default_all_touched))
    band = int(bin_spec.get("band", 1))

    fallback_resolution = cand_cfg.get("native_resolution_m", 10)

    grids = _normalize_evaluation_grids(
        evaluation_grids=evaluation_grids,
        fallback_resolution=fallback_resolution,
    )

    grids = _filter_evaluation_grids_by_native_resolution(
        evaluation_grids=grids,
        cand_cfg=cand_cfg,
        guard_cfg=native_guard_cfg,
    )

    if not grids:
        logger.warning(
            "[raster tile metrics] No evaluation grids remain after native-resolution "
            "guard for dataset '%s'. Returning empty metrics.",
            cand_cfg.get("name", "<unknown>"),
        )
        return pd.DataFrame()

    rows: List[dict] = []

    with rasterio.open(raster_path) as _src:
        ds = _open_raster_in_crs(_src, str(tiles.crs), bin_spec)
        nodata = bin_spec.get("nodata", None)

        for tile_row in tiles.itertuples():
            tile_id = int(tile_row.tile_id)
            geom = tile_row.geometry

            # Subset reference once per tile, reuse across all evaluation grids
            ref_tile = subset_by_tile(ref_all, ref_sindex, geom)
            ref_geoms = list(ref_tile.geometry.values) if not ref_tile.empty else []

            for grid in grids:
                grid_name = str(grid["name"])
                resolution = float(grid["resolution"])
                oversample = int(
                    grid["oversample_factor"]
                    if grid.get("oversample_factor", None) is not None
                    else base_oversample
                )
                all_touched = bool(
                    grid["all_touched"]
                    if grid.get("all_touched", None) is not None
                    else base_all_touched
                )

                transform, out_shape = _make_grid_aligned_transform(geom.bounds, resolution)
                pixel_area = _pixel_area_from_transform(transform)

                fill = float(nodata) if nodata is not None else np.nan

                try:
                    arr = _read_reprojected_tile(
                        src=ds,
                        band=band,
                        dst_shape=out_shape,
                        dst_transform=transform,
                        dst_crs=tiles.crs,
                        binarize_spec=bin_spec,
                        fill_value=fill,
                    )

                    aoi_mask = _aoi_mask_for_window(
                        aoi_union,
                        arr.shape,
                        transform,
                        all_touched=all_touched,
                    )
                    tile_mask = _aoi_mask_for_window(
                        geom,
                        arr.shape,
                        transform,
                        all_touched=all_touched,
                    )

                    valid = aoi_mask & tile_mask
                    method = bin_spec.get("method", "")

                    if nodata is not None:
                        if method in {"fraction", "percent", "area_m2"}:
                            valid &= np.isfinite(arr)
                        else:
                            valid &= (arr != nodata)
                    else:
                        valid &= np.isfinite(arr)

                    n_valid = int(valid.sum())

                    if n_valid == 0:
                        rows.append(
                            _empty_raster_tile_row(
                                tile_id,
                                pixel_area,
                                grid_name=grid_name,
                                resolution=resolution,
                            )
                        )
                        continue

                    ref_frac = _rasterize_ref_fraction(
                        ref_geoms,
                        out_shape=arr.shape,
                        transform=transform,
                        oversample=oversample,
                        all_touched=all_touched,
                    )
                    ref_bin = ref_frac >= tau_frac

                    A_ref = ref_frac * pixel_area
                    A_pred = predicted_area_from_raster(arr, transform, bin_spec)
                    pred_bin = pred_bin_from_pred_area(A_pred, transform, bin_spec, tau_frac)

                    tp = int(np.sum(valid & ref_bin & pred_bin))
                    fp = int(np.sum(valid & (~ref_bin) & pred_bin))
                    fn = int(np.sum(valid & ref_bin & (~pred_bin)))
                    tn = int(np.sum(valid & (~ref_bin) & (~pred_bin)))

                    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                    f1 = (
                        2.0 * precision * recall / (precision + recall)
                        if np.isfinite(precision)
                        and np.isfinite(recall)
                        and (precision + recall) > 0
                        else np.nan
                    )

                    ref_area_m2 = float(np.sum(A_ref[valid]))
                    pred_area_m2 = float(np.sum(A_pred[valid]))
                    valid_area_m2 = float(n_valid * pixel_area)

                    rel_area_error = (
                        (pred_area_m2 - ref_area_m2) / ref_area_m2
                        if ref_area_m2 > 0
                        else np.nan
                    )
                    signed_area_bias = rel_area_error

                    qd, ad = _compute_quantity_allocation_disagreement(
                        ref_bin, pred_bin, valid
                    )

                    rows.append(
                        {
                            "tile_id": tile_id,
                            "grid": grid_name,
                            "resolution_m": resolution,
                            "pixel_area_m2": float(pixel_area),
                            "n_pixels": int(arr.size),
                            "n_valid": n_valid,
                            "valid_area_m2": valid_area_m2,
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "tn": tn,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "ref_area_m2": ref_area_m2,
                            "pred_area_m2": pred_area_m2,
                            "rel_area_error": rel_area_error,
                            "signed_area_bias": signed_area_bias,
                            "quantity_disagreement": qd,
                            "allocation_disagreement": ad,
                            "native_resolution_m": float(cand_cfg["native_resolution_m"])
                            if cand_cfg.get("native_resolution_m") is not None
                            else np.nan,
                        }
                    )

                    del arr, ref_frac, ref_bin, A_ref, A_pred, pred_bin
                    del valid, aoi_mask, tile_mask

                except Exception as exc:
                    logger.exception(
                        "[raster tile metrics] tile=%s grid=%s resolution=%s failed: %s",
                        tile_id,
                        grid_name,
                        resolution,
                        exc,
                    )
                    rows.append(
                        _empty_raster_tile_row(
                            tile_id,
                            pixel_area,
                            grid_name=grid_name,
                            resolution=resolution,
                        )
                    )

            del ref_tile

        if ds is not _src:
            ds.close()

    return pd.DataFrame(rows)

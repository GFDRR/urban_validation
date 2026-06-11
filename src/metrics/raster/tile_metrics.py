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
    _block_average_binary_to_grid,
    _open_raster_in_crs,
    _read_reprojected_tile,
    _reproject_area_to_grid,
    _reproject_binary_to_grid,
)
from src.metrics.raster.rasterize import (
    _aoi_mask_for_window,
    _make_grid_aligned_transform,
    _pixel_area_from_transform,
    _rasterize_ref_fraction,
)
from src.utils.tiling import subset_by_tile

logger = logging.getLogger("Validation_Metrics")


def compute_raster_tile_metrics(
    raster_path: Path,
    cand_cfg: dict,
    ref_all: gpd.GeoDataFrame,
    ref_sindex,
    aoi_union,
    tiles: gpd.GeoDataFrame,
    min_building_m2: float,
    ref_min_building_m2: float,
    default_oversample: int = 4,
    default_all_touched: bool = False,
    evaluation_grids: Optional[List[dict]] = None,
    native_guard_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Tile-level raster validation for one candidate dataset, evaluated at one
    or more target grid resolutions.

    For each tile:
      1. Read the candidate raster at native resolution.
      2. Binarize at native resolution using the per-dataset threshold:
           tau_frac_native = min_building_m2 / native_pixel_area.
         Categorical methods (wsf_tracker, binary, nonzero, value_in) bypass
         tau_frac and use A_pred > 0 directly.
      3. Block-average the binary mask to each evaluation grid (fraction of built
         native pixels per eval cell), then threshold at tau_frac = ref_min_building_m2
         / eval_pixel_area — the same threshold applied to the reference side.
      4. Rasterize reference polygons onto the evaluation grid (fractional coverage).
         Reference binarization uses the global threshold:
           tau_frac = ref_min_building_m2 / eval_pixel_area.
         This keeps the reference definition of "built" consistent across all
         dataset comparisons at the same evaluation grid.
      5. Compute binary and area-based metrics.

    Parameters
    ----------
    min_building_m2 : float
        Per-dataset minimum detectable building size (m²). Controls prediction
        binarization at native resolution.
    ref_min_building_m2 : float
        Global minimum building size (m²) for reference binarization at the
        evaluation grid. Fixed across all datasets so comparisons are consistent.

    Returns one row per (tile, grid).
    """
    bin_spec = cand_cfg.get("binarize", {"method": "fraction"})
    rast_over = cand_cfg.get("rasterization", {}) or {}

    base_oversample = int(rast_over.get("oversample_factor", default_oversample))
    base_all_touched = bool(rast_over.get("all_touched", default_all_touched))
    band = int(bin_spec.get("band", 1))
    method = bin_spec.get("method", "fraction")
    nodata = bin_spec.get("nodata", None)
    fill = float(nodata) if nodata is not None else np.nan

    native_res = float(cand_cfg.get("native_resolution_m", 10))

    grids = _normalize_evaluation_grids(
        evaluation_grids=evaluation_grids,
        fallback_resolution=native_res,
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

        for tile_row in tiles.itertuples():
            tile_id = int(tile_row.tile_id)
            geom = tile_row.geometry

            ref_tile = subset_by_tile(ref_all, ref_sindex, geom)
            ref_geoms = list(ref_tile.geometry.values) if not ref_tile.empty else []
            ref_building_count = int(len(ref_tile))
            mean_ref_building_area_m2 = (
                float(ref_tile["area_m2"].mean())
                if not ref_tile.empty and "area_m2" in ref_tile.columns
                else np.nan
            )

            # Step 1: Read at native resolution (once per tile, shared across grids).
            native_transform, native_shape = _make_grid_aligned_transform(geom.bounds, native_res)
            native_pixel_area = _pixel_area_from_transform(native_transform)
            tau_frac_native = min(1.0, min_building_m2 / native_pixel_area)

            try:
                native_arr = _read_reprojected_tile(
                    src=ds,
                    band=band,
                    dst_shape=native_shape,
                    dst_transform=native_transform,
                    dst_crs=tiles.crs,
                    binarize_spec=bin_spec,
                    fill_value=fill,
                )
            except Exception as exc:
                logger.exception(
                    "[raster tile metrics] tile=%s native read failed: %s",
                    tile_id,
                    exc,
                )
                for grid in grids:
                    _t, _s = _make_grid_aligned_transform(geom.bounds, float(grid["resolution"]))
                    rows.append(
                        _empty_raster_tile_row(
                            tile_id,
                            _pixel_area_from_transform(_t),
                            grid_name=str(grid["name"]),
                            resolution=float(grid["resolution"]),
                        )
                    )
                del ref_tile
                continue

            # Native valid mask: tracks which pixels are within source extent / not nodata.
            if nodata is not None:
                if method in {"fraction", "percent", "area_m2"}:
                    native_valid = np.isfinite(native_arr)
                else:
                    native_valid = native_arr != float(nodata)
            else:
                native_valid = np.isfinite(native_arr)

            # Step 2: Binarize at native resolution.
            A_pred_native = predicted_area_from_raster(native_arr, native_transform, bin_spec)
            pred_bin_native = pred_bin_from_pred_area(
                A_pred_native, native_transform, bin_spec, tau_frac_native
            )
            pred_bin_native = pred_bin_native & native_valid

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
                tau_frac = min(1.0, ref_min_building_m2 / pixel_area)

                # Skip reprojection when native resolution already matches eval resolution
                # (e.g. TEMPO/GHSL at 100 m native on a 100 m eval grid).
                same_res = abs(native_res - resolution) / max(native_res, resolution) < 1e-3

                try:
                    # Step 3: Project binary native mask to eval grid, then threshold.
                    # tau_frac (ref_min_building_m2 / eval_pixel_area) is applied
                    # symmetrically to both pred and ref at the eval-grid step.
                    if same_res:
                        # Native and eval grids are identical — no reprojection needed.
                        # pred_bin_native is already binary (0/1); cast to float so the
                        # >= tau_frac comparison is consistent with the non-same_res path.
                        pred_frac_at_eval = pred_bin_native.astype(np.float32)
                        native_valid_at_eval = native_valid
                    else:
                        pred_frac_at_eval = _block_average_binary_to_grid(
                            pred_bin_native,
                            native_valid,
                            src_transform=native_transform,
                            src_crs=tiles.crs,
                            dst_shape=out_shape,
                            dst_transform=transform,
                            dst_crs=tiles.crs,
                        )
                        native_valid_at_eval = _reproject_binary_to_grid(
                            native_valid,
                            src_transform=native_transform,
                            src_crs=tiles.crs,
                            dst_shape=out_shape,
                            dst_transform=transform,
                            dst_crs=tiles.crs,
                        ).astype(bool)
                    pred_bin = pred_frac_at_eval >= tau_frac

                    aoi_mask = _aoi_mask_for_window(
                        aoi_union, out_shape, transform, all_touched=all_touched,
                    )
                    tile_mask = _aoi_mask_for_window(
                        geom, out_shape, transform, all_touched=all_touched,
                    )
                    valid = aoi_mask & tile_mask & native_valid_at_eval

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

                    # Step 4: Rasterize reference at evaluation grid (unchanged).
                    ref_frac = _rasterize_ref_fraction(
                        ref_geoms,
                        out_shape=out_shape,
                        transform=transform,
                        oversample=oversample,
                        all_touched=all_touched,
                    )
                    ref_bin = ref_frac >= tau_frac
                    A_ref = ref_frac * pixel_area

                    # Step 5: Metrics.
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
                    A_pred_eval = _reproject_area_to_grid(
                        A_pred_native,
                        native_valid,
                        src_transform=native_transform,
                        src_crs=tiles.crs,
                        dst_shape=out_shape,
                        dst_transform=transform,
                        dst_crs=tiles.crs,
                        native_pixel_area=native_pixel_area,
                        eval_pixel_area=pixel_area,
                    )
                    pred_area_m2 = float(np.sum(A_pred_eval[valid]))
                    valid_area_m2 = float(n_valid * pixel_area)
                    pred_building_count = (
                        pred_area_m2 / mean_ref_building_area_m2
                        if np.isfinite(mean_ref_building_area_m2)
                        and mean_ref_building_area_m2 > 0
                        else np.nan
                    )
                    delta_building_count = (
                        pred_building_count - ref_building_count
                        if np.isfinite(pred_building_count)
                        else np.nan
                    )
                    rel_delta_building_count = (
                        delta_building_count / ref_building_count
                        if ref_building_count > 0 and np.isfinite(delta_building_count)
                        else np.nan
                    )

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
                            "n_pixels": int(out_shape[0] * out_shape[1]),
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
                            "ref_building_count": ref_building_count,
                            "mean_ref_building_area_m2": mean_ref_building_area_m2,
                            "pred_building_count": pred_building_count,
                            "delta_building_count": delta_building_count,
                            "rel_delta_building_count": rel_delta_building_count,
                            "rel_area_error": rel_area_error,
                            "signed_area_bias": signed_area_bias,
                            "quantity_disagreement": qd,
                            "allocation_disagreement": ad,
                            "native_resolution_m": float(cand_cfg["native_resolution_m"])
                            if cand_cfg.get("native_resolution_m") is not None
                            else np.nan,
                        }
                    )

                    del pred_bin, pred_frac_at_eval, A_pred_eval, ref_frac, ref_bin, A_ref
                    del valid, native_valid_at_eval, aoi_mask, tile_mask

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

            del ref_tile, native_arr, A_pred_native, pred_bin_native, native_valid

        if ds is not _src:
            ds.close()

    return pd.DataFrame(rows)

"""
TODO: contains vector and raster metrics for assessing Building Footprint Datasets for different use cases

1. match_buildings_iou: chunked sjoin + vectorised ops to cap peak RAM
2. boundary_f_pair: unchanged (small per-call)
3. compute_tile_metrics: explicit del of intermediate arrays
4. compute_boundary_f_for_tile: unchanged (tile-scoped, manageable)
"""
from __future__ import annotations
import os
import yaml
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
import duckdb

logger = logging.getLogger("Validation_Metrics")
logger.setLevel(logging.INFO)
fmt = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
sh = logging.StreamHandler()
sh.setFormatter(fmt)
if not logger.handlers:
    logger.addHandler(sh)

# ── Tunable: maximum candidate-side rows per sjoin chunk ──────────────
# Keeps peak RAM of the vectorised geometry arrays bounded.
# On Colab free tier (~12 GB) 50k is conservative; increase on beefier machines.
_SJOIN_CHUNK_SIZE = int(os.environ.get("SJOIN_CHUNK_SIZE", 50_000))


def boundary_f_pair(ref_geom, cand_geom, tau_boundary_m: float) -> float:
    """Boundary F for a single matched pair (length-within-tolerance)."""
    rb = ref_geom.boundary
    cb = cand_geom.boundary
    if rb.length == 0 or cb.length == 0:
        return 0.0

    rb_buf = rb.buffer(tau_boundary_m)
    cb_buf = cb.buffer(tau_boundary_m)

    p = cb.intersection(rb_buf).length / cb.length if cb.length > 0 else 0.0
    r = rb.intersection(cb_buf).length / rb.length if rb.length > 0 else 0.0
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def compute_boundary_f_for_tile(ref_tile, cand_tile, matches_df, tau_boundary_m):
    """
    Compute boundary F-measure for all TPs in this tile using
    boundary length within buffered boundaries.
    """
    if matches_df.empty:
        return 0.0

    ref_ids = matches_df["ref_id"].unique()
    cand_ids = matches_df["cand_id"].unique()

    ref_geoms = ref_tile.loc[ref_ids].geometry
    cand_geoms = cand_tile.loc[cand_ids].geometry

    # union_all() on many boundaries creates a large multi-geometry; free
    # the per-building boundary series immediately after unioning.
    ref_bound = ref_geoms.boundary.union_all()
    del ref_geoms
    cand_bound = cand_geoms.boundary.union_all()
    del cand_geoms

    if ref_bound.length == 0 or cand_bound.length == 0:
        return 0.0

    ref_buffer = ref_bound.buffer(tau_boundary_m)
    cand_buffer = cand_bound.buffer(tau_boundary_m)

    # Precision: length of cand boundary within tau of ref boundary
    P_b = cand_bound.intersection(ref_buffer).length / cand_bound.length if cand_bound.length > 0 else 0.0
    del ref_buffer

    # Recall: length of ref boundary within tau of cand boundary
    R_b = ref_bound.intersection(cand_buffer).length / ref_bound.length if ref_bound.length > 0 else 0.0
    del cand_buffer, ref_bound, cand_bound

    if P_b + R_b == 0:
        return 0.0

    return 2 * P_b * R_b / (P_b + R_b)


def _safe_quantile(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s) else 0.0


def compute_tile_metrics(ref_tile, city, cand_tile, tau_overlap, tau_buffer_m, tau_boundary_m, tile_id, dataset_name):
    matches_df, ref_unmatched, cand_unmatched = match_buildings_iou(
        ref_tile, cand_tile, tau_overlap, tau_buffer_m=tau_buffer_m
    )

    # Per-match boundary F (only for TPs)
    if not matches_df.empty:
        bf_vals = []
        for ref_id, cand_id in matches_df[["ref_id", "cand_id"]].itertuples(index=False):
            bf_vals.append(boundary_f_pair(ref_tile.loc[ref_id].geometry, cand_tile.loc[cand_id].geometry, tau_boundary_m))
        matches_df = matches_df.copy()
        matches_df["boundary_f_pair"] = bf_vals
        del bf_vals  # FIX: free list immediately

    n_ref = len(ref_tile)
    n_cand = len(cand_tile)
    tp = len(matches_df)
    fn = len(ref_unmatched)
    fp = len(cand_unmatched)

    # FIX: free unmatched sets — no longer needed
    del ref_unmatched, cand_unmatched

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if tp > 0:
        mean_iou = float(matches_df["iou"].mean())
        median_iou = float(matches_df["iou"].median())
        iou_p25 = _safe_quantile(matches_df["iou"], 0.25)
        iou_p75 = _safe_quantile(matches_df["iou"], 0.75)
    else:
        mean_iou = median_iou = iou_p25 = iou_p75 = 0.0

    boundary_f_union = compute_boundary_f_for_tile(ref_tile, cand_tile, matches_df, tau_boundary_m)
    boundary_f_meanpair = float(matches_df["boundary_f_pair"].mean()) if tp > 0 and "boundary_f_pair" in matches_df.columns else 0.0

    mean_rel_area_error = float(matches_df["rel_area_error"].mean()) if tp > 0 else np.nan
    area_ref_sum = float(matches_df["area_ref"].sum()) if tp > 0 else 0.0
    area_cand_sum = float(matches_df["area_cand"].sum()) if tp > 0 else 0.0
    signed_area_bias = ((area_cand_sum - area_ref_sum) / area_ref_sum) if area_ref_sum > 0 else np.nan

    metrics = {
        "city": city,
        "dataset": dataset_name,
        "tile_id": tile_id,
        "n_ref": n_ref,
        "n_cand": n_cand,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "median_iou": median_iou,
        "iou_p25": iou_p25,
        "iou_p75": iou_p75,
        "boundary_f_union": boundary_f_union,
        "boundary_f_meanpair": boundary_f_meanpair,
        "mean_rel_area_error": mean_rel_area_error,
        "signed_area_bias": signed_area_bias,
        "tau_overlap": tau_overlap,
        "tau_buffer_m": tau_buffer_m,
        "tau_boundary_m": tau_boundary_m,
    }

    return metrics, matches_df


def _iou_with_buffer(ref_geom, cand_geom, tau_buffer_m: float = 0.0) -> float:
    """IoU with optional buffering to tolerate small georegistration offsets."""
    if tau_buffer_m and tau_buffer_m > 0:
        ref_geom = ref_geom.buffer(tau_buffer_m)
        cand_geom = cand_geom.buffer(tau_buffer_m)

    inter = ref_geom.intersection(cand_geom).area
    if inter <= 0:
        return 0.0
    union = ref_geom.union(cand_geom).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def match_buildings_iou(
    ref_tile,
    cand_tile,
    tau_overlap: float,
    tau_buffer_m: float = 0.0,
):
    """
    Vectorised 1-to-1 IoU matching with CHUNKED processing.

    MEMORY FIX: The original version materialised all geometry arrays at once.
    For dense tiles (e.g. 20k ref × 30k cand) the sjoin output, buffered
    copies, intersection arrays, and union arrays can easily exceed 4-8 GB.

    This version:
      a) Chunks the candidate GeoDataFrame before sjoin so each chunk's
         intermediate arrays stay bounded (~50k pairs max per chunk).
      b) Deletes intermediate numpy/shapely arrays as soon as IoU is computed.
      c) Concatenates only the lightweight (ref_id, cand_id, iou) triples
         before the greedy matching pass.
    """
    empty_cols = ["ref_id", "cand_id", "iou", "area_ref", "area_cand", "rel_area_error"]

    if ref_tile.empty and cand_tile.empty:
        return pd.DataFrame(columns=empty_cols), set(), set()
    if ref_tile.empty:
        return pd.DataFrame(columns=empty_cols), set(), set(cand_tile.index)
    if cand_tile.empty:
        return pd.DataFrame(columns=empty_cols), set(ref_tile.index), set()

    ref_g = ref_tile[["geometry"]].copy()
    cand_g = cand_tile[["geometry"]].copy()

    # ── Chunked sjoin + vectorised IoU ─────────────────────────────────
    # Split candidate into chunks to cap peak memory of geometry arrays.
    n_cand = len(cand_g)
    chunk_size = max(_SJOIN_CHUNK_SIZE, 1)
    all_triples = []  # lightweight list of (ref_id, cand_id, iou) arrays

    for start in range(0, n_cand, chunk_size):
        cand_chunk = cand_g.iloc[start : start + chunk_size]

        joined = gpd.sjoin(ref_g, cand_chunk, how="inner", predicate="intersects")
        if joined.empty:
            continue

        pairs = (
            joined.reset_index()
            .rename(columns={"index": "ref_id", "index_right": "cand_id"})
            [["ref_id", "cand_id"]]
        )
        del joined  # free sjoin result immediately

        ref_geoms_arr = ref_tile.loc[pairs["ref_id"].values, "geometry"].values
        cand_geoms_arr = cand_tile.loc[pairs["cand_id"].values, "geometry"].values

        if tau_buffer_m and tau_buffer_m > 0:
            ref_geoms_arr = shapely.buffer(ref_geoms_arr, tau_buffer_m)
            cand_geoms_arr = shapely.buffer(cand_geoms_arr, tau_buffer_m)

        inter_areas = shapely.area(shapely.intersection(ref_geoms_arr, cand_geoms_arr))
        # FIX: delete intersection geometries before computing union
        union_areas = shapely.area(shapely.union(ref_geoms_arr, cand_geoms_arr))

        # FIX: free bulky geometry arrays now that we only need scalar areas
        del ref_geoms_arr, cand_geoms_arr

        with np.errstate(divide="ignore", invalid="ignore"):
            ious = np.where(union_areas > 0, inter_areas / union_areas, 0.0)

        del inter_areas, union_areas  # FIX: free area arrays

        mask = ious > 0
        if mask.any():
            all_triples.append(np.column_stack([
                pairs["ref_id"].values[mask],
                pairs["cand_id"].values[mask],
                ious[mask],
            ]))
        del pairs, ious, mask

    # ref_g / cand_g no longer needed after the chunk loop
    del ref_g, cand_g

    # ── Assemble and run greedy matching ───────────────────────────────
    if not all_triples:
        return pd.DataFrame(columns=empty_cols), set(ref_tile.index), set(cand_tile.index)

    triples = np.concatenate(all_triples, axis=0)
    del all_triples

    ref_ids_arr = triples[:, 0].astype(int)
    cand_ids_arr = triples[:, 1].astype(int)
    ious_arr = triples[:, 2]
    del triples

    # Greedy 1-to-1 matching (same logic as original)
    order = np.argsort(-ious_arr)
    ref_ids_arr = ref_ids_arr[order]
    cand_ids_arr = cand_ids_arr[order]
    ious_arr = ious_arr[order]
    del order

    used_refs = set()
    used_cands = set()
    match_rows = []

    for ref_id, cand_id, iou in zip(ref_ids_arr, cand_ids_arr, ious_arr):
        if iou < tau_overlap:
            break
        if ref_id in used_refs or cand_id in used_cands:
            continue
        used_refs.add(ref_id)
        used_cands.add(cand_id)

        area_ref = float(ref_tile.loc[ref_id, "area_m2"])
        area_cand = float(cand_tile.loc[cand_id, "area_m2"])
        rel_area_error = (area_cand - area_ref) / area_ref if area_ref > 0 else np.nan

        match_rows.append({
            "ref_id": ref_id, "cand_id": cand_id, "iou": float(iou),
            "area_ref": area_ref, "area_cand": area_cand,
            "rel_area_error": rel_area_error,
        })

    del ref_ids_arr, cand_ids_arr, ious_arr  # FIX: free sorted arrays

    matches_df = pd.DataFrame(match_rows, columns=empty_cols)
    del match_rows
    ref_unmatched = set(ref_tile.index) - used_refs
    cand_unmatched = set(cand_tile.index) - used_cands
    return matches_df, ref_unmatched, cand_unmatched
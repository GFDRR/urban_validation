"""
TODO: contains vector and raster metrics for assessing Building Footprint Datasets for different use cases 
"""
from __future__ import annotations
import os 
import yaml
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
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
logger.addHandler(sh)


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

    P_b: fraction of candidate boundary within tau_boundary of any reference boundary
    R_b: fraction of reference boundary within tau_boundary of any candidate boundary
    F_b: harmonic mean of P_b and R_b
    """
    if matches_df.empty:
        return 0.0

    ref_ids = matches_df["ref_id"].unique()
    cand_ids = matches_df["cand_id"].unique()

    ref_geoms = ref_tile.loc[ref_ids].geometry
    cand_geoms = cand_tile.loc[cand_ids].geometry

    # Union of boundaries (Shapely 2.x: use union_all instead of unary_union)
    ref_bound = ref_geoms.boundary.union_all()
    cand_bound = cand_geoms.boundary.union_all()

    if ref_bound.length == 0 or cand_bound.length == 0:
        return 0.0

    # Buffers around boundaries
    ref_buffer = ref_bound.buffer(tau_boundary_m)
    cand_buffer = cand_bound.buffer(tau_boundary_m)

    # Precision: fraction of candidate boundary near reference boundary
    cand_correct_geom = cand_bound.intersection(ref_buffer)
    cand_correct_len = cand_correct_geom.length
    P_b = cand_correct_len / cand_bound.length if cand_bound.length > 0 else 0.0

    # Recall: fraction of reference boundary near candidate boundary
    ref_correct_geom = ref_bound.intersection(cand_buffer)
    ref_correct_len = ref_correct_geom.length
    R_b = ref_correct_len / ref_bound.length if ref_bound.length > 0 else 0.0

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

    n_ref = len(ref_tile)
    n_cand = len(cand_tile)
    tp = len(matches_df)
    fn = len(ref_unmatched)
    fp = len(cand_unmatched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # IoU distribution (TP-only)
    if tp > 0:
        mean_iou = float(matches_df["iou"].mean())
        median_iou = float(matches_df["iou"].median())
        iou_p25 = _safe_quantile(matches_df["iou"], 0.25)
        iou_p75 = _safe_quantile(matches_df["iou"], 0.75)
    else:
        mean_iou = median_iou = iou_p25 = iou_p75 = 0.0

    # Tile-level boundary F (union-based, for continuity with previous runs)
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

def match_buildings_iou(ref_tile: gpd.GeoDataFrame,
                        cand_tile: gpd.GeoDataFrame,
                        tau_overlap: float,
                        tau_buffer_m: float = 0.0):
    """
    1–1 greedy IoU matching between reference and candidate buildings.

    Rules:
    - Only ref–cand pairs with IoU >= tau_overlap become TP.
    - If a ref building has no candidate with IoU >= tau_overlap -> FN.
    - If a cand building has no ref with IoU >= tau_overlap -> FP.
    - If multiple candidates overlap one ref, the one with highest IoU is TP,
      all others are FP (unmatched cand).
    - If the best IoU for a ref–cand pair is < tau_overlap, the pair contributes
      one FP (cand) and one FN (ref) via the unmatched sets.

    Speed:
    - Uses a spatial join to generate intersecting ref–cand pairs, then computes IoU
      only for those pairs.

    Returns
    -------
    matches_df : DataFrame with one row per TP:
        ['ref_id', 'cand_id', 'iou', 'area_ref', 'area_cand', 'rel_area_error']
    ref_unmatched : set of reference indices (FN)
    cand_unmatched : set of candidate indices (FP)
    """
    empty_cols = ["ref_id", "cand_id", "iou", "area_ref", "area_cand", "rel_area_error"]

    # Edge cases: empty tiles
    if ref_tile.empty and cand_tile.empty:
        return pd.DataFrame(columns=empty_cols), set(), set()
    if ref_tile.empty:
        return pd.DataFrame(columns=empty_cols), set(), set(cand_tile.index)
    if cand_tile.empty:
        return pd.DataFrame(columns=empty_cols), set(ref_tile.index), set()

    # Spatial join to generate candidate pairs (keeps original indices)
    ref_g = ref_tile[["geometry"]].copy()
    cand_g = cand_tile[["geometry"]].copy()

    joined = gpd.sjoin(ref_g, cand_g, how="inner", predicate="intersects")
    if joined.empty:
        return pd.DataFrame(columns=empty_cols), set(ref_tile.index), set(cand_tile.index)

    # joined index = ref index; index_right = cand index
    pairs = joined.reset_index().rename(columns={"index": "ref_id"})[["ref_id", "index_right"]]
    pairs = pairs.rename(columns={"index_right": "cand_id"})

    # Compute IoU per pair
    iou_rows = []
    for ref_id, cand_id in pairs.itertuples(index=False):
        iou = _iou_with_buffer(ref_tile.loc[ref_id].geometry, cand_tile.loc[cand_id].geometry, tau_buffer_m)
        if iou > 0.0:
            iou_rows.append((ref_id, cand_id, iou))

    if not iou_rows:
        return pd.DataFrame(columns=empty_cols), set(ref_tile.index), set(cand_tile.index)

    iou_df = pd.DataFrame(iou_rows, columns=["ref_id", "cand_id", "iou"]).sort_values("iou", ascending=False)

    used_refs = set()
    used_cands = set()
    match_rows = []

    for _, row in iou_df.iterrows():
        ref_id = row["ref_id"]
        cand_id = row["cand_id"]
        iou = float(row["iou"])

        # Because sorted, we can break when IoU goes below threshold
        if iou < tau_overlap:
            break

        if (ref_id in used_refs) or (cand_id in used_cands):
            continue

        used_refs.add(ref_id)
        used_cands.add(cand_id)

        area_ref = float(ref_tile.loc[ref_id, "area_m2"])
        area_cand = float(cand_tile.loc[cand_id, "area_m2"])
        rel_area_error = (area_cand - area_ref) / area_ref if area_ref > 0 else np.nan

        match_rows.append({
            "ref_id": ref_id,
            "cand_id": cand_id,
            "iou": iou,
            "area_ref": area_ref,
            "area_cand": area_cand,
            "rel_area_error": rel_area_error,
        })

    matches_df = pd.DataFrame(match_rows, columns=empty_cols)

    ref_unmatched = set(ref_tile.index) - used_refs
    cand_unmatched = set(cand_tile.index) - used_cands

    return matches_df, ref_unmatched, cand_unmatched

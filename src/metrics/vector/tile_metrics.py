"""
Tile-level vector metric assembly.

Glues together IoU matching and boundary F into the per-tile metrics
dict and returns the matches DataFrame for downstream consolidation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.metrics.vector.boundary import (
    boundary_f_pair,
    compute_boundary_f_for_tile,
)
from src.metrics.vector.matching import match_buildings_iou

logger = logging.getLogger("Validation_Metrics")


def _safe_quantile(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s) else 0.0


def compute_tile_metrics(
    ref_tile,
    city,
    cand_tile,
    tau_overlap,
    tau_buffer_m,
    tau_boundary_m,
    tile_id,
    dataset_name,
    *,
    tile_area_km2: float = np.nan,
):
    matches_df, ref_unmatched, cand_unmatched = match_buildings_iou(
        ref_tile, cand_tile, tau_overlap, tau_buffer_m=tau_buffer_m
    )

    # Per-match boundary F (only for TPs)
    if not matches_df.empty:
        bf_vals = []
        for ref_id, cand_id in matches_df[["ref_id", "cand_id"]].itertuples(index=False):
            bf_vals.append(
                boundary_f_pair(
                    ref_tile.loc[ref_id].geometry,
                    cand_tile.loc[cand_id].geometry,
                    tau_boundary_m,
                )
            )
        matches_df = matches_df.copy()
        matches_df["boundary_f_pair"] = bf_vals
        del bf_vals

    n_ref = len(ref_tile)
    n_cand = len(cand_tile)
    tp = len(matches_df)
    fn = len(ref_unmatched)
    fp = len(cand_unmatched)

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

    boundary_f_union = compute_boundary_f_for_tile(
        ref_tile, cand_tile, matches_df, tau_boundary_m
    )
    boundary_f_meanpair = (
        float(matches_df["boundary_f_pair"].mean())
        if tp > 0 and "boundary_f_pair" in matches_df.columns
        else 0.0
    )

    mean_rel_area_error = float(matches_df["rel_area_error"].mean()) if tp > 0 else np.nan
    area_ref_sum = float(matches_df["area_ref"].sum()) if tp > 0 else 0.0
    area_cand_sum = float(matches_df["area_cand"].sum()) if tp > 0 else 0.0
    signed_area_bias = (
        ((area_cand_sum - area_ref_sum) / area_ref_sum)
        if area_ref_sum > 0
        else np.nan
    )

    # Total areas across ALL buildings in this tile (TP + FN for ref, TP + FP for cand).
    # Used by summarize_city to compute total_area_bias across the full city.
    ref_area_total_m2 = (
        float(ref_tile["area_m2"].sum()) if "area_m2" in ref_tile.columns else np.nan
    )
    cand_area_total_m2 = (
        float(cand_tile["area_m2"].sum()) if "area_m2" in cand_tile.columns else np.nan
    )
    mean_ref_building_area_m2 = (
        ref_area_total_m2 / n_ref if n_ref > 0 and np.isfinite(ref_area_total_m2) else np.nan
    )

    metrics = {
        "city": city,
        "dataset": dataset_name,
        "tile_id": tile_id,
        "tile_area_km2": tile_area_km2,
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
        "ref_area_total_m2": ref_area_total_m2,
        "mean_ref_building_area_m2": mean_ref_building_area_m2,
        "cand_area_total_m2": cand_area_total_m2,
        "tau_overlap": tau_overlap,
        "tau_buffer_m": tau_buffer_m,
        "tau_boundary_m": tau_boundary_m,
    }

    return metrics, matches_df
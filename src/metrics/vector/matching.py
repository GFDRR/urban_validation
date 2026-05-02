"""
Vectorised 1-to-1 IoU matching with chunked spatial join.

Splits the candidate side into chunks before sjoin to bound the peak
memory cost of the vectorised geometry arrays. The greedy 1-to-1
assignment after the chunk loop is unchanged from the original.
"""
from __future__ import annotations

import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

logger = logging.getLogger("Validation_Metrics")

# Tunable: maximum candidate-side rows per sjoin chunk. Caps peak RAM
# of the vectorised geometry arrays.
_SJOIN_CHUNK_SIZE = int(os.environ.get("SJOIN_CHUNK_SIZE", 50_000))


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

    # Chunked sjoin + vectorised IoU
    # Split candidate into chunks to cap peak memory of geometry arrays.
    n_cand = len(cand_g)
    chunk_size = max(_SJOIN_CHUNK_SIZE, 1)
    all_triples = []  # list of (ref_id, cand_id, iou) arrays

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
        union_areas = shapely.area(shapely.union(ref_geoms_arr, cand_geoms_arr))

        # Free bulky geometry arrays now that we only need scalar areas
        del ref_geoms_arr, cand_geoms_arr

        with np.errstate(divide="ignore", invalid="ignore"):
            ious = np.where(union_areas > 0, inter_areas / union_areas, 0.0)

        del inter_areas, union_areas

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

    # Assemble and run greedy matching
    if not all_triples:
        return pd.DataFrame(columns=empty_cols), set(ref_tile.index), set(cand_tile.index)

    triples = np.concatenate(all_triples, axis=0)
    del all_triples

    ref_ids_arr = triples[:, 0].astype(int)
    cand_ids_arr = triples[:, 1].astype(int)
    ious_arr = triples[:, 2]
    del triples

    # Greedy 1-to-1 matching
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

    del ref_ids_arr, cand_ids_arr, ious_arr

    matches_df = pd.DataFrame(match_rows, columns=empty_cols)
    del match_rows
    ref_unmatched = set(ref_tile.index) - used_refs
    cand_unmatched = set(cand_tile.index) - used_cands
    return matches_df, ref_unmatched, cand_unmatched
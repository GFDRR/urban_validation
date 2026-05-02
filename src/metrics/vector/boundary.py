"""
Boundary F-measure for matched building pairs.

boundary_f_pair: per-match score (length-within-tolerance).
compute_boundary_f_for_tile: tile-aggregate score using unioned boundaries.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("Validation_Metrics")


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

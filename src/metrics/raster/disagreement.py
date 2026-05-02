"""
Quantity and allocation disagreement.

Decomposes binary-map error into the portion attributable to a
mismatch in total built-up quantity (quantity disagreement) and the
portion attributable to misplaced built-up pixels (allocation
disagreement). Both are expressed as fractions of valid pixels.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("Validation_Metrics")


def _compute_quantity_allocation_disagreement(
    ref_bin: np.ndarray,
    pred_bin: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float]:
    """
    Pontius-style quantity and allocation disagreement in fractions of valid pixels.
    """
    ref_v = ref_bin[valid].astype(np.uint8)
    pred_v = pred_bin[valid].astype(np.uint8)

    n = len(ref_v)
    if n == 0:
        return np.nan, np.nan

    tp = np.sum((ref_v == 1) & (pred_v == 1))
    tn = np.sum((ref_v == 0) & (pred_v == 0))
    fp = np.sum((ref_v == 0) & (pred_v == 1))
    fn = np.sum((ref_v == 1) & (pred_v == 0))

    # for binary maps
    quantity = abs(fp - fn) / n
    allocation = 2.0 * min(fp, fn) / n
    return float(quantity), float(allocation)
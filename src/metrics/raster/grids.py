"""
Evaluation-grid normalization and native-resolution guard.

  - _normalize_evaluation_grids: ensures every requested grid has
    name / resolution / oversample_factor / all_touched fields.
  - _native_guard_settings: parses guard config defaults.
  - _filter_evaluation_grids_by_native_resolution: enforces the rule
    "do not evaluate finer than the dataset's native support".
  - _empty_raster_tile_row: builds the placeholder row for tiles
    with no valid pixels.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger("Validation_Metrics")


def _normalize_evaluation_grids(
    evaluation_grids: Optional[List[dict]],
    fallback_resolution: Optional[float] = None,
) -> List[dict]:
    """
    Normalize user-provided evaluation grids.

    If evaluation_grids is missing, fall back to a single grid using
    fallback_resolution. If that is also missing, default to 10 m.
    """
    if evaluation_grids:
        out = []
        for i, g in enumerate(evaluation_grids):
            if "resolution" not in g:
                raise ValueError(f"evaluation_grids[{i}] is missing 'resolution'")
            res = float(g["resolution"])
            if res <= 0:
                raise ValueError(f"evaluation_grids[{i}].resolution must be > 0")
            out.append(
                {
                    "name": str(g.get("name", f"{int(res)}m")),
                    "resolution": res,
                    "oversample_factor": g.get("oversample_factor", None),
                    "all_touched": g.get("all_touched", None),
                }
            )
        return out

    # fallback path
    if fallback_resolution is None:
        fallback_resolution = 10.0

    fallback_resolution = float(fallback_resolution)
    if fallback_resolution <= 0:
        raise ValueError("fallback_resolution must be > 0")

    return [
        {
            "name": f"{int(fallback_resolution)}m",
            "resolution": fallback_resolution,
            "oversample_factor": None,
            "all_touched": None,
        }
    ]


def _native_guard_settings(pre_cfg: Optional[dict]) -> dict:
    """
    Default settings for native-resolution guard.
    """
    pre_cfg = pre_cfg or {}
    guard = pre_cfg.get("native_resolution_guard", {}) or {}
    return {
        "enabled": bool(guard.get("enabled", True)),
        "mode": str(guard.get("mode", "skip_finer")).strip().lower(),
        "tolerance_factor": float(guard.get("tolerance_factor", 0.75)),
    }


def _filter_evaluation_grids_by_native_resolution(
    evaluation_grids: List[dict],
    cand_cfg: dict,
    guard_cfg: Optional[dict] = None,
) -> List[dict]:
    """
    Filter or validate requested evaluation grids against the dataset's
    native resolution.

    Rule:
      Do not evaluate at a finer grid than the dataset's native support,
      unless explicitly allowed.

    Parameters
    ----------
    evaluation_grids : list[dict]
        Normalized grids, each containing at least name and resolution.
    cand_cfg : dict
        Candidate raster config. May contain native_resolution_m,
        allow_finer_than_native, name.
    guard_cfg : dict
        Guard behavior config: enabled, mode (skip_finer | error |
        warn_only), tolerance_factor.

    Returns
    -------
    list[dict]
        Filtered list of allowed grids.
    """
    guard_cfg = guard_cfg or {}
    enabled = bool(guard_cfg.get("enabled", True))
    mode = str(guard_cfg.get("mode", "skip_finer")).strip().lower()
    tol = float(guard_cfg.get("tolerance_factor", 0.75))

    if not enabled:
        return evaluation_grids

    if cand_cfg.get("allow_finer_than_native", False):
        return evaluation_grids

    native_res = cand_cfg.get("native_resolution_m", None)
    if native_res is None:
        return evaluation_grids

    native_res = float(native_res)
    if native_res <= 0:
        raise ValueError(
            f"Invalid native_resolution_m={native_res} for raster dataset "
            f"{cand_cfg.get('name', '<unknown>')}"
        )

    min_allowed_eval_res = native_res * tol

    allowed = []
    blocked = []

    for g in evaluation_grids:
        res = float(g["resolution"])
        if res < min_allowed_eval_res:
            blocked.append(g)
        else:
            allowed.append(g)

    if blocked:
        blocked_txt = ", ".join(f"{g['name']} ({g['resolution']} m)" for g in blocked)
        ds_name = cand_cfg.get("name", "<unknown>")
        msg = (
            f"[native-resolution guard] dataset='{ds_name}' "
            f"native_resolution_m={native_res}, tolerance_factor={tol}, "
            f"minimum_allowed_eval_resolution={min_allowed_eval_res:.3f} m. "
            f"Blocked finer evaluation grid(s): {blocked_txt}"
        )

        if mode == "error":
            raise ValueError(msg)
        elif mode == "warn_only":
            logger.warning(msg)
            return evaluation_grids
        elif mode == "skip_finer":
            logger.warning(msg + " | Skipping blocked grid(s).")
        else:
            raise ValueError(
                f"Unknown native_resolution_guard mode={mode!r}. "
                f"Use one of: skip_finer, error, warn_only."
            )

    return allowed


def _empty_raster_tile_row(
    tile_id: int,
    pixel_area: float,
    *,
    grid_name: str,
    resolution: float,
) -> dict:
    """
    Empty row for tiles with no valid pixels.
    """
    return {
        "tile_id": tile_id,
        "grid": grid_name,
        "resolution_m": float(resolution),
        "pixel_area_m2": float(pixel_area),
        "n_pixels": 0,
        "n_valid": 0,
        "valid_area_m2": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "ref_area_m2": np.nan,
        "pred_area_m2": np.nan,
        "rel_area_error": np.nan,
        "signed_area_bias": np.nan,
        "quantity_disagreement": np.nan,
        "allocation_disagreement": np.nan,
    }
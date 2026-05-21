"""
Convert raster values to predicted built-up area (m²) per pixel,
and derive binary built/not-built masks from that area.
"""
from __future__ import annotations

import logging

import numpy as np
from rasterio.transform import Affine

from src.metrics.raster.rasterize import _pixel_area_from_transform

logger = logging.getLogger("Validation_Metrics")


def predicted_area_from_raster(arr: np.ndarray, transform: Affine, spec: dict) -> np.ndarray:
    """Convert raster values to predicted built-up area per pixel (m²).

    Supported spec["method"] values:
      wsf_tracker          — categorical time-bin codes; built if built_value_min ≤ code ≤ as_of_code
      wsf_tracker_fraction — pre-averaged WSF built fraction [0,1]; treated identically to fraction
      fraction             — values in [0, 1]; area = fraction × pixel_area
      percent              — values in [0, 100]; area = (v/100) × pixel_area
      area_m2              — values already in m²; optionally clamped to pixel_area
      binary               — built if value ≥ threshold
      nonzero              — built if value ≠ 0
      value_in             — built if value in spec["values"]
    """
    method = spec.get("method", "fraction")
    pixel_area = _pixel_area_from_transform(transform)
    arr_f = arr.astype("float32")

    if method == "wsf_tracker":
        as_of_code = int(spec.get("as_of_code", 19))
        built_min = int(spec.get("built_value_min", 1))
        nonbuilt = int(spec.get("nonbuilt_value", 0))
        built = (arr != nonbuilt) & (arr >= built_min) & (arr <= as_of_code)
        return built.astype("float32") * pixel_area

    if method == "percent":
        frac = np.clip(
            arr_f,
            float(spec.get("clamp_min", 0.0)),
            float(spec.get("clamp_max", 100.0)),
        ) / 100.0
        return np.clip(frac, 0.0, 1.0) * pixel_area

    if method in {"fraction", "wsf_tracker_fraction"}:
        frac = np.clip(
            arr_f,
            float(spec.get("clamp_min", 0.0)),
            float(spec.get("clamp_max", 1.0)),
        )
        return frac * pixel_area

    if method == "area_m2":
        A = np.maximum(arr_f, 0.0)
        if bool(spec.get("clamp_to_pixel_area", True)):
            A = np.minimum(A, pixel_area)
        return A

    if method == "binary":
        return (arr_f >= float(spec.get("threshold", 1.0))).astype("float32") * pixel_area

    if method == "nonzero":
        return (arr_f != 0).astype("float32") * pixel_area

    if method == "value_in":
        return np.isin(arr, list(spec.get("values", []))).astype("float32") * pixel_area

    raise ValueError(f"Unknown raster binarize method: {method!r}")


def pred_bin_from_pred_area(
    A_pred: np.ndarray,
    transform: Affine,
    spec: dict,
    tau_frac: float,
) -> np.ndarray:
    """Binary built/not-built mask from predicted area.

    Per-dataset threshold_frac overrides the global tau_frac.
    Categorical/binary methods use A_pred > 0 directly.
    """
    pixel_area = _pixel_area_from_transform(transform)
    tau = spec.get("threshold_frac", None)
    if tau is None:
        if spec.get("method", "") in {"wsf_tracker", "binary", "nonzero", "value_in"}:
            return A_pred > 0
        tau = tau_frac
    return (A_pred / pixel_area) >= float(tau)
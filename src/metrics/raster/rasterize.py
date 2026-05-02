"""
Reference-polygon rasterization helpers.

AOI mask, fractional building coverage via oversampling + block average,
grid-aligned transform construction, and pixel area in map units².
"""
from __future__ import annotations

import logging

import numpy as np
from rasterio import features as rio_features
from rasterio.transform import Affine

logger = logging.getLogger("Validation_Metrics")


def _aoi_mask_for_window(
    aoi_geom,
    out_shape: tuple,
    transform: Affine,
    all_touched: bool = False,
) -> np.ndarray:
    """Boolean mask of the AOI geometry rasterized onto the given grid."""
    mask = rio_features.rasterize(
        [(aoi_geom, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    )
    return mask.astype(bool)


def _rasterize_ref_fraction(
    ref_geoms: list,
    out_shape: tuple,
    transform: Affine,
    oversample: int = 4,
    all_touched: bool = False,
) -> np.ndarray:
    """Fractional building coverage per pixel via oversampling.

    Rasterizes reference polygons at `oversample`× resolution then block-averages
    down to `out_shape`, giving a float32 fraction in [0, 1].
    """
    if not ref_geoms:
        return np.zeros(out_shape, dtype="float32")

    if oversample <= 1:
        mask = rio_features.rasterize(
            [(g, 1) for g in ref_geoms],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=all_touched,
        )
        return mask.astype("float32")

    h, w = out_shape
    oh, ow = h * oversample, w * oversample
    hi_transform = transform * Affine.scale(1.0 / oversample, 1.0 / oversample)
    hi = rio_features.rasterize(
        [(g, 1) for g in ref_geoms],
        out_shape=(oh, ow),
        transform=hi_transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    ).astype("float32")
    return hi.reshape(h, oversample, w, oversample).mean(axis=(1, 3))


def _make_grid_aligned_transform(bounds, resolution: float) -> tuple[Affine, tuple[int, int]]:
    """
    Build a north-up transform aligned to a requested resolution for the
    given tile bounds.
    """
    minx, miny, maxx, maxy = bounds
    res = float(resolution)

    x0 = np.floor(minx / res) * res
    y0 = np.floor(miny / res) * res
    x1 = np.ceil(maxx / res) * res
    y1 = np.ceil(maxy / res) * res

    width = max(1, int(round((x1 - x0) / res)))
    height = max(1, int(round((y1 - y0) / res)))

    transform = Affine(res, 0.0, x0, 0.0, -res, y1)
    return transform, (height, width)


def _pixel_area_from_transform(transform: Affine) -> float:
    """Pixel area in map units² (assumes north-up, no rotation)."""
    return float(abs(transform.a * transform.e))

"""
Raster I/O helpers.

Selecting a resampling method, opening a raster in a target CRS via
WarpedVRT, reading a band reprojected onto a destination grid, and
reading a window without boundless reads.
"""
from __future__ import annotations

import logging

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.windows import intersection as win_intersection

logger = logging.getLogger("Validation_Metrics")


def _choose_resampling(binarize_spec: dict) -> Resampling:
    """Nearest for categorical/binary rasters; bilinear for continuous ones."""
    if binarize_spec.get("method", "") in {"wsf_tracker", "value_in", "nonzero", "binary"}:
        return Resampling.nearest
    return Resampling.bilinear


def _open_raster_in_crs(
    src: rasterio.io.DatasetReader,
    target_crs: str,
    binarize_spec: dict,
):
    """Return the dataset (or a WarpedVRT) in target_crs.

    Caller must close the VRT if one is created.
    """
    if src.crs is None:
        raise ValueError("Raster has no CRS.")
    if str(src.crs) == str(target_crs):
        return src
    return WarpedVRT(src, crs=target_crs, resampling=_choose_resampling(binarize_spec))


def _read_reprojected_tile(
    src,
    band: int,
    dst_shape: tuple,
    dst_transform: Affine,
    dst_crs,
    *,
    binarize_spec: dict,
    fill_value: float,
) -> np.ndarray:
    """
    Reproject a raster band into a destination array defined by
    (dst_shape, dst_transform, dst_crs).

    Avoids boundless reads on WarpedVRT and cleanly fills pixels outside
    source extent with fill_value.
    """
    dst = np.full(dst_shape, fill_value, dtype="float32")

    src_nodata = src.nodata
    if "nodata" in binarize_spec:
        src_nodata = binarize_spec["nodata"]

    reproject(
        source=rasterio.band(src, band),
        destination=dst,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=fill_value,
        resampling=_choose_resampling(binarize_spec),
    )
    return dst


def _read_window_padded(
    src,
    band: int,
    window: Window,
    out_shape: tuple,
    fill_value: float,
) -> np.ndarray:
    """Read a raster window without boundless reads, zero-padding out-of-bounds areas."""
    full = Window(0, 0, src.width, src.height)
    try:
        w_int = win_intersection(window, full)
    except Exception:
        return np.full(out_shape, fill_value, dtype="float32")

    out = np.full(out_shape, fill_value, dtype="float32")
    data = src.read(band, window=w_int, boundless=False).astype("float32")
    row_off = max(0, int(round(w_int.row_off - window.row_off)))
    col_off = max(0, int(round(w_int.col_off - window.col_off)))
    h, w = data.shape
    # Clamp to avoid overflow when FP rounding makes data slightly larger than out_shape
    h_clip = min(h, out_shape[0] - row_off)
    w_clip = min(w, out_shape[1] - col_off)
    if h_clip > 0 and w_clip > 0:
        out[row_off: row_off + h_clip, col_off: col_off + w_clip] = data[:h_clip, :w_clip]
    return out
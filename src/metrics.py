"""
contains vector and raster metrics for assessing Building Footprint Datasets for different use cases

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
from typing import List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import box
import duckdb
import rasterio
from rasterio import features as rio_features
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import from_bounds, Window
from rasterio.windows import intersection as win_intersection
import rasterio.windows as riowin

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

# tunable: maximum candidate-side rows per sjoin chunk: keeps peak RAM of the vectorised geometry arrays bounded.
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

    del ref_ids_arr, cand_ids_arr, ious_arr  # FIX: free sorted arrays

    matches_df = pd.DataFrame(match_rows, columns=empty_cols)
    del match_rows
    ref_unmatched = set(ref_tile.index) - used_refs
    cand_unmatched = set(cand_tile.index) - used_cands
    return matches_df, ref_unmatched, cand_unmatched


# Raster validation
def _choose_resampling(binarize_spec: dict) -> Resampling:
    """Nearest for categorical/binary rasters; bilinear for continuous ones."""
    if binarize_spec.get("method", "") in {"wsf_tracker", "value_in", "nonzero", "binary"}:
        return Resampling.nearest
    return Resampling.bilinear


def _open_raster_in_crs(src: rasterio.io.DatasetReader, target_crs: str, binarize_spec: dict):
    """Return the dataset (or a WarpedVRT) in target_crs. Caller must close the VRT if one is created."""
    if src.crs is None:
        raise ValueError("Raster has no CRS.")
    if str(src.crs) == str(target_crs):
        return src
    return WarpedVRT(src, crs=target_crs, resampling=_choose_resampling(binarize_spec))


def _aoi_mask_for_window(aoi_geom, out_shape: tuple, transform: Affine,
                          all_touched: bool = False) -> np.ndarray:
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


def _rasterize_ref_fraction(ref_geoms: list, out_shape: tuple, transform: Affine,
                              oversample: int = 4, all_touched: bool = False) -> np.ndarray:
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


def _read_window_padded(src, band: int, window: Window, out_shape: tuple,
                         fill_value: float) -> np.ndarray:
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


def _pixel_area_from_transform(transform: Affine) -> float:
    """Pixel area in map units² (assumes north-up, no rotation)."""
    return float(abs(transform.a * transform.e))


def predicted_area_from_raster(arr: np.ndarray, transform: Affine, spec: dict) -> np.ndarray:
    """Convert raster values to predicted built-up area per pixel (m²).

    Supported spec["method"] values:
      wsf_tracker  — categorical time-bin codes; built if built_value_min ≤ code ≤ as_of_code
      fraction     — values in [0, 1]; area = fraction × pixel_area
      percent      — values in [0, 100]; area = (v/100) × pixel_area
      area_m2      — values already in m²; optionally clamped to pixel_area
      binary       — built if value ≥ threshold
      nonzero      — built if value ≠ 0
      value_in     — built if value in spec["values"]
    """
    method = spec.get("method", "fraction")
    pixel_area = _pixel_area_from_transform(transform)
    arr_f = arr.astype("float32")

    if method == "wsf_tracker":
        as_of_code = int(spec.get("as_of_code", 19))
        built_min  = int(spec.get("built_value_min", 1))
        nonbuilt   = int(spec.get("nonbuilt_value", 0))
        built = (arr != nonbuilt) & (arr >= built_min) & (arr <= as_of_code)
        return built.astype("float32") * pixel_area

    if method == "percent":
        frac = np.clip(arr_f, float(spec.get("clamp_min", 0.0)),
                       float(spec.get("clamp_max", 100.0))) / 100.0
        return np.clip(frac, 0.0, 1.0) * pixel_area

    if method == "fraction":
        frac = np.clip(arr_f, float(spec.get("clamp_min", 0.0)),
                       float(spec.get("clamp_max", 1.0)))
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


def pred_bin_from_pred_area(A_pred: np.ndarray, transform: Affine,
                             spec: dict, tau_frac: float) -> np.ndarray:
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


def compute_raster_tile_metrics(
    raster_path: Path,
    cand_cfg: dict,
    ref_all: gpd.GeoDataFrame,
    ref_sindex,
    aoi_union,
    tiles: gpd.GeoDataFrame,
    tau_frac: float,
    default_oversample: int = 4,
    default_all_touched: bool = False,
) -> pd.DataFrame:
    """Tile-level raster validation for one candidate dataset.

    For each tile:
      1. Read the candidate raster window.
      2. Rasterize reference polygons onto the candidate grid (fractional coverage).
      3. Compute binary & area-based accuracy metrics.

    Returns a DataFrame with one row per evaluated tile.
    """
    bin_spec    = cand_cfg.get("binarize", {"method": "fraction"})
    rast_over   = cand_cfg.get("rasterization", {}) or {}
    oversample  = int(rast_over.get("oversample_factor", default_oversample))
    all_touched = bool(rast_over.get("all_touched", default_all_touched))
    band        = int(bin_spec.get("band", 1))

    rows: List[dict] = []

    with rasterio.open(raster_path) as _src:
        ds = _open_raster_in_crs(_src, str(tiles.crs), bin_spec)

        # nodata = bin_spec["nodata"] if "nodata" in bin_spec else ds.nodata
        nodata = bin_spec.get("nodata", None)

        for tile_row in tiles.itertuples():
            tile_id  = int(tile_row.tile_id)
            geom     = tile_row.geometry
            minx, miny, maxx, maxy = geom.bounds

            try:
                win = from_bounds(minx, miny, maxx, maxy, transform=ds.transform)
                if win.width <= 0 or win.height <= 0:
                    continue

                out_shape  = (int(round(win.height)), int(round(win.width)))
                # fill       = float(nodata) if nodata is not None else 0.0
                fill = float(nodata) if nodata is not None else np.nan
                arr        = _read_window_padded(ds, band, win, out_shape, fill_value=fill)
                transform  = riowin.transform(win, ds.transform)
                pixel_area = _pixel_area_from_transform(transform)

                # Valid pixels: inside AOI and not nodata
                aoi_mask = _aoi_mask_for_window(aoi_union, arr.shape, transform,
                                                all_touched=all_touched)
                valid = aoi_mask.copy()
                method = bin_spec.get("method", "")

                if nodata is not None:
                    if method in {"fraction", "percent", "area_m2"}:
                        # Do NOT mask zeros — they are valid
                        valid &= np.isfinite(arr)
                    else:
                        valid &= (arr != nodata)

                n_valid = int(valid.sum())
                if n_valid == 0:
                    rows.append(_empty_raster_tile_row(tile_id, pixel_area))
                    continue

                # Prediction
                A_pred   = predicted_area_from_raster(arr, transform, bin_spec)
                pred_bin = pred_bin_from_pred_area(A_pred, transform, bin_spec, tau_frac)

                # Reference rasterization
                possible  = list(ref_sindex.intersection(geom.bounds))
                ref_tile  = ref_all.iloc[possible]
                ref_tile  = ref_tile[ref_tile.intersects(geom)]
                ref_geoms = list(ref_tile.geometry)

                f_ref    = _rasterize_ref_fraction(ref_geoms, arr.shape, transform,
                                                   oversample=oversample, all_touched=all_touched)
                A_ref    = f_ref * pixel_area
                ref_bin  = (f_ref >= tau_frac)

                # Confusion matrix on valid pixels
                p  = pred_bin[valid]
                r  = ref_bin[valid]
                tp = int(np.logical_and(p, r).sum())
                fp = int(np.logical_and(p, ~r).sum())
                fn = int(np.logical_and(~p, r).sum())

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1        = (2 * precision * recall / (precision + recall)
                             if (precision + recall) > 0 else 0.0)

                # Area-based metrics
                area_ref  = float(A_ref[valid].sum())
                area_pred = float(A_pred[valid].sum())
                rel_err   = abs(area_pred - area_ref) / area_ref if area_ref > 0 else np.nan
                bias      = area_pred - area_ref

                # Pontius-style disagreement (binary)
                n_pred = int(p.sum())
                n_ref  = int(r.sum())
                qd     = abs(n_pred - n_ref) / n_valid
                ad     = (2 * min(fp, fn)) / n_valid

                rows.append({
                    "tile_id":                 tile_id,
                    "tp":                      tp,
                    "fp":                      fp,
                    "fn":                      fn,
                    "precision":               precision,
                    "recall":                  recall,
                    "f1":                      f1,
                    "quantity_disagreement":   float(qd),
                    "allocation_disagreement": float(ad),
                    "n_valid_pixels":          n_valid,
                    "pixel_area_m2":           float(pixel_area),
                    "valid_area_m2":           float(n_valid) * float(pixel_area),
                    "area_ref_m2":             area_ref,
                    "area_pred_m2":            area_pred,
                    "rel_area_error":          rel_err,
                    "signed_area_bias_m2":     bias,
                })

                del arr, A_pred, pred_bin, f_ref, A_ref, ref_bin, aoi_mask, valid, p, r

            except Exception as exc:
                logger.warning(
                    "Tile %d skipped due to error: %s: %s",
                    tile_id, type(exc).__name__, exc,
                )
                continue

    return pd.DataFrame(rows)


def _empty_raster_tile_row(tile_id: int, pixel_area: float) -> dict:
    return {
        "tile_id":                 tile_id,
        "tp":                      0,
        "fp":                      0,
        "fn":                      0,
        "precision":               0.0,
        "recall":                  0.0,
        "f1":                      0.0,
        "quantity_disagreement":   0.0,
        "allocation_disagreement": 0.0,
        "n_valid_pixels":          0,
        "pixel_area_m2":           float(pixel_area),
        "valid_area_m2":           0.0,
        "area_ref_m2":             0.0,
        "area_pred_m2":            0.0,
        "rel_area_error":          np.nan,
        "signed_area_bias_m2":     0.0,
    }
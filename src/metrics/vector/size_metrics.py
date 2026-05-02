"""
Per-size-bin metrics and per-city density summary for vector validation.

Two outputs that are not currently first-class in the pipeline:

  1. Per (city, dataset, size_bin): match counts, IoU / boundary-F /
     rel_area_error stats, plus per-bin precision and recall using
     ref-side and cand-side bin totals.

  2. Per (city, source): building counts, mean / median / quartile
     building sizes, AOI area, and density (buildings per km²),
     where `source` is the reference or any candidate dataset.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger("Validation_Metrics")

# Defaults match plot_iou_per_building_sizes in src/plots/output.py
_DEFAULT_SIZE_BINS = [0, 25, 50, 100, 500, 1000, np.inf]
_DEFAULT_SIZE_LABELS = ["<25", "25–50", "50–100", "100–500", "500–1000", ">1000"]


def _bin_areas(
    areas: pd.Series,
    size_bins: List[float],
    size_bin_labels: List[str],
) -> pd.Series:
    """Bin a Series of building areas into a Categorical of bin labels."""
    return pd.cut(
        areas.astype(float),
        bins=size_bins,
        labels=size_bin_labels,
        include_lowest=True,
    )


def _count_by_bin(
    areas: pd.Series,
    size_bins: List[float],
    size_bin_labels: List[str],
) -> pd.Series:
    """Return a Series indexed by bin label with counts."""
    if areas is None or len(areas) == 0:
        return pd.Series(0, index=pd.Index(size_bin_labels, name="size_bin"), dtype=int)
    binned = _bin_areas(areas, size_bins, size_bin_labels)
    counts = binned.value_counts(dropna=False).reindex(size_bin_labels, fill_value=0)
    counts.index.name = "size_bin"
    return counts.astype(int)


def compute_size_bin_metrics(
    matches_df: pd.DataFrame,
    *,
    ref_all: gpd.GeoDataFrame,
    cand_all: gpd.GeoDataFrame,
    dataset_id: str,
    ds_name: str,
    size_bins: Optional[List[float]] = None,
    size_bin_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute per (city, dataset, size_bin) metrics for one candidate dataset.

    The size bin is defined on the *reference* building area for matches
    and for ref-side counts; for candidate-side counts the bin is defined
    on the candidate building area. This means ``recall_in_bin`` is the
    fraction of ref buildings of size S that were matched, and
    ``precision_in_bin`` is the fraction of cand buildings of size S
    that were matched.

    Returns columns:
      city, dataset, size_bin,
      n_matches, mean_iou, median_iou,
      mean_rel_area_error, mean_boundary_f,
      n_ref_in_bin, n_cand_in_bin,
      recall_in_bin, precision_in_bin
    """
    if size_bins is None:
        size_bins = _DEFAULT_SIZE_BINS
    if size_bin_labels is None:
        size_bin_labels = _DEFAULT_SIZE_LABELS

    # Reference / candidate totals per bin (denominators)
    ref_counts = _count_by_bin(
        ref_all["area_m2"] if (ref_all is not None and "area_m2" in ref_all.columns) else pd.Series(dtype=float),
        size_bins,
        size_bin_labels,
    )
    cand_counts = _count_by_bin(
        cand_all["area_m2"] if (cand_all is not None and "area_m2" in cand_all.columns) else pd.Series(dtype=float),
        size_bins,
        size_bin_labels,
    )

    # Match-side stats per bin (numerators + IoU / boundary / area-error)
    if matches_df is None or matches_df.empty:
        ref_match_counts = pd.Series(
            0, index=pd.Index(size_bin_labels, name="size_bin"), dtype=int
        )
        cand_match_counts = ref_match_counts.copy()
        match_stats = pd.DataFrame(
            {
                "mean_iou": np.nan,
                "median_iou": np.nan,
                "mean_rel_area_error": np.nan,
                "mean_boundary_f": np.nan,
            },
            index=pd.Index(size_bin_labels, name="size_bin"),
        )
    else:
        m = matches_df.copy()
        m["size_bin_ref"] = _bin_areas(m["area_ref"], size_bins, size_bin_labels)
        m["size_bin_cand"] = _bin_areas(m["area_cand"], size_bins, size_bin_labels)

        ref_match_counts = (
            m["size_bin_ref"].value_counts(dropna=False)
            .reindex(size_bin_labels, fill_value=0)
            .astype(int)
        )
        ref_match_counts.index.name = "size_bin"

        cand_match_counts = (
            m["size_bin_cand"].value_counts(dropna=False)
            .reindex(size_bin_labels, fill_value=0)
            .astype(int)
        )
        cand_match_counts.index.name = "size_bin"

        agg_kwargs = {
            "mean_iou": ("iou", "mean"),
            "median_iou": ("iou", "median"),
            "mean_rel_area_error": ("rel_area_error", "mean"),
        }
        if "boundary_f_pair" in m.columns:
            agg_kwargs["mean_boundary_f"] = ("boundary_f_pair", "mean")

        match_stats = (
            m.groupby("size_bin_ref", observed=False)
            .agg(**agg_kwargs)
            .reindex(size_bin_labels)
        )
        match_stats.index.name = "size_bin"
        if "mean_boundary_f" not in match_stats.columns:
            match_stats["mean_boundary_f"] = np.nan

    # Assemble per-bin output
    rows: List[dict] = []
    for label in size_bin_labels:
        n_ref_b = int(ref_counts.get(label, 0))
        n_cand_b = int(cand_counts.get(label, 0))
        n_matches_ref = int(ref_match_counts.get(label, 0))
        n_matches_cand = int(cand_match_counts.get(label, 0))

        recall = (n_matches_ref / n_ref_b) if n_ref_b > 0 else np.nan
        precision = (n_matches_cand / n_cand_b) if n_cand_b > 0 else np.nan

        rows.append(
            {
                "city": dataset_id,
                "dataset": ds_name,
                "size_bin": label,
                "n_matches": n_matches_ref,
                "mean_iou": float(match_stats.loc[label, "mean_iou"])
                if not pd.isna(match_stats.loc[label, "mean_iou"]) else np.nan,
                "median_iou": float(match_stats.loc[label, "median_iou"])
                if not pd.isna(match_stats.loc[label, "median_iou"]) else np.nan,
                "mean_rel_area_error": float(match_stats.loc[label, "mean_rel_area_error"])
                if not pd.isna(match_stats.loc[label, "mean_rel_area_error"]) else np.nan,
                "mean_boundary_f": float(match_stats.loc[label, "mean_boundary_f"])
                if not pd.isna(match_stats.loc[label, "mean_boundary_f"]) else np.nan,
                "n_ref_in_bin": n_ref_b,
                "n_cand_in_bin": n_cand_b,
                "recall_in_bin": recall,
                "precision_in_bin": precision,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------
# City-level density and average building size summary
# -----------------------------------------------------------------

def _area_stats(areas: pd.Series) -> Dict[str, float]:
    """Mean / median / p25 / p75 of building areas, NaN-safe."""
    if areas is None or len(areas) == 0:
        return {
            "mean_area_m2": np.nan,
            "median_area_m2": np.nan,
            "p25_area_m2": np.nan,
            "p75_area_m2": np.nan,
        }
    a = areas.astype(float)
    return {
        "mean_area_m2": float(a.mean()),
        "median_area_m2": float(a.median()),
        "p25_area_m2": float(a.quantile(0.25)),
        "p75_area_m2": float(a.quantile(0.75)),
    }


def aoi_area_km2(aoi_proj: gpd.GeoDataFrame) -> float:
    """Total area (km²) of the projected AOI."""
    if aoi_proj is None or aoi_proj.empty:
        return float("nan")
    return float(aoi_proj.geometry.area.sum() / 1_000_000.0)


def compute_city_density_summary(
    *,
    dataset_id: str,
    aoi_area_km2_value: float,
    ref_areas: pd.Series,
    cand_areas: Dict[str, pd.Series],
) -> pd.DataFrame:
    """Build the per (city, source) density and average-size summary.

    Parameters
    ----------
    dataset_id : str
        City / dataset identifier.
    aoi_area_km2_value : float
        Dissolved AOI area in km², used as denominator for density.
    ref_areas : pd.Series
        Reference building ``area_m2`` values (post-min-area filter).
    cand_areas : dict[str, pd.Series]
        Mapping of candidate dataset name -> Series of ``area_m2``.

    Returns
    -------
    DataFrame with one row per source (reference + each candidate):
        city, source, n_buildings, aoi_area_km2,
        density_per_km2,
        mean_area_m2, median_area_m2, p25_area_m2, p75_area_m2
    """
    rows: List[dict] = []

    def _row(source: str, areas: pd.Series) -> dict:
        n = int(len(areas)) if areas is not None else 0
        density = (n / aoi_area_km2_value) if aoi_area_km2_value and aoi_area_km2_value > 0 else np.nan
        stats = _area_stats(areas)
        return {
            "city": dataset_id,
            "source": source,
            "n_buildings": n,
            "aoi_area_km2": float(aoi_area_km2_value) if aoi_area_km2_value is not None else np.nan,
            "density_per_km2": float(density) if not pd.isna(density) else np.nan,
            **stats,
        }

    rows.append(_row("reference", ref_areas))
    for ds_name, areas in cand_areas.items():
        rows.append(_row(ds_name, areas))

    return pd.DataFrame(rows)

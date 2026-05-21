import datetime
import gc
from pathlib import Path

import yaml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns

from src.utils.aggregation import aggregate_weighted

try:
    from matplotlib._pylab_helpers import Gcf
except ImportError:
    Gcf = None

# Default building-size bins (m²)
_DEFAULT_SIZE_BINS   = [0, 25, 50, 100, 500, 1000, np.inf]
_DEFAULT_SIZE_LABELS = ["<25", "25–50", "50–100", "100–500", "500–1000", ">1000"]

def load_config(path: str | Path) -> dict:
    with open(path, "r") as fp:
        return yaml.safe_load(fp)

def fig_name(city: str, stem: str, ext: str = "png") -> str:
    """Return a timestamped filename: <city>_<stem>_<YYYYMMDD_HHMMSS>.<ext>"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{city.lower()}_{stem}_{ts}.{ext}"


def save_figure(fig: plt.Figure, figures_dir: Path, filename: str, dpi: int = 200) -> None:
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / filename, dpi=dpi, bbox_inches="tight")

def summarize_city(
    city: str,
    metrics_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    aoi_area_km2: float = float("nan"),
) -> pd.DataFrame:
    """Aggregate tile-level metrics and match statistics into one row per dataset."""
    rows = []
    for ds, mds in metrics_df.groupby("dataset"):
        tp     = int(mds["tp"].sum())
        fp     = int(mds["fp"].sum())
        fn     = int(mds["fn"].sum())
        n_ref  = int(mds["n_ref"].sum())
        n_cand = int(mds["n_cand"].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        count_delta_total = n_cand - n_ref
        count_ratio_total = n_cand / n_ref if n_ref > 0 else float("nan")
        rel_count_delta_total = count_delta_total / n_ref if n_ref > 0 else float("nan")
        ref_density_per_km2 = n_ref / aoi_area_km2 if np.isfinite(aoi_area_km2) and aoi_area_km2 > 0 else float("nan")
        cand_density_per_km2 = n_cand / aoi_area_km2 if np.isfinite(aoi_area_km2) and aoi_area_km2 > 0 else float("nan")
        density_delta_per_km2 = (
            cand_density_per_km2 - ref_density_per_km2
            if np.isfinite(cand_density_per_km2) and np.isfinite(ref_density_per_km2)
            else float("nan")
        )
        density_ratio_per_km2 = (
            cand_density_per_km2 / ref_density_per_km2
            if np.isfinite(cand_density_per_km2) and np.isfinite(ref_density_per_km2) and ref_density_per_km2 > 0
            else float("nan")
        )

        dsmatches = (
            matches_df[matches_df["dataset"] == ds]
            if not matches_df.empty else pd.DataFrame()
        )
        if not dsmatches.empty:
            ious     = dsmatches["iou"].astype(float)
            bf       = (dsmatches["boundary_f_pair"].astype(float)
                        if "boundary_f_pair" in dsmatches.columns
                        else pd.Series(dtype=float))
            rel_area = dsmatches["rel_area_error"].astype(float)

            iou_mean    = float(ious.mean())
            iou_median  = float(ious.median())
            iou_p25     = float(ious.quantile(0.25))
            iou_p75     = float(ious.quantile(0.75))
            bf_mean     = float(bf.mean()) if len(bf) else 0.0

            rel_area_mean   = float(rel_area.mean())
            rel_area_median = float(rel_area.median())

            area_ref_sum  = float(dsmatches["area_ref"].sum())
            area_cand_sum = float(dsmatches["area_cand"].sum())
            signed_area_bias = (
                (area_cand_sum - area_ref_sum) / area_ref_sum
                if area_ref_sum > 0 else float("nan")
            )
        else:
            iou_mean = iou_median = iou_p25 = iou_p75 = bf_mean = 0.0
            rel_area_mean = rel_area_median = signed_area_bias = float("nan")

        def _r(v):
            return round(v, 4) if not np.isnan(v) else float("nan")

        rows.append({
            "city":                     city,
            "dataset":                  ds,
            "n_sub_areas":              int(mds["sub_area_id"].nunique()) if "sub_area_id" in mds.columns else 1,
            "n_tiles":                  int(mds["tile_id"].nunique()),
            "n_ref_total":              n_ref,
            "n_cand_total":             n_cand,
            "count_delta_total":        count_delta_total,
            "count_ratio_total":        count_ratio_total,
            "rel_count_delta_total":    rel_count_delta_total,
            "ref_density_per_km2":      ref_density_per_km2,
            "cand_density_per_km2":     cand_density_per_km2,
            "density_delta_per_km2":    density_delta_per_km2,
            "density_ratio_per_km2":    density_ratio_per_km2,
            "tp_total":                 tp,
            "fp_total":                 fp,
            "fn_total":                 fn,
            "precision_city":           _r(precision),
            "recall_city":              _r(recall),
            "f1_city":                  _r(f1),
            "iou_mean_tp":              _r(iou_mean),
            "iou_median_tp":            _r(iou_median),
            "iou_p25_tp":               _r(iou_p25),
            "iou_p75_tp":               _r(iou_p75),
            "boundary_f_meanpair_tp":   _r(bf_mean),
            "rel_area_error_mean_tp":   _r(rel_area_mean),
            "rel_area_error_median_tp": _r(rel_area_median),
            "signed_area_bias_tp":      _r(signed_area_bias),
        })

    return pd.DataFrame(rows)


def plot_vector_count_summary(
    city_summary: pd.DataFrame,
    figures_dir: Path,
    city: str,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Compare reference and candidate building counts at city level."""
    if city_summary is None or city_summary.empty:
        return

    plot_df = city_summary.copy().sort_values("dataset")
    labels = plot_df["dataset"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    axes[0].bar(x - width / 2, plot_df["n_ref_total"], width=width, label="Reference", color="slateblue")
    axes[0].bar(x + width / 2, plot_df["n_cand_total"], width=width, label="Candidate", color="darkorange")
    axes[0].set_title(f"{city} — City-level building counts", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Building count")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].bar(x, plot_df["count_delta_total"], color="steelblue")
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.9)
    axes[1].set_title("Candidate minus reference", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count delta")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    for idx, value in enumerate(plot_df["rel_count_delta_total"]):
        if pd.notna(value):
            axes[1].text(idx, plot_df["count_delta_total"].iloc[idx], f"{value:.0%}", ha="center", va="bottom", fontsize=8)

    sns.despine()
    fig.tight_layout()
    save_figure(fig, figures_dir, fig_name(city, "vector_building_counts", fmt), dpi=dpi)
    plt.close(fig)


def tile_f1_box_plot(
    metrics_all: pd.DataFrame,
    figures_dir: Path,
    city: str,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Box plot of per-tile F1 scores, one box per dataset."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(
        data=metrics_all, x="dataset", y="f1", hue="dataset", ax=ax,
        palette="Set2", linewidth=0.8, legend=False,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    ax.set_title(f"Tile-level F1 scores — {city}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    fig.tight_layout()
    save_figure(fig, figures_dir, fig_name(city, "tile_f1_boxplot", fmt), dpi=dpi)
    plt.close(fig)

def tile_f1_spatial_dist(
    tiles: gpd.GeoDataFrame,
    metrics_all: pd.DataFrame,
    figures_dir: Path,
    city: str,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Choropleth map of per-tile F1, one figure per dataset."""
    for ds in metrics_all["dataset"].unique():
        metrics_ds = metrics_all[metrics_all["dataset"] == ds]
        if metrics_ds.empty:
            continue
        tiles_metrics = tiles.merge(metrics_ds[["tile_id", "f1"]], on="tile_id", how="left")
        del metrics_ds
        # Reproject to WGS-84 for plotting. GeoPandas' default aspect="auto" uses cos(latitude°) on geographic CRS; if you plot projected metres
        # without reprojecting (or CRS/geometry disagree), y is wrong and you get ValueError: aspect must be finite and positive.
        tiles_plot = tiles_metrics.to_crs("EPSG:4326")
        del tiles_metrics
        bounds = tiles_plot.total_bounds
        if not np.all(np.isfinite(bounds)):
            del tiles_plot
            continue
        fig, ax = plt.subplots(figsize=(8, 8))
        tiles_plot.plot(
            column="f1", ax=ax, legend=True, cmap="viridis",
            vmin=0, vmax=1, edgecolor="none",
            legend_kwds={"label": "F1 score", "shrink": 0.6},
            aspect="equal",
        )
        del tiles_plot
        ax.set_title(f"{city} — Tile-level F1 — {ds}", fontsize=13, fontweight="bold")
        ax.set_axis_off()
        fig.tight_layout()
        save_figure(fig, figures_dir, fig_name(city, f"tile_f1_spatial_{ds}", fmt), dpi=dpi)
        plt.close(fig)
        gc.collect()


def plot_iou_dist(
    metrics_all: pd.DataFrame,
    matches_all: pd.DataFrame,
    figures_dir: Path,
    city: str,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """IoU histogram per dataset. TP pairs use real IoU; FP+FN contribute zero."""
    for ds in metrics_all["dataset"].unique():
        m_ds = metrics_all[metrics_all["dataset"] == ds]
        tp   = int(m_ds["tp"].sum())
        fp   = int(m_ds["fp"].sum())
        fn   = int(m_ds["fn"].sum())

        ious_tp  = matches_all[matches_all["dataset"] == ds]["iou"].dropna()
        ious_all = pd.concat(
            [ious_tp, pd.Series(np.zeros(fp + fn))], ignore_index=True
        )
        del ious_tp

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(ious_all, bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axvline(
            float(ious_all.mean()), color="tomato", linestyle="--", linewidth=1.2,
            label=f"Mean IoU = {float(ious_all.mean()):.3f}",
        )
        ax.set_title(
            f"{city} — IoU distribution (TP matched + FP/FN as 0) — {ds}",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("IoU")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        sns.despine()
        fig.tight_layout()
        save_figure(fig, figures_dir, fig_name(city, f"iou_dist_{ds}", fmt), dpi=dpi)
        plt.close(fig)
        del ious_all


def plot_iou_per_building_sizes(
    matches_all: pd.DataFrame,
    figures_dir: Path,
    city: str,
    size_bins: list = _DEFAULT_SIZE_BINS,
    size_bin_labels: list = _DEFAULT_SIZE_LABELS,
    use_explicit_bins: bool = True,
    n_quantile_bins: int = 5,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Median IoU and median relative area error vs reference building size class."""
    for ds in matches_all["dataset"].unique():
        m_ds = matches_all[matches_all["dataset"] == ds].copy()
        if m_ds.empty:
            continue

        m_ds["size_bin"] = (
            pd.cut(m_ds["area_ref"], bins=size_bins, labels=size_bin_labels, include_lowest=True)
            if use_explicit_bins
            else pd.qcut(m_ds["area_ref"], q=n_quantile_bins, duplicates="drop")
        )

        size_stats = (
            m_ds.groupby("size_bin", observed=True)
            .agg(
                mean_iou=("iou", "mean"),
                median_iou=("iou", "median"),
                median_rel_area_error=("rel_area_error", "median"),
                count=("iou", "size"),
            )
            .reset_index()
        )
        del m_ds
        x = size_stats["size_bin"].astype(str).tolist()

        for col, colour, ylabel, stem in [
            ("median_iou",            "steelblue",  "Median IoU",                  f"iou_by_size_{ds}"),
            ("median_rel_area_error", "darkorange", "Median relative area error",  f"area_error_by_size_{ds}"),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(x, size_stats[col], marker="o", color=colour, linewidth=1.8)
            if col == "median_iou":
                ax.set_ylim(0, 1)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlabel("Reference building footprint area (m²)", fontsize=11)
            ax.set_title(f"{city} — {ylabel} by building size — {ds}", fontsize=12, fontweight="bold")
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
            ax.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=30, ha="right")
            sns.despine()
            fig.tight_layout()
            save_figure(fig, figures_dir, fig_name(city, stem, fmt), dpi=dpi)
            plt.close(fig)
        del size_stats


def plot_raster_count_summary(
    city_summary: pd.DataFrame,
    figures_dir: Path,
    city: str,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Compare predicted and reference building counts for raster outputs."""
    if city_summary is None or city_summary.empty:
        return

    plot_df = city_summary.copy().sort_values([c for c in ["dataset", "grid", "resolution_m"] if c in city_summary.columns])
    labels = plot_df.apply(
        lambda row: " | ".join(
            str(row[col]) for col in ["dataset", "grid"] if col in plot_df.columns and pd.notna(row[col])
        ),
        axis=1,
    ).tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    axes[0].bar(x - width / 2, plot_df["ref_building_count"], width=width, label="Reference", color="slateblue")
    axes[0].bar(x + width / 2, plot_df["pred_building_count"], width=width, label="Predicted", color="darkorange")
    axes[0].set_title(f"{city} — Raster building counts", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Building count")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].bar(x, plot_df["delta_building_count"], color="steelblue")
    axes[1].axhline(0, color="grey", linestyle="--", linewidth=0.9)
    axes[1].set_title("Predicted minus reference", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Count delta")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)

    for idx, value in enumerate(plot_df["rel_delta_building_count"]):
        if pd.notna(value):
            axes[1].text(idx, plot_df["delta_building_count"].iloc[idx], f"{value:.0%}", ha="center", va="bottom", fontsize=8)

    sns.despine()
    fig.tight_layout()
    save_figure(fig, figures_dir, fig_name(city, "raster_building_counts", fmt), dpi=dpi)
    plt.close(fig)

# raster output 
def summarize_raster_city(
    city: str,
    metrics_tiles: pd.DataFrame,
    *,
    n_ref_buildings: int = 0,
    aoi_area_km2: float = float("nan"),
    buildings_per_km2: float = float("nan"),
    avg_building_size_m2: float = float("nan"),
) -> pd.DataFrame:
    """
    Aggregate tile-level raster metrics into one summary row per dataset and grid.
    """
    rows = []

    group_cols = ["dataset"]
    if "grid" in metrics_tiles.columns:
        group_cols.append("grid")
    if "resolution_m" in metrics_tiles.columns:
        group_cols.append("resolution_m")

    weighted_cols = [
        "precision",
        "recall",
        "f1",
        "rel_area_error",
        "signed_area_bias",
        "quantity_disagreement",
        "allocation_disagreement",
    ]
    weighted_lookup = {}
    weighted_df = aggregate_weighted(
        metrics_tiles,
        metric_cols=weighted_cols,
        weight_col="valid_area_m2",
        groupby_cols=group_cols,
    )
    if not weighted_df.empty:
        weighted_lookup = {
            tuple(row[col] for col in group_cols): row
            for _, row in weighted_df.iterrows()
        }

    for keys, g in metrics_tiles.groupby(group_cols):
        if isinstance(keys, tuple):
            key_map = dict(zip(group_cols, keys))
        else:
            key_map = {group_cols[0]: keys}
        key_tuple = tuple(key_map[col] for col in group_cols)
        weighted_row = weighted_lookup.get(key_tuple)

        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        fn = int(g["fn"].sum())

        valid_area_total_m2 = float(g["valid_area_m2"].sum())
        tp_m2 = float((g["tp"] * g["pixel_area_m2"]).sum())
        fp_m2 = float((g["fp"] * g["pixel_area_m2"]).sum())
        fn_m2 = float((g["fn"] * g["pixel_area_m2"]).sum())

        precision_area = tp_m2 / (tp_m2 + fp_m2) if (tp_m2 + fp_m2) > 0 else 0.0
        recall_area = tp_m2 / (tp_m2 + fn_m2) if (tp_m2 + fn_m2) > 0 else 0.0
        f1_area = (
            2 * precision_area * recall_area / (precision_area + recall_area)
            if (precision_area + recall_area) > 0 else 0.0
        )

        f1_s = g["f1"].dropna()
        err_s = g["rel_area_error"].dropna()
        qd_s = g["quantity_disagreement"].dropna()
        ad_s = g["allocation_disagreement"].dropna()

        ref_area_total_m2 = float(g["ref_area_m2"].dropna().sum()) if "ref_area_m2" in g.columns else np.nan
        pred_area_total_m2 = float(g["pred_area_m2"].dropna().sum()) if "pred_area_m2" in g.columns else np.nan
        ref_building_count = float(n_ref_buildings)
        pred_building_count = (
            (pred_area_total_m2 / avg_building_size_m2)
            if np.isfinite(avg_building_size_m2) and avg_building_size_m2 > 0
            else np.nan
        )
        delta_building_count = (
            pred_building_count - ref_building_count
            if np.isfinite(pred_building_count)
            else np.nan
        )
        rel_delta_building_count = (
            delta_building_count / ref_building_count
            if ref_building_count > 0 and np.isfinite(delta_building_count)
            else np.nan
        )
        signed_area_bias = (
            (pred_area_total_m2 - ref_area_total_m2) / ref_area_total_m2
            if np.isfinite(ref_area_total_m2) and ref_area_total_m2 > 0
            else np.nan
        )

        def _r(v):
            return round(float(v), 4) if np.isfinite(v) else float("nan")

        row = {
            "city": city,
            "dataset": key_map["dataset"],
            "grid": key_map.get("grid", None),
            "resolution_m": key_map.get("resolution_m", None),
            "n_ref_buildings": n_ref_buildings,
            "aoi_area_km2": round(aoi_area_km2, 4) if np.isfinite(aoi_area_km2) else float("nan"),
            "buildings_per_km2": round(buildings_per_km2, 2) if np.isfinite(buildings_per_km2) else float("nan"),
            "avg_building_size_m2": round(avg_building_size_m2, 2) if np.isfinite(avg_building_size_m2) else float("nan"),
            "ref_building_count": ref_building_count,
            "pred_building_count": pred_building_count,
            "delta_building_count": delta_building_count,
            "rel_delta_building_count": rel_delta_building_count,
            "n_tiles": int(g["tile_id"].nunique()),
            "valid_area_total_m2": valid_area_total_m2,
            "ref_area_total_m2": ref_area_total_m2,
            "pred_area_total_m2": pred_area_total_m2,
            "tp_total": tp,
            "fp_total": fp,
            "fn_total": fn,
            "precision_area": _r(precision_area),
            "recall_area": _r(recall_area),
            "f1_area": _r(f1_area),
            "f1_tile_mean": _r(f1_s.mean()) if len(f1_s) else float("nan"),
            "f1_tile_median": _r(f1_s.median()) if len(f1_s) else float("nan"),
            "rel_area_error_mean": _r(err_s.mean()) if len(err_s) else float("nan"),
            "rel_area_error_median": _r(err_s.median()) if len(err_s) else float("nan"),
            "signed_area_bias": _r(signed_area_bias),
            "quantity_disagreement_mean": _r(qd_s.mean()) if len(qd_s) else float("nan"),
            "allocation_disagreement_mean": _r(ad_s.mean()) if len(ad_s) else float("nan"),
            "precision_weighted_mean": _r(weighted_row["precision"]) if weighted_row is not None and pd.notna(weighted_row["precision"]) else float("nan"),
            "recall_weighted_mean": _r(weighted_row["recall"]) if weighted_row is not None and pd.notna(weighted_row["recall"]) else float("nan"),
            "f1_weighted_mean": _r(weighted_row["f1"]) if weighted_row is not None and pd.notna(weighted_row["f1"]) else float("nan"),
            "rel_area_error_weighted_mean": _r(weighted_row["rel_area_error"]) if weighted_row is not None and pd.notna(weighted_row["rel_area_error"]) else float("nan"),
            "signed_area_bias_weighted_mean": _r(weighted_row["signed_area_bias"]) if weighted_row is not None and pd.notna(weighted_row["signed_area_bias"]) else float("nan"),
            "quantity_disagreement_weighted_mean": _r(weighted_row["quantity_disagreement"]) if weighted_row is not None and pd.notna(weighted_row["quantity_disagreement"]) else float("nan"),
            "allocation_disagreement_weighted_mean": _r(weighted_row["allocation_disagreement"]) if weighted_row is not None and pd.notna(weighted_row["allocation_disagreement"]) else float("nan"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def plot_raster_tile_f1_boxplot(
    metrics_tiles: pd.DataFrame,
    figures_dir: Path,
    city: str,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Box plot of per-tile raster F1, one box per dataset."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(
        data=metrics_tiles, x="dataset", y="f1", hue="dataset", ax=ax,
        palette="Set2", linewidth=0.8, legend=False,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    ax.set_title(f"Raster tile-level F1 scores — {city}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    fig.tight_layout()
    save_figure(fig, figures_dir, fig_name(city, "raster_tile_f1_boxplot", fmt), dpi=dpi)
    plt.close(fig)


def plot_raster_rel_area_error_boxplot(
    metrics_tiles: pd.DataFrame,
    figures_dir: Path,
    city: str,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Box plot of per-tile relative area error, one box per dataset."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(
        data=metrics_tiles, x="dataset", y="rel_area_error", hue="dataset", ax=ax,
        palette="Set2", linewidth=0.8, legend=False,
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    ax.axhline(0, linestyle="--", linewidth=1, color="grey")
    ax.set_title(f"Raster tile-level relative area error — {city}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=11)
    ax.set_ylabel("Relative area error", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()
    fig.tight_layout()
    save_figure(fig, figures_dir, fig_name(city, "raster_tile_rel_area_error_boxplot", fmt), dpi=dpi)
    plt.close(fig)


def purge_matplotlib() -> None:
    """Release all open matplotlib figures and run gc."""
    plt.close("all")
    if Gcf is not None:
        try:
            Gcf.destroy_all()
        except Exception:
            pass
    gc.collect()

"""
Figure generators for vector and raster validation.

Each generator runs a fixed list of plot calls with per-plot exception
isolation so a failure in one figure doesn't kill the others. Callers
are responsible for purging matplotlib state when the generator is done.
"""
from __future__ import annotations

import logging
import traceback
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

from src.plots.output import (
    plot_iou_dist,
    plot_iou_per_building_sizes,
    plot_raster_count_summary,
    plot_raster_rel_area_error_boxplot,
    plot_raster_tile_f1_boxplot,
    plot_vector_count_summary,
    tile_f1_box_plot,
    tile_f1_spatial_dist,
)

log = logging.getLogger("UrbanValidator.figures")


class VectorFigureGenerator:
    """Generate the standard set of vector validation figures for one city."""

    def __init__(self, cfg: dict, *, dpi: int, fmt: str):
        self.cfg = cfg
        self.dpi = dpi
        self.fmt = fmt

    def make(
        self,
        *,
        dataset_id: str,
        metrics_all: pd.DataFrame,
        matches_all: pd.DataFrame,
        city_summary: pd.DataFrame,
        tiles: gpd.GeoDataFrame,
        figures_dir: Path,
        show_count_plots: bool = True,
    ) -> None:
        matplotlib.use("Agg")
        city_label = dataset_id.replace("-", " ").title()

        size_bins_cfg = self.cfg.get("size_bins", {})
        size_bins = size_bins_cfg.get("bins", [0, 25, 50, 100, 500, 1000, np.inf])
        size_bin_labels = size_bins_cfg.get(
            "labels", ["<25", "25–50", "50–100", "100–500", "500–1000", ">1000"]
        )

        # Tile-level F1 boxplot
        try:
            tile_f1_box_plot(
                metrics_all, figures_dir, city_label, dpi=self.dpi, fmt=self.fmt
            )
        except Exception:
            log.warning(
                "[%s] tile_f1_box_plot failed:\n%s",
                dataset_id, traceback.format_exc(),
            )

        # Spatial F1 choropleth
        try:
            tile_f1_spatial_dist(
                tiles, metrics_all, figures_dir, city_label,
                dpi=self.dpi, fmt=self.fmt,
            )
        except Exception:
            log.warning(
                "[%s] tile_f1_spatial_dist failed:\n%s",
                dataset_id, traceback.format_exc(),
            )

        # IoU distribution histograms
        try:
            plot_iou_dist(
                metrics_all, matches_all, figures_dir, city_label,
                dpi=self.dpi, fmt=self.fmt,
            )
        except Exception:
            log.warning(
                "[%s] plot_iou_dist failed:\n%s",
                dataset_id, traceback.format_exc(),
            )

        # IoU and area error vs building size
        if not matches_all.empty:
            try:
                plot_iou_per_building_sizes(
                    matches_all, figures_dir, city_label,
                    size_bins=size_bins,
                    size_bin_labels=size_bin_labels,
                    dpi=self.dpi, fmt=self.fmt,
                )
            except Exception:
                log.warning(
                    "[%s] plot_iou_per_building_sizes failed:\n%s",
                    dataset_id, traceback.format_exc(),
                )

        if show_count_plots:
            try:
                plot_vector_count_summary(
                    city_summary,
                    figures_dir,
                    city_label,
                    dpi=self.dpi,
                    fmt=self.fmt,
                )
            except Exception:
                log.warning(
                    "[%s] plot_vector_count_summary failed:\n%s",
                    dataset_id, traceback.format_exc(),
                )


class RasterFigureGenerator:
    """Generate the standard set of raster validation figures for one city."""

    def __init__(self, cfg: dict, *, dpi: int, fmt: str):
        self.cfg = cfg
        self.dpi = dpi
        self.fmt = fmt

    def make(
        self,
        *,
        dataset_id: str,
        metrics_all: pd.DataFrame,
        city_summary: pd.DataFrame,
        tiles: gpd.GeoDataFrame,
        figures_dir: Path,
        show_count_plots: bool = True,
    ) -> None:
        matplotlib.use("Agg")
        city_label = dataset_id.replace("-", " ").title()

        # Boxplots across all (dataset, grid) combos: expand the dataset label
        # so each grid resolution is its own box.
        metrics_plot = metrics_all.copy()
        if "grid" in metrics_plot.columns:
            metrics_plot["dataset"] = (
                metrics_plot["dataset"].astype(str)
                + " | "
                + metrics_plot["grid"].astype(str)
            )

        try:
            plot_raster_tile_f1_boxplot(
                metrics_plot, figures_dir, city_label,
                dpi=self.dpi, fmt=self.fmt,
            )
        except Exception:
            log.warning(
                "[%s] plot_raster_tile_f1_boxplot failed:\n%s",
                dataset_id, traceback.format_exc(),
            )

        try:
            plot_raster_rel_area_error_boxplot(
                metrics_plot, figures_dir, city_label,
                dpi=self.dpi, fmt=self.fmt,
            )
        except Exception:
            log.warning(
                "[%s] plot_raster_rel_area_error_boxplot failed:\n%s",
                dataset_id, traceback.format_exc(),
            )

        if show_count_plots:
            try:
                plot_raster_count_summary(
                    city_summary,
                    figures_dir,
                    city_label,
                    dpi=self.dpi,
                    fmt=self.fmt,
                )
            except Exception:
                log.warning(
                    "[%s] plot_raster_count_summary failed:\n%s",
                    dataset_id, traceback.format_exc(),
                )

        # Spatial maps: one per (dataset, grid) for raster
        try:
            if "grid" in metrics_all.columns:
                for (ds_name, grid_name), g in metrics_all.groupby(["dataset", "grid"]):
                    g_plot = g.copy()
                    g_plot["dataset"] = f"{ds_name} | {grid_name}"
                    tile_f1_spatial_dist(
                        tiles, g_plot, figures_dir, city_label,
                        dpi=self.dpi, fmt=self.fmt,
                    )
            else:
                tile_f1_spatial_dist(
                    tiles, metrics_all, figures_dir, city_label,
                    dpi=self.dpi, fmt=self.fmt,
                )
        except Exception:
            log.warning(
                "[%s] tile_f1_spatial_dist (raster) failed:\n%s",
                dataset_id, traceback.format_exc(),
            )
"""
Building dataset validation pipeline.

Usage:
    v = Validator("configs/validation_configs.yaml")
    v.validate_vector()
    v.validate_raster()
"""
from __future__ import annotations

import gc
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import yaml

from src.metrics import compute_tile_metrics, compute_raster_tile_metrics
from src.output import (
    plot_iou_dist,
    plot_iou_per_building_sizes,
    plot_raster_rel_area_error_boxplot,
    plot_raster_tile_f1_boxplot,
    purge_matplotlib,
    summarize_city,
    summarize_raster_city,
    tile_f1_box_plot,
    tile_f1_spatial_dist,
)
from src.utils import (
    consolidate_match_chunks,
    get_projected_crs,
    load_buildings,
    load_validation_datasets,
    log_memory,
    make_tiles,
    subset_by_tile,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Flush accumulated match chunks to disk every N tiles to cap memory usage
_MATCH_FLUSH_INTERVAL = 100


class UrbanValidator:
    """
    Validates building footprint datasets against reference data.

    Vector sources validated: any candidate parquet files in the dataset's
    vector/ folder that match the enabled dataset names in the config.

    Raster validation is not yet implemented.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.root     = Path(self.cfg["root_dir"])
        self.data_dir = self.root / self.cfg["data_dir"]
        self.datasets = load_validation_datasets(self.cfg, self.data_dir)
        log.info("Loaded %d dataset(s) for validation.", len(self.datasets))

    def validate_vector(self) -> Dict[str, bool]:
        """Run vector validation for all datasets. Returns {dataset_id: success}."""
        results: Dict[str, bool] = {}
        for ds in self.datasets:
            try:
                results[ds["id"]] = self._validate_vector_dataset(ds)
            except Exception:
                log.exception("[%s] Unhandled error during vector validation.", ds["id"])
                results[ds["id"]] = False
        return results

    def validate_raster(self) -> Dict[str, bool]:
        """Run raster validation for all datasets. Returns {dataset_id: success}."""
        results: Dict[str, bool] = {}
        for ds in self.datasets:
            try:
                results[ds["id"]] = self._validate_raster_dataset(ds)
            except Exception:
                log.exception("[%s] Unhandled error during raster validation.", ds["id"])
                results[ds["id"]] = False
        return results

    # Vector validation
    def _validate_vector_dataset(self, ds: dict) -> bool:
        """Run the full vector validation pipeline for one dataset."""
        dataset_id = ds["id"]

        # Support both single ref_path (legacy) and multi-file ref_paths list.
        ref_paths: List[Path] = ds.get("ref_paths") or []
        if not ref_paths and ds.get("ref_path"):
            ref_paths = [ds["ref_path"]]

        existing_refs = [p for p in ref_paths if p.exists()]
        if not existing_refs:
            missing = " | ".join(str(p) for p in ref_paths) if ref_paths else "none specified"
            log.warning("[%s] Reference file(s) not found (%s) — skipping.", dataset_id, missing)
            return False

        if len(existing_refs) < len(ref_paths):
            missing_count = len(ref_paths) - len(existing_refs)
            log.warning("[%s] %d of %d reference file(s) missing — continuing with %d.",
                        dataset_id, missing_count, len(ref_paths), len(existing_refs))

        vec_pre      = self.cfg["vector"]["preprocessing"]
        min_area     = float(vec_pre["min_area_m2"])
        tile_size    = float(vec_pre["tile_size_m"])
        tau_overlap  = float(vec_pre["tau_overlap"])
        tau_buffer   = float(vec_pre["tau_buffer_m"])
        tau_boundary = float(vec_pre["tau_boundary"])
        fix_geoms    = bool(vec_pre.get("fix_invalid_geoms", True))

        out_cfg = self.cfg.get("output", {})
        overwrite = bool(out_cfg.get("overwrite", False))
        dpi = int(out_cfg.get("figures", {}).get("dpi", 200))
        fmt = str(out_cfg.get("figures", {}).get("fmt", "png"))

        city_slug   = dataset_id.lower()
        metrics_dir = self.root / "outputs" / "metrics" / city_slug
        figures_dir = self.root / "outputs" / "figures" / city_slug
        metrics_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        sentinel = metrics_dir / "vector_metrics_tiles_all_datasets.parquet"
        if sentinel.exists() and not overwrite:
            log.info("[%s] Already complete — skipping.", dataset_id)
            return True

        log.info("━━━━  %s  ━━━━", dataset_id)
        log_memory(f"{dataset_id} start")

        # Detect projected CRS and build tiles from the dissolved AOI
        crs      = get_projected_crs(ds["aoi"])
        aoi_proj = ds["aoi"].to_crs(crs)
        tiles    = make_tiles(aoi_proj, tile_size)
        log.info("[%s] Working CRS: %s | %d tiles.", dataset_id, crs, len(tiles))
        del aoi_proj

        tiles_path = self.data_dir / dataset_id / "tiles" / f"{city_slug}_tiles.gpkg"
        tiles_path.parent.mkdir(parents=True, exist_ok=True)
        tiles.to_file(tiles_path, driver="GPKG")
        gc.collect()

        # Load reference buildings — merge all ref files for multi-AOI datasets
        ref_parts = [
            load_buildings(path=p, crs_work=crs, min_area_m2=min_area, fix_invalid_geoms=fix_geoms)
            for p in existing_refs
        ]
        ref_all = (
            gpd.GeoDataFrame(pd.concat(ref_parts, ignore_index=True), crs=ref_parts[0].crs)
            if len(ref_parts) > 1 else ref_parts[0]
        )
        del ref_parts
        ref_sindex = ref_all.sindex
        log.info("[%s] Reference buildings: %d (from %d file(s))",
                 dataset_id, len(ref_all), len(existing_refs))

        # Iterate over candidate datasets
        vec_dir             = self.data_dir / dataset_id / "vector"
        per_ds_tile_paths:  List[Path] = []
        per_ds_match_paths: List[Path] = []

        for cand_cfg in self.cfg["vector"]["datasets"]:
            if not cand_cfg.get("enabled", True):
                continue

            ds_name = cand_cfg["name"]
            pattern = f"{city_slug.replace('-', '_')}_{ds_name}*.parquet"
            candidate_files = sorted(vec_dir.glob(pattern))

            if not candidate_files:
                log.warning("[%s / %s] No candidate files found (pattern: %s).",
                            dataset_id, ds_name, pattern)
                continue

            cand_path = candidate_files[0]
            log.info("[%s / %s] Candidate: %s", dataset_id, ds_name, cand_path.name)

            tile_path, match_path = self._run_candidate(
                dataset_id=dataset_id,
                ds_name=ds_name,
                cand_path=cand_path,
                ref_all=ref_all,
                ref_sindex=ref_sindex,
                tiles=tiles,
                metrics_dir=metrics_dir,
                crs=crs,
                min_area=min_area,
                fix_geoms=fix_geoms,
                tau_overlap=tau_overlap,
                tau_buffer=tau_buffer,
                tau_boundary=tau_boundary,
            )
            if tile_path:
                per_ds_tile_paths.append(tile_path)
            per_ds_match_paths.append(match_path)

        del ref_all, ref_sindex
        gc.collect()

        if not per_ds_tile_paths:
            log.warning("[%s] No tile metrics produced — skipping output.", dataset_id)
            return False

        # Combine all-datasets outputs and save
        metrics_all = pd.concat(
            [pd.read_parquet(p) for p in per_ds_tile_paths], ignore_index=True
        )
        metrics_all.to_parquet(sentinel, index=False)

        matches_all = pd.concat(
            [pd.read_parquet(p) for p in per_ds_match_paths if p.exists()],
            ignore_index=True,
        )
        matches_all.to_parquet(
            metrics_dir / "vector_matches_all_datasets.parquet", index=False
        )

        city_summary = summarize_city(dataset_id, metrics_all, matches_all)
        city_summary.to_parquet(
            metrics_dir / "vector_city_summary_all_datasets.parquet", index=False
        )
        city_summary.to_csv(
            metrics_dir / "vector_city_summary_all_datasets.csv", index=False
        )
        log.info("[%s] City summary saved.", dataset_id)
        del city_summary

        # Generate all figures
        try:
            self._make_figures(
                dataset_id=dataset_id,
                metrics_all=metrics_all,
                matches_all=matches_all,
                tiles=tiles,
                figures_dir=figures_dir,
                dpi=dpi,
                fmt=fmt,
            )
        finally:
            purge_matplotlib()

        del metrics_all, matches_all, tiles
        gc.collect()
        log_memory(f"{dataset_id} end")
        log.info("[%s] ✓ Complete.", dataset_id)
        return True

    def _run_candidate(
        self,
        *,
        dataset_id: str,
        ds_name: str,
        cand_path: Path,
        ref_all: gpd.GeoDataFrame,
        ref_sindex,
        tiles: gpd.GeoDataFrame,
        metrics_dir: Path,
        crs: str,
        min_area: float,
        fix_geoms: bool,
        tau_overlap: float,
        tau_buffer: float,
        tau_boundary: float,
    ):
        """Run tile-level IoU matching for one candidate dataset."""
        cand_all    = load_buildings(path=cand_path, crs_work=crs,
                                     min_area_m2=min_area, fix_invalid_geoms=fix_geoms)
        cand_sindex = cand_all.sindex
        log.info("[%s / %s] Candidate buildings: %d", dataset_id, ds_name, len(cand_all))

        ds_tile_metrics: List[dict]         = []
        ds_match_chunks: List[pd.DataFrame] = []
        chunk_counter                       = [0]  # mutable container for nested flush

        def _flush():
            if not ds_match_chunks:
                return
            tmp = metrics_dir / f"_tmp_matches_{ds_name}_{chunk_counter[0]:04d}.parquet"
            pd.concat(ds_match_chunks, ignore_index=True).to_parquet(tmp, index=False)
            chunk_counter[0] += 1
            ds_match_chunks.clear()

        for tile_row in tiles.itertuples():
            tile_geom = tile_row.geometry
            tile_id   = int(tile_row.tile_id)

            ref_tile  = subset_by_tile(ref_all,  ref_sindex,  tile_geom)
            cand_tile = subset_by_tile(cand_all, cand_sindex, tile_geom)

            if ref_tile.empty and cand_tile.empty:
                continue

            metrics, matches_df = compute_tile_metrics(
                ref_tile, dataset_id, cand_tile,
                tau_overlap, tau_buffer, tau_boundary,
                tile_id, ds_name,
            )
            ds_tile_metrics.append(metrics)

            if not matches_df.empty:
                matches_df = matches_df.copy()
                matches_df["city"]    = dataset_id
                matches_df["dataset"] = ds_name
                matches_df["tile_id"] = tile_id
                ds_match_chunks.append(matches_df)

            if len(ds_match_chunks) >= _MATCH_FLUSH_INTERVAL:
                _flush()

            del ref_tile, cand_tile, metrics, matches_df

        del cand_all, cand_sindex
        gc.collect()

        # Save per-dataset tile metrics
        tile_out   = metrics_dir / f"vector_metrics_tiles_{ds_name}.parquet"
        ds_tile_df = pd.DataFrame(ds_tile_metrics)
        del ds_tile_metrics

        returned_tile_path: Optional[Path] = None
        if not ds_tile_df.empty:
            ds_tile_df.to_parquet(tile_out, index=False)
            log.info("[%s / %s] Tile metrics saved → %s", dataset_id, ds_name, tile_out.name)
            returned_tile_path = tile_out
        del ds_tile_df

        # Consolidate match chunks into one file
        match_out = metrics_dir / f"vector_matches_{ds_name}.parquet"
        _flush()
        consolidate_match_chunks(metrics_dir, ds_name, match_out)

        log_memory(f"{dataset_id}/{ds_name} done")
        return returned_tile_path, match_out

    # Vector figure generation
    def _make_figures(
        self,
        *,
        dataset_id: str,
        metrics_all: pd.DataFrame,
        matches_all: pd.DataFrame,
        tiles: gpd.GeoDataFrame,
        figures_dir: Path,
        dpi: int,
        fmt: str,
    ) -> None:
        """Generate and save all standard visualizations for one dataset."""
        matplotlib.use("Agg")
        city_label = dataset_id.replace("-", " ").title()

        size_bins_cfg   = self.cfg.get("size_bins", {})
        size_bins       = size_bins_cfg.get("bins",   [0, 25, 50, 100, 500, 1000, np.inf])
        size_bin_labels = size_bins_cfg.get("labels", ["<25", "25–50", "50–100", "100–500", "500–1000", ">1000"])

        # Tile-level F1 boxplot (one box per candidate dataset)
        try:
            tile_f1_box_plot(metrics_all, figures_dir, city_label, dpi=dpi, fmt=fmt)
        except Exception:
            log.warning("[%s] tile_f1_box_plot failed:\n%s", dataset_id, traceback.format_exc())

        # Spatial F1 choropleth map (one figure per candidate dataset)
        try:
            tile_f1_spatial_dist(tiles, metrics_all, figures_dir, city_label, dpi=dpi, fmt=fmt)
        except Exception:
            log.warning("[%s] tile_f1_spatial_dist failed:\n%s", dataset_id, traceback.format_exc())

        # IoU distribution histograms (one figure per candidate dataset)
        try:
            plot_iou_dist(metrics_all, matches_all, figures_dir, city_label, dpi=dpi, fmt=fmt)
        except Exception:
            log.warning("[%s] plot_iou_dist failed:\n%s", dataset_id, traceback.format_exc())

        # IoU and area error vs building size (one figure per candidate dataset)
        if not matches_all.empty:
            try:
                plot_iou_per_building_sizes(
                    matches_all, figures_dir, city_label,
                    size_bins=size_bins,
                    size_bin_labels=size_bin_labels,
                    dpi=dpi, fmt=fmt,
                )
            except Exception:
                log.warning("[%s] plot_iou_per_building_sizes failed:\n%s",
                            dataset_id, traceback.format_exc())

    # Raster validation
    def _validate_raster_dataset(self, ds: dict) -> bool:
        """Run the full raster validation pipeline for one dataset."""
        dataset_id = ds["id"]

        rast_cfg = self.cfg.get("raster", {})
        if not rast_cfg:
            log.warning("[%s] No raster config found — skipping.", dataset_id)
            return False

        rast_pre     = rast_cfg.get("preprocessing", {})
        tau_frac     = float(rast_pre.get("tau_frac", 0.2))
        oversample   = int(rast_pre.get("oversample_factor", 4))
        all_touched  = bool(rast_pre.get("all_touched", False))
        min_area     = float(self.cfg["vector"]["preprocessing"]["min_area_m2"])
        fix_geoms    = bool(self.cfg["vector"]["preprocessing"].get("fix_invalid_geoms", True))
        tile_size    = float(self.cfg["vector"]["preprocessing"]["tile_size_m"])

        out_cfg   = self.cfg.get("output", {})
        overwrite = bool(out_cfg.get("overwrite", False))
        dpi       = int(out_cfg.get("figures", {}).get("dpi", 200))
        fmt       = str(out_cfg.get("figures", {}).get("fmt", "png"))

        city_slug   = dataset_id.lower()
        metrics_dir = self.root / "outputs" / "metrics" / city_slug
        figures_dir = self.root / "outputs" / "figures" / city_slug
        metrics_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        sentinel = metrics_dir / "raster_metrics_tiles_all_datasets.parquet"
        if sentinel.exists() and not overwrite:
            log.info("[%s] Raster already complete — skipping.", dataset_id)
            return True

        # Reference files
        ref_paths: List[Path] = ds.get("ref_paths") or []
        if not ref_paths and ds.get("ref_path"):
            ref_paths = [ds["ref_path"]]
        existing_refs = [p for p in ref_paths if p.exists()]
        if not existing_refs:
            missing = " | ".join(str(p) for p in ref_paths) if ref_paths else "none specified"
            log.warning("[%s] Reference file(s) not found (%s) — skipping raster validation.",
                        dataset_id, missing)
            return False

        log.info("━━━━  %s (raster)  ━━━━", dataset_id)
        log_memory(f"{dataset_id} raster start")

        # Build tiles from AOI
        crs      = get_projected_crs(ds["aoi"])
        aoi_proj = ds["aoi"].to_crs(crs)
        tiles    = make_tiles(aoi_proj, tile_size)
        aoi_union = aoi_proj.geometry.union_all()
        log.info("[%s] Raster CRS: %s | %d tiles.", dataset_id, crs, len(tiles))
        del aoi_proj

        # Load and merge reference buildings
        ref_parts = [
            load_buildings(path=p, crs_work=crs, min_area_m2=min_area,
                           fix_invalid_geoms=fix_geoms)
            for p in existing_refs
        ]
        ref_all = (
            gpd.GeoDataFrame(pd.concat(ref_parts, ignore_index=True), crs=ref_parts[0].crs)
            if len(ref_parts) > 1 else ref_parts[0]
        )
        del ref_parts
        ref_sindex = ref_all.sindex
        log.info("[%s] Reference buildings: %d", dataset_id, len(ref_all))

        # Discover and evaluate each raster candidate
        rast_dir          = self.data_dir / dataset_id / "raster"
        city_slug_us      = city_slug.replace("-", "_")
        per_ds_tile_paths: List[Path] = []

        for cand_cfg in rast_cfg.get("datasets", []):
            if not cand_cfg.get("enabled", True):
                continue

            ds_name = cand_cfg["name"].replace("-", "_")
            year    = cand_cfg.get("year")          # e.g. 2023, "2023q4", 2025

            # Build an exact filename when year is given; fall back to glob otherwise.
            # File naming mirrors the downloader:
            #   obt        → {city_slug}_obt_{year}.tif
            #   tempo      → {city_slug}_tempo_{year}.tif   (year = "2023q4")
            #   ghsl_*     → {city_slug}_ghsl_{product}_{year}.tif
            if year is not None:
                exact = rast_dir / f"{city_slug_us}_{ds_name}_{year}.tif"
                candidate_files = [exact] if exact.exists() else []
                pattern = f"{city_slug_us}_{ds_name}_{year}.tif"
            else:
                pattern = f"{city_slug_us}_{ds_name}*.tif"
                candidate_files = sorted(rast_dir.glob(pattern))

            if not candidate_files:
                log.warning("[%s / %s] No raster file found (pattern: %s).",
                            dataset_id, ds_name, pattern)
                continue

            cand_path = candidate_files[0]

            # Label used for output files; includes year to avoid collisions across runs
            ds_label = f"{ds_name}_{year}" if year is not None else ds_name
            log.info("[%s / %s] Raster candidate: %s", dataset_id, ds_label, cand_path.name)

            try:
                tile_path = self._run_raster_candidate(
                    dataset_id=dataset_id,
                    ds_label=ds_label,
                    cand_path=cand_path,
                    cand_cfg=cand_cfg,
                    ref_all=ref_all,
                    ref_sindex=ref_sindex,
                    aoi_union=aoi_union,
                    tiles=tiles,
                    metrics_dir=metrics_dir,
                    tau_frac=tau_frac,
                    oversample=oversample,
                    all_touched=all_touched,
                )
            except Exception:
                log.exception("[%s / %s] Raster candidate failed — skipping.",
                              dataset_id, ds_label)
                tile_path = None
            if tile_path:
                per_ds_tile_paths.append(tile_path)

        del ref_all, ref_sindex
        gc.collect()

        if not per_ds_tile_paths:
            log.warning("[%s] No raster tile metrics produced — skipping output.", dataset_id)
            return False

        # Combine and save
        metrics_all = pd.concat(
            [pd.read_parquet(p) for p in per_ds_tile_paths], ignore_index=True
        )
        metrics_all.to_parquet(sentinel, index=False)

        city_summary = summarize_raster_city(dataset_id, metrics_all)
        city_summary.to_parquet(
            metrics_dir / "raster_city_summary_all_datasets.parquet", index=False
        )
        city_summary.to_csv(
            metrics_dir / "raster_city_summary_all_datasets.csv", index=False
        )
        log.info("[%s] Raster city summary saved.", dataset_id)
        del city_summary

        try:
            self._make_raster_figures(
                dataset_id=dataset_id,
                metrics_all=metrics_all,
                tiles=tiles,
                figures_dir=figures_dir,
                dpi=dpi,
                fmt=fmt,
            )
        finally:
            purge_matplotlib()

        del metrics_all, tiles
        gc.collect()
        log_memory(f"{dataset_id} raster end")
        log.info("[%s] ✓ Raster complete.", dataset_id)
        return True

    def _run_raster_candidate(
        self,
        *,
        dataset_id: str,
        ds_label: str,
        cand_path: Path,
        cand_cfg: dict,
        ref_all: gpd.GeoDataFrame,
        ref_sindex,
        aoi_union,
        tiles: gpd.GeoDataFrame,
        metrics_dir: Path,
        tau_frac: float,
        oversample: int,
        all_touched: bool,
    ) -> Optional[Path]:
        """Evaluate one raster candidate over all tiles. Returns tile-metrics parquet path."""
        tile_df = compute_raster_tile_metrics(
            raster_path=cand_path,
            cand_cfg=cand_cfg,
            ref_all=ref_all,
            ref_sindex=ref_sindex,
            aoi_union=aoi_union,
            tiles=tiles,
            tau_frac=tau_frac,
            default_oversample=oversample,
            default_all_touched=all_touched,
        )
        if tile_df.empty:
            log.warning("[%s / %s] No raster tile metrics produced.", dataset_id, ds_label)
            return None

        tile_df["city"]    = dataset_id
        tile_df["dataset"] = ds_label

        out_path = metrics_dir / f"raster_metrics_tiles_{ds_label}.parquet"
        tile_df.to_parquet(out_path, index=False)
        log.info("[%s / %s] Raster tile metrics saved → %s", dataset_id, ds_label, out_path.name)
        log_memory(f"{dataset_id}/{ds_label} raster done")
        return out_path

    def _make_raster_figures(
        self,
        *,
        dataset_id: str,
        metrics_all: pd.DataFrame,
        tiles: gpd.GeoDataFrame,
        figures_dir: Path,
        dpi: int,
        fmt: str,
    ) -> None:
        """Generate and save standard raster visualisations for one dataset."""
        matplotlib.use("Agg")
        city_label = dataset_id.replace("-", " ").title()

        try:
            plot_raster_tile_f1_boxplot(metrics_all, figures_dir, city_label, dpi=dpi, fmt=fmt)
        except Exception:
            log.warning("[%s] plot_raster_tile_f1_boxplot failed:\n%s",
                        dataset_id, traceback.format_exc())

        try:
            tile_f1_spatial_dist(tiles, metrics_all, figures_dir, city_label, dpi=dpi, fmt=fmt)
        except Exception:
            log.warning("[%s] tile_f1_spatial_dist (raster) failed:\n%s",
                        dataset_id, traceback.format_exc())

        try:
            plot_raster_rel_area_error_boxplot(metrics_all, figures_dir, city_label,
                                               dpi=dpi, fmt=fmt)
        except Exception:
            log.warning("[%s] plot_raster_rel_area_error_boxplot failed:\n%s",
                        dataset_id, traceback.format_exc())


"""
Vector validation runner.

Runs tile-level IoU matching for every enabled candidate vector dataset
in the city's vector/ folder, collects tile and match parquet outputs,
combines them into per-city all-datasets parquets, builds the city
summary, and dispatches figure generation.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from src.metrics.vector.tile_metrics import compute_tile_metrics
from src.metrics.vector.size_metrics import (
    aoi_area_km2,
    compute_city_density_summary,
    compute_size_bin_metrics,
)
from src.plots.output import purge_matplotlib, summarize_city
from src.utils.buildings import load_buildings
from src.utils.memory import log_memory
from src.utils.tiling import subset_by_tile
from src.validate.base import BaseValidationRunner
from src.plots.figures import VectorFigureGenerator
from src.validate.match_writer import MatchChunkWriter

log = logging.getLogger("UrbanValidator.vector")

# Flush accumulated match chunks to disk every N tiles to cap memory usage.
_MATCH_FLUSH_INTERVAL = 100


class VectorValidationRunner(BaseValidationRunner):
    """Tile-level IoU matching against reference for every enabled candidate."""

    sentinel_name = "vector_metrics_tiles_all_datasets.parquet"

    def run(self, ds: dict) -> bool:
        dataset_id = ds["id"]

        existing_refs = self._resolve_ref_paths(ds)
        if not existing_refs:
            return False

        # Config readers
        vec_pre = self.cfg["vector"]["preprocessing"]
        min_area = float(vec_pre["min_area_m2"])
        tile_size = float(vec_pre["tile_size_m"])
        tau_overlap = float(vec_pre["tau_overlap"])
        tau_buffer = float(vec_pre["tau_buffer_m"])
        tau_boundary = float(vec_pre["tau_boundary"])
        fix_geoms = bool(vec_pre.get("fix_invalid_geoms", True))

        # Size-bin config (used for per-bin metrics and figures)
        size_bins_cfg = self.cfg.get("size_bins", {})
        size_bins = size_bins_cfg.get(
            "bins", [0, 25, 50, 100, 500, 1000, np.inf]
        )
        size_bin_labels = size_bins_cfg.get(
            "labels", ["<25", "25–50", "50–100", "100–500", "500–1000", ">1000"]
        )

        overwrite, dpi, fmt = self._read_output_cfg()

        metrics_dir, figures_dir = self._setup_output_dirs(dataset_id)
        if self._is_already_complete(metrics_dir, dataset_id, overwrite):
            return True

        log.info("━━━━  %s  ━━━━", dataset_id)
        log_memory(f"{dataset_id} start")

        # Tiling
        tiles, crs = self._build_tiles(ds, tile_size)
        self._save_tiles(tiles, dataset_id)

        # AOI area in km² — denominator for density metrics. Must be
        # computed in a metric CRS, so reproject the AOI once here.
        aoi_km2 = aoi_area_km2(ds["aoi"].to_crs(crs))
        log.info("[%s] AOI dissolved area: %.3f km².", dataset_id, aoi_km2)

        # Reference buildings
        ref_all = self._load_reference_buildings(
            ref_paths=existing_refs,
            crs_work=crs,
            min_area_m2=min_area,
            fix_invalid_geoms=fix_geoms,
            dataset_id=dataset_id,
        )
        ref_sindex = ref_all.sindex
        log.info(
            "[%s] Reference buildings: %d (from %d file(s))",
            dataset_id, len(ref_all), len(existing_refs),
        )

        # Per-candidate runs
        city_slug = dataset_id.lower()
        vec_dir = self.data_dir / dataset_id / "vector"
        per_ds_tile_paths: List[Path] = []
        per_ds_match_paths: List[Path] = []
        per_ds_size_bin_paths: List[Path] = []
        cand_areas_by_dataset: Dict[str, pd.Series] = {}

        for cand_cfg in self.cfg["vector"]["datasets"]:
            if not cand_cfg.get("enabled", True):
                continue

            ds_name = cand_cfg["name"]
            pattern = f"{city_slug.replace('-', '_')}_{ds_name}*.parquet"
            candidate_files = sorted(vec_dir.glob(pattern))

            if not candidate_files:
                log.warning(
                    "[%s / %s] No candidate files found (pattern: %s).",
                    dataset_id, ds_name, pattern,
                )
                continue

            cand_path = candidate_files[0]
            log.info("[%s / %s] Candidate: %s", dataset_id, ds_name, cand_path.name)

            tile_path, match_path, size_bin_path, cand_areas = self._run_candidate(
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
                size_bins=size_bins,
                size_bin_labels=size_bin_labels,
            )
            if tile_path:
                per_ds_tile_paths.append(tile_path)
            per_ds_match_paths.append(match_path)
            if size_bin_path is not None:
                per_ds_size_bin_paths.append(size_bin_path)
            if cand_areas is not None:
                cand_areas_by_dataset[ds_name] = cand_areas

        # Capture reference areas before ref_all is freed
        ref_areas = ref_all["area_m2"].copy() if "area_m2" in ref_all.columns else pd.Series(dtype=float)

        del ref_all, ref_sindex
        gc.collect()

        if not per_ds_tile_paths:
            log.warning("[%s] No tile metrics produced — skipping output.", dataset_id)
            return False

        # Combine and save city-level outputs
        sentinel = metrics_dir / self.sentinel_name
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

        # Combined per-(city,dataset,size_bin) metrics
        if per_ds_size_bin_paths:
            size_bin_all = pd.concat(
                [pd.read_parquet(p) for p in per_ds_size_bin_paths],
                ignore_index=True,
            )
            size_bin_all.to_parquet(
                metrics_dir / "vector_size_bin_metrics_all_datasets.parquet",
                index=False,
            )
            size_bin_all.to_csv(
                metrics_dir / "vector_size_bin_metrics_all_datasets.csv",
                index=False,
            )
            log.info("[%s] Per-size-bin metrics saved.", dataset_id)
            del size_bin_all

        # City density and average building size summary
        density_summary = compute_city_density_summary(
            dataset_id=dataset_id,
            aoi_area_km2_value=aoi_km2,
            ref_areas=ref_areas,
            cand_areas=cand_areas_by_dataset,
        )
        density_summary.to_parquet(
            metrics_dir / "vector_city_density_summary.parquet", index=False
        )
        density_summary.to_csv(
            metrics_dir / "vector_city_density_summary.csv", index=False
        )
        log.info("[%s] City density summary saved.", dataset_id)
        del density_summary, ref_areas, cand_areas_by_dataset

        # Figures
        try:
            figs = VectorFigureGenerator(self.cfg, dpi=dpi, fmt=fmt)
            figs.make(
                dataset_id=dataset_id,
                metrics_all=metrics_all,
                matches_all=matches_all,
                tiles=tiles,
                figures_dir=figures_dir,
            )
        finally:
            purge_matplotlib()

        del metrics_all, matches_all, tiles
        gc.collect()
        log_memory(f"{dataset_id} end")
        log.info("[%s] ✓ Complete.", dataset_id)
        return True

    # -----------------------------------------------------------------
    # Per-candidate tile-level IoU matching
    # -----------------------------------------------------------------

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
        size_bins: List[float],
        size_bin_labels: List[str],
    ) -> Tuple[Optional[Path], Path, Optional[Path], Optional[pd.Series]]:
        """Run tile-level IoU matching for one candidate dataset.

        Returns
        -------
        (tile_metrics_path, match_path, size_bin_path, cand_areas)
            tile_metrics_path : per-dataset tile metrics parquet, or None if empty.
            match_path        : per-dataset consolidated matches parquet.
            size_bin_path     : per-dataset per-size-bin metrics parquet, or None.
            cand_areas        : Series of candidate building areas, or None if no
                                candidates were loaded.
        """
        cand_all = load_buildings(
            path=cand_path,
            crs_work=crs,
            min_area_m2=min_area,
            fix_invalid_geoms=fix_geoms,
        )
        cand_sindex = cand_all.sindex
        log.info(
            "[%s / %s] Candidate buildings: %d",
            dataset_id, ds_name, len(cand_all),
        )

        ds_tile_metrics: List[dict] = []
        match_writer = MatchChunkWriter(
            metrics_dir, ds_name, flush_interval=_MATCH_FLUSH_INTERVAL
        )

        for tile_row in tiles.itertuples():
            tile_geom = tile_row.geometry
            tile_id = int(tile_row.tile_id)

            ref_tile = subset_by_tile(ref_all, ref_sindex, tile_geom)
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
                matches_df["city"] = dataset_id
                matches_df["dataset"] = ds_name
                matches_df["tile_id"] = tile_id
                match_writer.append(matches_df)

            del ref_tile, cand_tile, metrics, matches_df

        # Save per-dataset tile metrics
        tile_out = metrics_dir / f"vector_metrics_tiles_{ds_name}.parquet"
        ds_tile_df = pd.DataFrame(ds_tile_metrics)
        del ds_tile_metrics

        returned_tile_path: Optional[Path] = None
        if not ds_tile_df.empty:
            ds_tile_df.to_parquet(tile_out, index=False)
            log.info(
                "[%s / %s] Tile metrics saved → %s",
                dataset_id, ds_name, tile_out.name,
            )
            returned_tile_path = tile_out
        del ds_tile_df

        # Consolidate match chunks into one file
        match_out = match_writer.finalize()

        # Per-size-bin metrics — needs cand_all in scope plus the matches
        # we just wrote. Read the consolidated matches back in (small
        # relative to the geometry GDFs, and cand_all has not been freed).
        size_bin_path: Optional[Path] = None
        try:
            matches_for_bins = (
                pd.read_parquet(match_out) if match_out.exists() else pd.DataFrame()
            )
            size_bin_df = compute_size_bin_metrics(
                matches_df=matches_for_bins,
                ref_all=ref_all,
                cand_all=cand_all,
                dataset_id=dataset_id,
                ds_name=ds_name,
                size_bins=size_bins,
                size_bin_labels=size_bin_labels,
            )
            del matches_for_bins
            if not size_bin_df.empty:
                size_bin_path = (
                    metrics_dir / f"vector_size_bin_metrics_{ds_name}.parquet"
                )
                size_bin_df.to_parquet(size_bin_path, index=False)
                log.info(
                    "[%s / %s] Per-size-bin metrics saved → %s",
                    dataset_id, ds_name, size_bin_path.name,
                )
            del size_bin_df
        except Exception:
            log.exception(
                "[%s / %s] Per-size-bin metrics failed — continuing.",
                dataset_id, ds_name,
            )

        # Capture candidate areas before cand_all is freed
        cand_areas = (
            cand_all["area_m2"].copy() if "area_m2" in cand_all.columns else None
        )

        del cand_all, cand_sindex
        gc.collect()

        log_memory(f"{dataset_id}/{ds_name} done")
        return returned_tile_path, match_out, size_bin_path, cand_areas
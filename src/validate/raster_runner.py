"""
Raster validation runner.

For each enabled candidate raster product, computes tile-level pixel
metrics across one or more evaluation grids. Aggregates per-candidate
parquets into a per-city all-datasets parquet, builds the city summary,
and dispatches figure generation.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd

from src.metrics.raster.grids import _native_guard_settings
from src.metrics.raster.tile_metrics import compute_raster_tile_metrics
from src.plots.output import purge_matplotlib, summarize_raster_city
from src.utils.memory import log_memory
from src.validate.base import BaseValidationRunner
from src.plots.figures import RasterFigureGenerator

log = logging.getLogger("UrbanValidator.raster")


class RasterValidationRunner(BaseValidationRunner):
    """Tile-level pixel metrics against rasterised reference for each candidate."""

    sentinel_name = "raster_metrics_tiles_all_datasets.parquet"

    def run(self, ds: dict) -> bool:
        dataset_id = ds["id"]

        rast_cfg = self.cfg.get("raster", {})
        if not rast_cfg:
            log.warning("[%s] No raster config found — skipping.", dataset_id)
            return False

        # Raster preprocessing config
        rast_pre = rast_cfg.get("preprocessing", {})
        ref_min_building_m2 = float(rast_pre.get("min_building_m2", 20.0))
        oversample = int(rast_pre.get("oversample_factor", 4))
        all_touched = bool(rast_pre.get("all_touched", False))
        evaluation_grids = rast_pre.get("evaluation_grids", None)
        native_guard_cfg = _native_guard_settings(rast_pre)

        # Vector-section reuse: tile size + cleanup options come from the
        # vector preprocessing block (kept identical to the legacy validator).
        min_area = float(self.cfg["vector"]["preprocessing"]["min_area_m2"])
        fix_geoms = bool(
            self.cfg["vector"]["preprocessing"].get("fix_invalid_geoms", True)
        )
        tile_size = float(self.cfg["vector"]["preprocessing"]["tile_size_m"])

        overwrite, dpi, fmt = self._read_output_cfg()
        show_count_plots = bool(
            rast_cfg.get("reporting", {}).get("derive_counts", True)
        )

        metrics_dir, figures_dir = self._setup_output_dirs(dataset_id)

        sentinel = metrics_dir / self.sentinel_name
        if sentinel.exists() and not overwrite:
            log.info("[%s] Raster already complete — skipping.", dataset_id)
            return True

        existing_refs = self._resolve_ref_paths(ds)
        if not existing_refs:
            log.warning(
                "[%s] Reference file(s) missing — skipping raster validation.",
                dataset_id,
            )
            return False

        log.info("━━━━  %s (raster)  ━━━━", dataset_id)
        log_memory(f"{dataset_id} raster start")

        # Tiling — additionally need the AOI union in the projected CRS.
        tiles, aoi_union, crs = self._build_tiles_with_aoi_union(ds, tile_size)
        log.info("[%s] Raster CRS: %s | %d tiles.", dataset_id, crs, len(tiles))

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
            "[%s] Raster reference buildings: %d (from %d file(s))",
            dataset_id, len(ref_all), len(existing_refs),
        )

        n_ref_buildings = len(ref_all)
        aoi_area_km2 = aoi_union.area / 1e6
        buildings_per_km2 = n_ref_buildings / aoi_area_km2 if aoi_area_km2 > 0 else float("nan")
        avg_building_size_m2 = float(ref_all.geometry.area.mean()) if n_ref_buildings > 0 else float("nan")

        # Per-candidate runs
        city_slug = dataset_id.lower()
        rast_dir = self.data_dir / dataset_id / "raster"
        per_ds_tile_paths: List[Path] = []

        for cand_cfg in rast_cfg.get("datasets", []):
            if not cand_cfg.get("enabled", True):
                continue

            ds_name = cand_cfg["name"]
            year = cand_cfg.get("year", None)
            slug = city_slug.replace("-", "_")
            default_pattern = (
                f"{slug}_{ds_name}_{year}*"
                if year is not None
                else f"{slug}_{ds_name}*"
            )
            pattern = cand_cfg.get("pattern", default_pattern)
            candidate_files = sorted(rast_dir.glob(pattern))

            if not candidate_files:
                log.warning(
                    "[%s / %s] No raster file found (pattern: %s).",
                    dataset_id, ds_name, pattern,
                )
                continue

            cand_path = candidate_files[0]
            ds_label = f"{ds_name}_{year}" if year is not None else ds_name
            log.info(
                "[%s / %s] Raster candidate: %s",
                dataset_id, ds_label, cand_path.name,
            )

            ds_min_building_m2 = float(
                cand_cfg.get("min_building_m2", ref_min_building_m2)
            )

            try:
                tile_path = self._run_candidate(
                    dataset_id=dataset_id,
                    ds_label=ds_label,
                    cand_path=cand_path,
                    cand_cfg=cand_cfg,
                    ref_all=ref_all,
                    ref_sindex=ref_sindex,
                    aoi_union=aoi_union,
                    tiles=tiles,
                    metrics_dir=metrics_dir,
                    min_building_m2=ds_min_building_m2,
                    ref_min_building_m2=ref_min_building_m2,
                    oversample=oversample,
                    all_touched=all_touched,
                    evaluation_grids=evaluation_grids,
                    native_guard_cfg=native_guard_cfg,
                )
            except Exception:
                log.exception(
                    "[%s / %s] Raster candidate failed — skipping.",
                    dataset_id, ds_label,
                )
                tile_path = None

            if tile_path:
                per_ds_tile_paths.append(tile_path)

        del ref_all, ref_sindex
        gc.collect()

        if not per_ds_tile_paths:
            log.warning(
                "[%s] No raster tile metrics produced — skipping output.",
                dataset_id,
            )
            return False

        metrics_all = pd.concat(
            [pd.read_parquet(p) for p in per_ds_tile_paths],
            ignore_index=True,
        )
        metrics_all.to_parquet(sentinel, index=False)

        city_summary = summarize_raster_city(
            dataset_id,
            metrics_all,
            n_ref_buildings=n_ref_buildings,
            aoi_area_km2=aoi_area_km2,
            buildings_per_km2=buildings_per_km2,
            avg_building_size_m2=avg_building_size_m2,
        )
        city_summary.to_parquet(
            metrics_dir / "raster_city_summary_all_datasets.parquet",
            index=False,
        )
        city_summary.to_csv(
            metrics_dir / "raster_city_summary_all_datasets.csv",
            index=False,
        )
        log.info("[%s] Raster city summary saved.", dataset_id)

        try:
            figs = RasterFigureGenerator(self.cfg, dpi=dpi, fmt=fmt)
            figs.make(
                dataset_id=dataset_id,
                metrics_all=metrics_all,
                city_summary=city_summary,
                tiles=tiles,
                figures_dir=figures_dir,
                show_count_plots=show_count_plots,
            )
        finally:
            purge_matplotlib()

        del city_summary, metrics_all, tiles
        gc.collect()
        log_memory(f"{dataset_id} raster end")
        log.info("[%s] ✓ Raster complete.", dataset_id)
        return True

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _build_tiles_with_aoi_union(self, ds: dict, tile_size_m: float):
        """
        Like base._build_tiles, but also returns the projected AOI union
        which raster validation needs for the AOI mask.
        Returns (tiles, aoi_union, crs).
        """
        from src.utils.geometry import get_projected_crs
        from src.utils.tiling import make_tiles

        crs = get_projected_crs(ds["aoi"])
        aoi_proj = ds["aoi"].to_crs(crs)
        tiles = make_tiles(aoi_proj, tile_size_m)
        aoi_union = aoi_proj.geometry.union_all()
        del aoi_proj
        return tiles, aoi_union, crs

    def _run_candidate(
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
        min_building_m2: float,
        ref_min_building_m2: float,
        oversample: int,
        all_touched: bool,
        evaluation_grids: Optional[List[dict]] = None,
        native_guard_cfg: Optional[dict] = None,
    ) -> Optional[Path]:
        """Evaluate one raster candidate over all tiles and requested grids."""
        tile_df = compute_raster_tile_metrics(
            raster_path=cand_path,
            cand_cfg=cand_cfg,
            ref_all=ref_all,
            ref_sindex=ref_sindex,
            aoi_union=aoi_union,
            tiles=tiles,
            min_building_m2=min_building_m2,
            ref_min_building_m2=ref_min_building_m2,
            default_oversample=oversample,
            default_all_touched=all_touched,
            evaluation_grids=evaluation_grids,
            native_guard_cfg=native_guard_cfg,
        )
        if tile_df.empty:
            log.warning(
                "[%s / %s] No raster tile metrics produced.",
                dataset_id, ds_label,
            )
            return None

        tile_df["city"] = dataset_id
        tile_df["dataset"] = ds_label

        out_path = metrics_dir / f"raster_metrics_tiles_{ds_label}.parquet"
        tile_df.to_parquet(out_path, index=False)
        log.info(
            "[%s / %s] Raster tile metrics saved → %s",
            dataset_id, ds_label, out_path.name,
        )
        log_memory(f"{dataset_id}/{ds_label} raster done")
        return out_path
    
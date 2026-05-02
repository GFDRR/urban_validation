"""
Base class for validation runners.

VectorValidationRunner and RasterValidationRunner share substantial
setup logic: resolving reference paths (with the legacy single-path
fallback), building output directories, deciding whether to short-circuit
on a sentinel file, choosing a projected CRS, building tiles, and
loading + merging reference building footprints.

This base class owns that shared setup and exposes a small set of
helper methods. Subclasses implement run(ds) for the actual per-source
validation loop.
"""
from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import pandas as pd

from src.utils import (
    get_projected_crs,
    load_buildings,
    log_memory,
    make_tiles,
)

log = logging.getLogger("UrbanValidator.runner")


class BaseValidationRunner(ABC):
    """
    Abstract base for vector and raster validation runners.

    Subclasses must implement:
      - sentinel_name      : str    name of the per-city sentinel file
      - run(ds)            : bool   run validation for one dataset
    """

    sentinel_name: str = "validation_metrics_all_datasets.parquet"

    def __init__(self, cfg: dict, root: Path, data_dir: Path):
        self.cfg = cfg
        self.root = Path(root)
        self.data_dir = Path(data_dir)

    # -----------------------------------------------------------------
    # Public entry point — subclasses override
    # -----------------------------------------------------------------

    @abstractmethod
    def run(self, ds: dict) -> bool:
        """Run validation for one dataset. Returns True on success."""
        raise NotImplementedError

    # -----------------------------------------------------------------
    # Shared setup helpers
    # -----------------------------------------------------------------

    def _resolve_ref_paths(self, ds: dict) -> List[Path]:
        """
        Resolve which reference files actually exist on disk. Supports
        both the multi-file ref_paths list and the legacy single
        ref_path. Logs warnings for missing files.
        """
        ref_paths: List[Path] = ds.get("ref_paths") or []
        if not ref_paths and ds.get("ref_path"):
            ref_paths = [ds["ref_path"]]

        existing_refs = [p for p in ref_paths if p.exists()]

        if not existing_refs:
            missing = " | ".join(str(p) for p in ref_paths) if ref_paths else "none specified"
            log.warning(
                "[%s] Reference file(s) not found (%s) — skipping.",
                ds["id"], missing,
            )
            return []

        if len(existing_refs) < len(ref_paths):
            missing_count = len(ref_paths) - len(existing_refs)
            log.warning(
                "[%s] %d of %d reference file(s) missing — continuing with %d.",
                ds["id"], missing_count, len(ref_paths), len(existing_refs),
            )

        return existing_refs

    def _setup_output_dirs(self, dataset_id: str) -> Tuple[Path, Path]:
        """Create and return (metrics_dir, figures_dir) for one city."""
        city_slug = dataset_id.lower()
        metrics_dir = self.root / "outputs" / "metrics" / city_slug
        figures_dir = self.root / "outputs" / "figures" / city_slug
        metrics_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        return metrics_dir, figures_dir

    def _is_already_complete(
        self,
        metrics_dir: Path,
        dataset_id: str,
        overwrite: bool,
    ) -> bool:
        """Check the sentinel file and short-circuit when already done."""
        sentinel = metrics_dir / self.sentinel_name
        if sentinel.exists() and not overwrite:
            log.info("[%s] Already complete — skipping.", dataset_id)
            return True
        return False

    def _build_tiles(
        self,
        ds: dict,
        tile_size_m: float,
    ) -> Tuple[gpd.GeoDataFrame, str]:
        """
        Choose a projected CRS for the AOI, reproject, and build tiles.
        Returns (tiles, projected_crs_string).
        """
        crs = get_projected_crs(ds["aoi"])
        aoi_proj = ds["aoi"].to_crs(crs)
        tiles = make_tiles(aoi_proj, tile_size_m)
        log.info("[%s] Working CRS: %s | %d tiles.", ds["id"], crs, len(tiles))
        del aoi_proj
        return tiles, crs

    def _save_tiles(self, tiles: gpd.GeoDataFrame, dataset_id: str) -> Path:
        """Persist tiles for one city to the standard location."""
        city_slug = dataset_id.lower()
        tiles_path = self.data_dir / dataset_id / "tiles" / f"{city_slug}_tiles.gpkg"
        tiles_path.parent.mkdir(parents=True, exist_ok=True)
        tiles.to_file(tiles_path, driver="GPKG")
        gc.collect()
        return tiles_path

    def _load_reference_buildings(
        self,
        ref_paths: List[Path],
        crs_work: str,
        min_area_m2: float,
        fix_invalid_geoms: bool,
        dataset_id: str,
    ) -> gpd.GeoDataFrame:
        """
        Load reference buildings from one or more files and merge them.
        For multi-AOI datasets this concatenates all reference parts.
        """
        ref_parts = [
            load_buildings(
                path=p,
                crs_work=crs_work,
                min_area_m2=min_area_m2,
                fix_invalid_geoms=fix_invalid_geoms,
            )
            for p in ref_paths
        ]
        ref_all = (
            gpd.GeoDataFrame(
                pd.concat(ref_parts, ignore_index=True),
                crs=ref_parts[0].crs,
            )
            if len(ref_parts) > 1 else ref_parts[0]
        )
        del ref_parts
        return ref_all

    # -----------------------------------------------------------------
    # Common config readers
    # -----------------------------------------------------------------

    def _read_output_cfg(self) -> Tuple[bool, int, str]:
        """Return (overwrite, dpi, fmt) from the output section of the config."""
        out_cfg = self.cfg.get("output", {})
        overwrite = bool(out_cfg.get("overwrite", False))
        figs_cfg = out_cfg.get("figures", {})
        dpi = int(figs_cfg.get("dpi", 200))
        fmt = str(figs_cfg.get("fmt", "png"))
        return overwrite, dpi, fmt
    
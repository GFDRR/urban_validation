"""
Main entry point for the AOI-level building dataset download pipeline.

UrbanDownloader is now a thin orchestrator: it loads the AOI inventory,
instantiates the enabled runners, and dispatches them per-dataset. All
source-specific logic lives in src/download/.

Vector sources: Overture Maps, Global Building Atlas (GBA), GloBFP
Raster sources: Google Open Buildings Temporal (OBT), Microsoft TEMPO,
                GHSL, WSF Tracker

Usage:
    downloader = UrbanDownloader("configs/data_configs.yaml")
    downloader.download_vector()   # Overture, GBA, GloBFP
    downloader.download_raster()   # OBT, TEMPO, GHSL, WSF Tracker

Prepared by: Rufai Omowunmi Balogun
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from src.config import load_config
from src.download.db_connection import connect_duckdb, resolve_overture_base
from src.download.vector import (
    OvertureRunner,
    GBARunner,
    GloBFPRunner,
)
from src.download.raster import (
    OBTRunner,
    TEMPORunner,
    GHSLRunner,
    WSFTrackerRunner,
)
from src.utils.ee_utils import init_earth_engine
from src.utils.aoi_inventory import load_all_aois
from src.utils.tiling import resolve_out_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("UrbanDownloader")


class UrbanDownloader:
    """
    Orchestrates downloads for all cities in the AOI inventory.

    Vector sources: Overture Maps, Global Building Atlas (GBA), GloBFP
    Raster sources: Google Open Buildings Temporal (OBT), Microsoft TEMPO,
                    GHSL, WSF Tracker

    Enable/disable each source in configs/data_configs.yaml.
    """

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.overwrite = self.config.output.overwrite
        self.datasets = load_all_aois(self.config)
        log.info("Loaded %d dataset(s) from inventory.", len(self.datasets))

    # -----------------------------------------------------------------
    # VECTOR
    # -----------------------------------------------------------------

    def download_vector(self) -> Dict[str, List[str]]:
        """Download Overture, GBA, and GloBFP footprints for all datasets."""
        cfg = self.config
        runners = self._build_vector_runners()
        if not runners:
            raise ValueError(
                "No vector source enabled in config. "
                "Set overture.enabled, gba.enabled, or globfp.enabled to true."
            )

        # GloBFP needs the world grid loaded once before running.
        for r in runners:
            if isinstance(r, GloBFPRunner):
                r.prepare()

        all_outputs: Dict[str, List[str]] = {}
        con = connect_duckdb(cfg)
        try:
            for r in runners:
                r.bind_connection(con)

            for ds in self.datasets:
                out_root = resolve_out_root(cfg, ds["id"], "vector")
                outputs: List[str] = []
                for r in runners:
                    try:
                        outputs += r.run(ds, out_root)
                    except Exception:
                        log.exception(
                            "  [%s] FAILED for dataset %s", r.name, ds["id"]
                        )
                all_outputs[ds["id"]] = outputs
                log.info("Vector done | %s | %d file(s)", ds["id"], len(outputs))
        finally:
            con.close()

        log.info(
            "Vector download complete | total=%d",
            sum(len(v) for v in all_outputs.values()),
        )
        return all_outputs

    # -----------------------------------------------------------------
    # RASTER
    # -----------------------------------------------------------------

    def download_raster(self) -> Dict[str, List[str]]:
        """Download OBT, Microsoft TEMPO, GHSL and WSF Tracker rasters."""
        cfg = self.config
        runners = self._build_raster_runners()
        if not runners:
            raise ValueError(
                "No raster source enabled in config. "
                "Set datasets.google_open_buildings_temporal.enabled, "
                "datasets.microsoft_tempo.enabled, datasets.ghsl.enabled, "
                "or datasets.wsf_tracker.enabled to true."
            )

        # Earth Engine init only if at least one EE-backed runner is enabled.
        needs_ee = any(isinstance(r, (OBTRunner, GHSLRunner)) for r in runners)
        if needs_ee:
            init_earth_engine(cfg.ee_project)

        all_outputs: Dict[str, List[str]] = {}
        for ds in self.datasets:
            out_root = resolve_out_root(cfg, ds["id"], "raster")
            outputs: List[str] = []
            for r in runners:
                log.info("  ── %s | %s ──", r.name, ds["id"])
                try:
                    outputs += r.run(ds, out_root)
                except Exception:
                    log.exception("  [%s] FAILED for dataset %s", r.name, ds["id"])
            all_outputs[ds["id"]] = outputs
            log.info("Raster done | %s | %d file(s)", ds["id"], len(outputs))

        log.info(
            "Raster download complete | total=%d",
            sum(len(v) for v in all_outputs.values()),
        )
        return all_outputs

    # -----------------------------------------------------------------
    # Runner factories
    # -----------------------------------------------------------------

    def _build_vector_runners(self) -> List:
        cfg = self.config
        runners: List = []

        # GBA first (preserves original ordering inside each dataset loop).
        if cfg.gba.enabled:
            runners.append(
                GBARunner(cfg, cfg.gba, overwrite=self.overwrite)
            )
        if cfg.overture.enabled:
            base_path = resolve_overture_base(cfg)
            runners.append(
                OvertureRunner(
                    cfg,
                    cfg.overture,
                    overwrite=self.overwrite,
                    base_path=base_path,
                )
            )
        if cfg.globfp.enabled:
            runners.append(
                GloBFPRunner(cfg, cfg.globfp, overwrite=self.overwrite)
            )
        return runners

    def _build_raster_runners(self) -> List:
        cfg = self.config
        sources = cfg.datasets
        runners: List = []

        if sources.google_open_buildings_temporal.enabled:
            runners.append(
                OBTRunner(
                    cfg,
                    sources.google_open_buildings_temporal,
                    overwrite=self.overwrite,
                )
            )
        if sources.microsoft_tempo.enabled:
            runners.append(
                TEMPORunner(cfg, sources.microsoft_tempo, overwrite=self.overwrite)
            )
        if sources.ghsl.enabled:
            runners.append(
                GHSLRunner(cfg, sources.ghsl, overwrite=self.overwrite)
            )
        if sources.wsf_tracker.enabled:
            runners.append(
                WSFTrackerRunner(cfg, sources.wsf_tracker, overwrite=self.overwrite)
            )
        return runners

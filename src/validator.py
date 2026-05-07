"""
Main entry point for the building-dataset validation pipeline.

UrbanValidator is now a thin orchestrator: it loads the AOI inventory
and dispatches each dataset to the vector and raster runners. All
per-source validation logic lives in src/validate/.

Usage:
    v = UrbanValidator("configs/validation_configs.yaml")
    v.validate_vector()
    v.validate_raster()
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import yaml

from src.utils.aoi_inventory import load_validation_datasets
from src.validate.vector_runner import VectorValidationRunner
from src.validate.raster_runner import RasterValidationRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class UrbanValidator:
    """
    Validates building footprint datasets against reference data.

    Vector sources validated: any candidate parquet files in the
    dataset's vector/ folder that match the enabled dataset names in
    the config. Raster sources validated: any rasters matching the
    configured patterns under the dataset's raster/ folder.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.root = Path(self.cfg["root_dir"])
        self.data_dir = self.root / self.cfg["data_dir"]
        self.datasets = load_validation_datasets(self.cfg, self.data_dir)
        log.info("Loaded %d dataset(s) for validation.", len(self.datasets))

        self._vector_runner = VectorValidationRunner(
            self.cfg, self.root, self.data_dir
        )
        self._raster_runner = RasterValidationRunner(
            self.cfg, self.root, self.data_dir
        )

    def validate_vector(self) -> Dict[str, bool]:
        """Run vector validation for all datasets. Returns {dataset_id: success}."""
        results: Dict[str, bool] = {}
        for ds in self.datasets:
            try:
                results[ds["id"]] = self._vector_runner.run(ds)
            except Exception:
                log.exception(
                    "[%s] Unhandled error during vector validation.", ds["id"]
                )
                results[ds["id"]] = False
        return results

    def validate_raster(self) -> Dict[str, bool]:
        """Run raster validation for all datasets. Returns {dataset_id: success}."""
        results: Dict[str, bool] = {}
        for ds in self.datasets:
            try:
                results[ds["id"]] = self._raster_runner.run(ds)
            except Exception:
                log.exception(
                    "[%s] Unhandled error during raster validation.", ds["id"]
                )
                results[ds["id"]] = False
        return results

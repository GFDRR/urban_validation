"""
Main entry point for running Urban Downloader and Validator pipelines end to end.

Usage guidance:
- For Colab: Use the provided project root path and ensure Drive is mounted.
  In a Colab notebook cell:
        %run main.py
        !python main.py --data-config configs/data_configs.yaml --val-config configs/validation_configs.yaml
- From the CLI (project needs to be set up within a Colab environment):
    python main.py
    python main.py --skip-download
    python main.py --skip-download --skip-raster
    python main.py --data-config configs/data_configs.yaml --val-config configs/validation_configs.yaml

Prepared by: Rufai Omowunmi Balogun
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

from src.utils.runtime import colab_setup, is_colab
from src.downloader import UrbanDownloader
from src.validator import UrbanValidator

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_download(data_config: str) -> None:
    downloader = UrbanDownloader(data_config)
    log.info("Downloading: vector datasets")
    downloader.download_vector()
    log.info("Downloading: raster datasets")
    downloader.download_raster()
    log.info("Datasets download completed.")


def run_vector_validation(val_config: str) -> None:
    log.info("Vector validation: validating building footprints against reference data.")
    results = UrbanValidator(val_config).validate_vector()
    ok = sum(v for v in results.values())
    log.info("Vector validation complete — %d/%d cities succeeded.", ok, len(results))


def run_raster_validation(val_config: str) -> None:
    log.info("Raster validation: validating raster datasets against reference data.")
    results = UrbanValidator(val_config).validate_raster()
    ok = sum(v for v in results.values())
    log.info("Raster validation complete — %d/%d cities succeeded.", ok, len(results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Urban Validation pipeline runner")
    parser.add_argument("--data-config", default="configs/data_configs.yaml")
    parser.add_argument("--val-config", default="configs/validation_configs.yaml")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--skip-raster", action="store_true")
    parser.add_argument(
        "--project-root",
        default="/content/drive/MyDrive/Gates Foundation/Building Dataset Validation",
        help="Project root to use in Colab.",
    )

    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    if is_colab():
        project_root = args.project_root
        colab_setup(project_root)
        data_config = str(Path(project_root) / args.data_config)
        val_config = str(Path(project_root) / args.val_config)
    else:
        data_config = args.data_config
        val_config = args.val_config

    log.info("Using data config: %s", data_config)
    log.info("Using val config: %s", val_config)

    if not args.skip_download:
        run_download(data_config)

    if not args.skip_vector:
        run_vector_validation(val_config)

    if not args.skip_raster:
        run_raster_validation(val_config)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
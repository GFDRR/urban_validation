"""
Urban Validation — end-to-end pipeline runner.

Usage (Colab):
    %run main.py

Usage (local):
    python main.py
    python main.py --skip-download
    python main.py --skip-download --skip-raster
    python main.py --data-config configs/data_configs.yaml --val-config configs/validation_configs.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Colab detection ───────────────────────────────────────────────────────────

def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _colab_setup(project_root: str) -> None:
    from google.colab import drive
    drive.mount("/content/drive")
    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


# ── Pipeline steps ────────────────────────────────────────────────────────────

def run_download(data_config: str) -> None:
    from src.downloader import UrbanDownloader
    downloader = UrbanDownloader(data_config)
    log.info("=== Download: vector ===")
    downloader.download_vector()
    log.info("=== Download: raster ===")
    downloader.download_raster()


def run_vector_validation(val_config: str) -> None:
    from src.validator import Validator
    log.info("=== Validation: vector ===")
    results = Validator(val_config).validate_vector()
    ok  = sum(v for v in results.values())
    log.info("Vector validation complete — %d/%d cities succeeded.", ok, len(results))


def run_raster_validation(val_config: str) -> None:
    from src.validator import Validator
    log.info("=== Validation: raster ===")
    results = Validator(val_config).validate_raster()
    ok = sum(v for v in results.values())
    log.info("Raster validation complete — %d/%d cities succeeded.", ok, len(results))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Urban Validation pipeline runner")
    parser.add_argument("--data-config",   default="configs/data_configs.yaml")
    parser.add_argument("--val-config",    default="configs/validation_configs.yaml")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-vector",   action="store_true")
    parser.add_argument("--skip-raster",   action="store_true")

    # argparse raises SystemExit when called from Colab (%run / no argv),
    # so fall back to all-defaults in that case.
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    if _is_colab():
        project_root = "/content/drive/MyDrive/Gates Foundation/Building Dataset Validation"
        _colab_setup(project_root)
        data_config = f"{project_root}/{args.data_config}"
        val_config  = f"{project_root}/{args.val_config}"
    else:
        data_config = args.data_config
        val_config  = args.val_config

    if not args.skip_download:
        run_download(data_config)

    if not args.skip_vector:
        run_vector_validation(val_config)

    if not args.skip_raster:
        run_raster_validation(val_config)

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()

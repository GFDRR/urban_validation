"""
Urban Validation — end-to-end pipeline runner.

Usage (Colab notebook cell):
    %run main.py
    !python main.py --data-config configs/data_configs.yaml --val-config configs/validation_configs.yaml

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


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _in_ipython_kernel() -> bool:
    """
    True when running inside an interactive IPython/Colab notebook kernel.
    False for plain `python main.py`.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and getattr(ip, "kernel", None) is not None
    except Exception:
        return False


def _mount_drive_if_needed(mount_point: str = "/content/drive") -> None:
    """
    Mount Google Drive only when running in an interactive Colab notebook kernel.
    Skip mounting for plain python execution to avoid AttributeError from Colab internals.
    """
    if not _is_colab():
        return

    if os.path.exists(os.path.join(mount_point, "MyDrive")):
        log.info("Google Drive already available at %s", mount_point)
        return

    if not _in_ipython_kernel():
        log.warning(
            "Detected Colab package, but not an interactive notebook kernel. "
            "Skipping drive.mount(). If your files are in Drive, mount it first in a notebook cell."
        )
        return

    try:
        from google.colab import drive
        log.info("Mounting Google Drive at %s", mount_point)
        drive.mount(mount_point)
    except Exception as e:
        log.warning("Could not mount Google Drive automatically: %s", e)


def _colab_setup(project_root: str) -> None:
    _mount_drive_if_needed("/content/drive")

    if not os.path.exists(project_root):
        raise FileNotFoundError(
            f"Expected project root does not exist: {project_root}\n"
            "If running in Colab, make sure Drive is mounted and the path is correct."
        )

    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    log.info("Working directory set to: %s", project_root)


def run_download(data_config: str) -> None:
    from src.downloader import UrbanDownloader

    downloader = UrbanDownloader(data_config)
    log.info("=== Download: vector ===")
    downloader.download_vector()
    log.info("=== Download: raster ===")
    downloader.download_raster()


def run_vector_validation(val_config: str) -> None:
    from src.validator import UrbanValidator

    log.info("=== Validation: vector ===")
    results = UrbanValidator(val_config).validate_vector()
    ok = sum(v for v in results.values())
    log.info("Vector validation complete — %d/%d cities succeeded.", ok, len(results))


def run_raster_validation(val_config: str) -> None:
    from src.validator import UrbanValidator

    log.info("=== Validation: raster ===")
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

    if _is_colab():
        project_root = args.project_root
        _colab_setup(project_root)
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

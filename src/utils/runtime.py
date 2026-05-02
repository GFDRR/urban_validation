"""
Runtime detection and environment setup helpers.

Detect whether code is running inside a Colab notebook kernel vs a
plain script, mount Google Drive when appropriate, and bootstrap the
project working directory and import path.
"""
from __future__ import annotations

import logging
import os
import sys

log = logging.getLogger(__name__)


def is_colab() -> bool:
    """True when the google.colab package is importable."""
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


def mount_drive(mount_point: str = "/content/drive") -> None:
    """
    Mount Google Drive only when running in an interactive Colab notebook
    kernel. Skip mounting for plain python execution to avoid AttributeError
    from Colab internals.
    """
    if not is_colab():
        return

    if os.path.exists(os.path.join(mount_point, "MyDrive")):
        log.info("Google Drive already available at %s", mount_point)
        return

    if not _in_ipython_kernel():
        log.warning(
            "Detected Colab package, but not an interactive notebook kernel. "
            "Skipping drive.mount(). If your files are in Drive, mount it first "
            "in a notebook cell."
        )
        return

    try:
        from google.colab import drive
        log.info("Mounting Google Drive at %s", mount_point)
        drive.mount(mount_point)
    except Exception as e:
        log.warning("Could not mount Google Drive automatically: %s", e)


def colab_setup(project_root: str) -> None:
    """Mount Drive (if applicable), chdir into project_root, and add it to sys.path."""
    mount_drive("/content/drive")

    if not os.path.exists(project_root):
        raise FileNotFoundError(
            f"Expected project root does not exist: {project_root}\n"
            "If running in Colab, make sure Drive is mounted and the path is correct."
        )

    os.chdir(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    log.info("Working directory set to: %s", project_root)
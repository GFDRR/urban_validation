"""
Process memory logging.

Tiny helper for spotting RSS growth across stages of the pipeline.
Silently no-ops if psutil is unavailable.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def log_memory(label: str = "") -> None:
    """Log current process RSS — useful for spotting memory leaks."""
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / 1024 ** 2
        log.info("MEM [%s] RSS = %.0f MB", label, rss_mb)
    except Exception:
        pass

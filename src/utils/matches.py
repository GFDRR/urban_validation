"""
Match-chunk consolidation.

The vector validation pipeline flushes per-tile match DataFrames to
numbered temp parquet files to bound memory. This helper concatenates
them into a single output and removes the temp files.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def consolidate_match_chunks(
    metrics_dir: Path,
    ds_name: str,
    final_path: Path,
) -> None:
    """Read all temp match chunks for ds_name, concat, write final file, delete temps."""
    chunk_files = sorted(metrics_dir.glob(f"_tmp_matches_{ds_name}_*.parquet"))
    if chunk_files:
        df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
        df.to_parquet(final_path, index=False)
        del df
        for f in chunk_files:
            f.unlink()
    else:
        pd.DataFrame(columns=[
            "ref_id", "cand_id", "iou", "area_ref", "area_cand",
            "rel_area_error", "city", "dataset", "tile_id",
        ]).to_parquet(final_path, index=False)
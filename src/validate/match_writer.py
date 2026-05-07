"""
Buffered, chunk-flushed writer for per-tile match records.

During vector validation a city can produce millions of match rows
across all tiles. Holding them all in a single in-memory list before
writing exhausts RAM on dense cities. This writer accumulates match
DataFrames in a small buffer, flushes them to numbered temp parquet
files every N appends, then consolidates the temp files into one
parquet at the end.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.utils.matches import consolidate_match_chunks


class MatchChunkWriter:
    """
    Buffer per-tile match DataFrames and flush them to temp parquet
    files at a fixed interval to cap peak memory.

    Usage:
        writer = MatchChunkWriter(metrics_dir, ds_name, flush_interval=100)
        for tile in tiles:
            ...
            writer.append(matches_df)
        final_path = writer.finalize()  # consolidates and removes temp files
    """

    def __init__(
        self,
        metrics_dir: Path,
        ds_name: str,
        *,
        flush_interval: int = 100,
    ):
        self.metrics_dir = Path(metrics_dir)
        self.ds_name = ds_name
        self.flush_interval = int(flush_interval)
        self._buffer: List[pd.DataFrame] = []
        self._chunk_counter = 0
        self.final_path = self.metrics_dir / f"vector_matches_{ds_name}.parquet"

    def append(self, matches_df: pd.DataFrame) -> None:
        """Add a match DataFrame to the buffer; flush if interval reached."""
        if matches_df is None or matches_df.empty:
            return
        self._buffer.append(matches_df)
        if len(self._buffer) >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        """Write the current buffer to a numbered temp parquet file."""
        if not self._buffer:
            return
        tmp = (
            self.metrics_dir
            / f"_tmp_matches_{self.ds_name}_{self._chunk_counter:04d}.parquet"
        )
        pd.concat(self._buffer, ignore_index=True).to_parquet(tmp, index=False)
        self._chunk_counter += 1
        self._buffer.clear()

    def finalize(self) -> Path:
        """
        Flush any remaining buffered DataFrames, consolidate all temp
        chunk files into a single output parquet, and remove the temp
        files. Returns the final output path.
        """
        self.flush()
        consolidate_match_chunks(self.metrics_dir, self.ds_name, self.final_path)
        return self.final_path
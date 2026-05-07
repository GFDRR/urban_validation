"""
Vector runners: Overture, GBA, GloBFP.

Each runner is responsible for one source. They all inherit from
BaseVectorRunner, which supplies the DuckDB extract method and
sub-AOI tagging.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import geopandas as gpd
import pandas as pd

from src.download.base import BaseVectorRunner
from src.utils.glopfp import (
    download_globfp_grid_tile,
    ensure_world_grid,
    get_grid_ids_for_geometry,
)

log = logging.getLogger("UrbanDownloader.vector")


class OvertureRunner(BaseVectorRunner):
    """Download Overture Maps building footprints from S3."""

    name = "overture"

    def __init__(self, config, source_cfg, *, overwrite: bool = False, base_path: str):
        super().__init__(config, source_cfg, overwrite=overwrite)
        self.base_path = base_path

    def run(self, ds: dict, out_root: Path) -> List[str]:
        theme = self.source_cfg.theme
        types = self.source_cfg.types or []
        outputs: List[str] = []

        for typ in types:
            out_path = out_root / f"{ds['slug']}_overture_{typ}.parquet"
            results = self._extract(
                ds["aoi"],
                out_path,
                base_path=self.base_path,
                theme=theme,
                typ=typ,
            )
            self._tag_vector_output(out_path, ds)
            outputs += results

        return outputs


class GBARunner(BaseVectorRunner):
    """Download Global Building Atlas footprints from S3 parquet glob."""

    name = "gba"

    def run(self, ds: dict, out_root: Path) -> List[str]:
        glob = str(self.source_cfg.s3_url).rstrip("/")
        if "*" not in glob:
            glob += "/*.parquet"

        out_path = out_root / f"{ds['slug']}_gba.parquet"
        results = self._extract(ds["aoi"], out_path, parquet_glob=glob)
        self._tag_vector_output(out_path, ds)
        return results


class GloBFPRunner(BaseVectorRunner):
    """
    Download GloBFP building footprints by:
      1. resolving the global 1° grid tiles intersecting the AOI
      2. downloading each tile's shapefile
      3. running the AOI-clipped extract per tile
      4. merging per-tile outputs into a single parquet
    """

    name = "globfp"

    def __init__(self, config, source_cfg, *, overwrite: bool = False):
        super().__init__(config, source_cfg, overwrite=overwrite)
        self._world_grid: Path | None = None

    def prepare(self) -> None:
        """Ensure the world grid shapefile is available before running."""
        self._world_grid = ensure_world_grid(self.config)

    def run(self, ds: dict, out_root: Path) -> List[str]:
        if self._world_grid is None:
            self.prepare()

        final_path = out_root / f"{ds['slug']}_globfp.parquet"
        if final_path.exists() and not self.overwrite:
            log.debug("Skip existing: %s", final_path)
            return [str(final_path)]

        aoi_union = ds["aoi"].union_all()
        grid_ids = get_grid_ids_for_geometry(self._world_grid, aoi_union)
        log.info("GloBFP | %s | %d grid tile(s) intersect AOI.", ds["id"], len(grid_ids))

        if len(grid_ids) == 0:
            log.info("GloBFP | %s | no intersecting grid tiles for AOI; skipping.", ds["id"])
            return []

        per_grid_paths: List[Path] = []
        for grid_id in grid_ids:
            tmp_path = out_root / f"{ds['slug']}_globfp_grid{grid_id}.parquet"
            try:
                shp = download_globfp_grid_tile(self.config, grid_id)
                self._extract(ds["aoi"], tmp_path, shp_path=shp)
                per_grid_paths.append(tmp_path)
                log.info("GloBFP OK | %s | grid=%d -> %s", ds["id"], grid_id, tmp_path.name)
            except Exception:
                log.exception("GloBFP failed | %s | grid=%d", ds["id"], grid_id)

        if not per_grid_paths:
            log.warning(
                "GloBFP | %s | intersecting tiles found, but none downloaded successfully.",
                ds["id"],
            )
            return []

        if len(per_grid_paths) == 1:
            per_grid_paths[0].replace(final_path)
        else:
            log.info("GloBFP | %s | merging %d grid tiles.", ds["id"], len(per_grid_paths))
            merged = gpd.GeoDataFrame(
                pd.concat([gpd.read_parquet(p) for p in per_grid_paths], ignore_index=True),
                crs="EPSG:4326",
            )
            merged.to_parquet(final_path, index=False)
            for p in per_grid_paths:
                p.unlink(missing_ok=True)
            log.info(
                "GloBFP | %s | merged -> %s (%d buildings)",
                ds["id"],
                final_path.name,
                len(merged),
            )

        self._tag_vector_output(final_path, ds)
        return [str(final_path)]
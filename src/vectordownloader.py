"""Class to download vector datasets for Global Urban Validation workflow

Download utility functions for global building data."""

from __future__ import annotations
import os
import re
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import geopandas as gpd
import requests
from shapely.geometry import mapping

from src.config import load_config
from src.utils import load_aoi, ensure_world_grid, get_grid_ids_for_geometry, download_globfp_grid_tile

# logging info
logger = logging.getLogger("Urban_Vector_Downloader")
logger.setLevel(logging.INFO)
fmt = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
sh = logging.StreamHandler()
sh.setFormatter(fmt)
if not logger.handlers:
    logger.addHandler(sh)


OVERTURE_STAC_ROOT = "https://stac.overturemaps.org/catalog.json"

class UrbanVectorDownloader:
    """
    Unified Vector Datasets downloader for:
      - Overture Maps themes/types parquet
      - Global Building Atlas (GBA) parquet
      - 3D-GloBFP stored in parquet as well
    """

    def __init__(
        self,
        config_path: str,
        *,
        log: Optional[logging.Logger] = None,
    ):
        self.logger = log or logging.getLogger("Urban_Vector_Downloader")
        self.config = load_config(config_path)
        self.aoi =  load_aoi(
            self.config.aoi.path,
            crs_out=self.config.aoi.crs_out,
            buffer_meters=self.config.aoi.buffer_meters,
            dissolve=True,
            logger=self.logger,
        )

        self.out_root = self.infer_out_root()
        self.out_root.mkdir(parents=True, exist_ok=True)
        
        self.overwrite = bool(self.config.output.overwrite)
        self.results: Dict[str, List[str]] = {}

        self.logger.info("AOI columns: %s", list(self.aoi.columns))
        self.root_slug = self.infer_root_slug()

    def infer_out_root(self) -> Path:
        aoi_path = Path(self.config.aoi.path)
        root = aoi_path.parent.parent if aoi_path.parent.name.lower() == "aoi" else aoi_path.parent
        return root / "vector"
    
    def infer_root_slug(self) -> str:
        aoi_path = Path(self.config.aoi.path)
        root = aoi_path.parent.parent if aoi_path.parent.name.lower() == "aoi" else aoi_path.parent
        return root.name.replace("-", "_").replace(" ", "_")
    def dataset_prefix(self, ds_tag: str) -> str:
        return f"{self.root_slug}_{ds_tag}"
    

    def connect_db(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("INSTALL httpfs;  LOAD httpfs;")
        con.execute("INSTALL s3;      LOAD s3;")

        if getattr(self.config, "overture", None) and getattr(self.config.overture, "enabled", False):
            con.execute(f"SET s3_region='{self.config.overture.s3_region}';")
        elif getattr(self.config, "gba", None) and getattr(self.config.gba, "enabled", False):
            con.execute(f"SET s3_region='{self.config.gba.s3_region}';")
        con.execute("SET s3_url_style='path';")
        con.execute("SET s3_endpoint='s3.us-west-2.amazonaws.com';")
        con.execute("SET s3_use_ssl=true;")
        return con
    

    def _gba_parquet_glob(self) -> str:
        s = str(self.config.gba.s3_url).rstrip("/")
        return s if "*" in s else s + "/*.parquet"
        
    def get_overture_base_path(self) -> str:
        # Overture STAC root exposes ".latest" and is designed to keep scripts stable across releases
        rel = str(self.config.overture.release)
        if rel == "latest":
            self.logger.info("Resolving latest Overture release from STAC")
            j = requests.get(OVERTURE_STAC_ROOT, timeout=30).json()
            rel = j["latest"]
        return f"{self.config.overture.s3_url}{rel}".rstrip("/")

    def run_connection(self) -> List[str]:
        con = self.connect_db()
        try:
            gba_enabled      = getattr(self.config, "gba",     None) and self.config.gba.enabled
            overture_enabled = getattr(self.config, "overture", None) and self.config.overture.enabled
            globfp_enabled   = getattr(self.config, "globfp",  None) and self.config.globfp.enabled

            self.logger.info(
                "Download pipeline | aois=%d | overwrite=%s | gba=%s | overture=%s | globfp=%s",
                len(self.aoi), self.overwrite, gba_enabled, overture_enabled, globfp_enabled,
            )

            if not any([gba_enabled, overture_enabled, globfp_enabled]):
                raise ValueError(
                    "No data source enabled. Set gba.enabled, overture.enabled, "
                    "or globfp.enabled to true in your config."
                )

            outputs: List[str] = []

            if gba_enabled:
                outputs += self._run_gba(con)

            if overture_enabled:
                outputs += self._run_overture(con)

            if globfp_enabled:
                outputs += self._run_globfp(con)

            self.logger.info("Download complete | total outputs=%d", len(outputs))
            return outputs
        finally:
            con.close()


    def _extract_all_aoi_rows(
        self,
        con,
        out_name: str,
        **extract_kwargs,
    ) -> List[str]:
        """
        Iterate every AOI row, call extract_data_for_aoi with shared kwargs,
        and return the list of written output paths.
        `out_name` is the parquet filename (without directory).
        """
        outputs = []
        for row in self.aoi.itertuples(index=False):
            out_path = self.out_root / out_name
            try:
                out_file = self.extract_data_for_aoi(
                    con=con,
                    aoi_geom=row.geometry,
                    out_path=out_path,
                    overwrite=self.overwrite,
                    **extract_kwargs,
                )
                outputs.append(out_file)
                self.logger.info("Wrote: %s", out_file)
            except Exception:
                self.logger.exception("Failed | out_path=%s", out_path)
        return outputs


    def _run_gba(self, con) -> List[str]:
        parquet_glob = self._gba_parquet_glob()
        file_prefix = self.dataset_prefix("gba")
        self.logger.info("GBA enabled | glob=%s", parquet_glob)
        return self._extract_all_aoi_rows(
            con,
            out_name=f"{file_prefix}.parquet",
            parquet_glob=parquet_glob,
        )


    def _run_overture(self, con) -> List[str]:
        theme     = self.config.overture.theme
        types     = self.config.overture.types or []
        base_path = self.get_overture_base_path()
        file_prefix = self.dataset_prefix("overture")
        self.logger.info("Overture enabled | types=%s", types)

        outputs = []
        for typ in types:
            outputs += self._extract_all_aoi_rows(
                con,
                out_name=f"{file_prefix}_{typ}.parquet",
                base_path=base_path,
                theme=theme,
                typ=typ,
            )
        return outputs

    def _run_globfp(self, con) -> List[str]:
        self.logger.info("GloBFP enabled")
        world_grid_shp = ensure_world_grid(self.config)
        file_prefix = self.dataset_prefix("globfp")
        outputs = []

        for row in self.aoi.itertuples(index=False):
            for grid_id in get_grid_ids_for_geometry(world_grid_shp, row.geometry):
                # out_path = self.out_root / f"{file_prefix}_grid={grid_id}.parquet"
                out_path = self.out_root / f"{file_prefix}.parquet"
                try:
                    shp_path = download_globfp_grid_tile(self.config, grid_id)
                    out_file = self.extract_data_for_aoi(
                        con=con,
                        aoi_geom=row.geometry,
                        out_path=out_path,
                        overwrite=self.overwrite,
                        shp_path=shp_path,
                    )
                    outputs.append(out_file)
                    self.logger.info("GloBFP OK | grid=%d -> %s", grid_id, out_file)
                except Exception:
                    self.logger.exception("GloBFP failed | grid=%d", grid_id)

        return outputs

    def extract_data_for_aoi(
        self,
        *,
        con,
        aoi_geom,
        out_path,
        overwrite=False,
        # source: supply exactly one of these
        shp_path=None,
        parquet_glob=None,
        base_path=None,
        theme="",
        typ="",
    ):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            self.logger.debug("Skip existing: %s", out_path)
            return str(out_path)

        source_expr, bbox_filter = self.filter_by_source(
            shp_path=shp_path,
            parquet_glob=parquet_glob,
            base_path=base_path,
            theme=theme,
            typ=typ,
        )

        minx, miny, maxx, maxy = aoi_geom.bounds
        aoi_geojson = json.dumps(mapping(aoi_geom)).replace("'", "''")

        sql = f"""
            COPY (
                WITH aoi AS (SELECT ST_GeomFromGeoJSON('{aoi_geojson}') AS geom)
                SELECT b.*
                FROM {source_expr} AS b, aoi
                WHERE {bbox_filter.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy)}
                AND ST_Intersects(b.{self._geom_col(shp_path)}, aoi.geom)
            )
            TO '{out_path}' (FORMAT PARQUET);
        """
        self.logger.info(
            "Spatial query | source=%s | bounds=[%.4f, %.4f, %.4f, %.4f]",
            source_expr[:60], minx, miny, maxx, maxy,
        )
        con.execute(sql)
        return str(out_path)


    def filter_by_source(self, *, shp_path, parquet_glob, base_path, theme, typ):
        """Return (FROM expression, WHERE bbox template) for the requested source."""
        if shp_path is not None:
            expr   = f"ST_Read('{shp_path}')"
            bbox   = (
                "ST_XMin(ST_Envelope(b.geom)) <= {maxx}"
                " AND ST_XMax(ST_Envelope(b.geom)) >= {minx}"
                " AND ST_YMin(ST_Envelope(b.geom)) <= {maxy}"
                " AND ST_YMax(ST_Envelope(b.geom)) >= {miny}"
            )
        elif parquet_glob is not None:
            expr   = f"read_parquet('{parquet_glob}', hive_partitioning=true)"
            bbox   = (
                "b.bbox.xmin <= {maxx} AND b.bbox.xmax >= {minx}"
                " AND b.bbox.ymin <= {maxy} AND b.bbox.ymax >= {miny}"
            )
        elif base_path is not None:
            glob   = f"{base_path}/theme={theme}/type={typ}/*.parquet"
            expr   = f"read_parquet('{glob}', hive_partitioning=true)"
            bbox   = (
                "b.bbox.xmin <= {maxx} AND b.bbox.xmax >= {minx}"
                " AND b.bbox.ymin <= {maxy} AND b.bbox.ymax >= {miny}"
            )
        else:
            raise ValueError(
                "provide exactly one of: shp_path, parquet_glob, or base_path."
            )
        return expr, bbox


    @staticmethod
    def _geom_col(shp_path) -> str:
        """Shapefiles use 'geom'; parquet sources use 'geometry'."""
        return "geom" if shp_path is not None else "geometry"


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    UrbanVectorDownloader(config_path).run_connection()

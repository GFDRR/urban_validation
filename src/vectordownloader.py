"""Class to download vector datasets for Global Urban Validation workflow

Download utility functions for global building data."""


from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import duckdb
import fiona
import geopandas as gpd
import requests
import yaml
from shapely.geometry import mapping

from src.config import load_config

logger = logging.getLogger("Urban_Vector_Downloader")
logger.setLevel(logging.INFO)
fmt = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
sh = logging.StreamHandler()
sh.setFormatter(fmt)
logger.addHandler(sh)


OVERTURE_STAC_ROOT = "https://stac.overturemaps.org/catalog.json"

class UrbanVectorDownloader:
    """
    Unified downloader for:
      - Overture Maps themes/types parquet
      - Global Building Atlas (GBA) parquet
    """

    def __init__(
        self,
        config_path: str,
        *,
        log: Optional[logging.Logger] = None,
    ):
        self.logger = log or logging.getLogger("Urban_Vector_Downloader")
        self.config = load_config(config_path)
        self._aoi: Optional[gpd.GeoDataFrame] = None
        
        self.aoi = self.read_aoi()
        self.out_root = Path(self.config.output.root_dir)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.overwrite = bool(self.config.output.overwrite)
        self.results: Dict[str, List[str]] = {}


    def read_aoi(self):
        """Read the AOI (Area of Interest) from the specified path."""
        aoi_path = self.config.aoi.path
        crs_out = self.config.aoi.crs_out
        buff = float(self.config.aoi.buffer_meters or 0)

        aoi = gpd.read_file(aoi_path)
        if aoi.crs is None:
            self.logger.warning("AOI CRS missing; assuming EPSG:4326")
            aoi = aoi.set_crs("EPSG:4326")

        if buff > 0:
            if aoi.crs is not None and not aoi.crs.is_projected:
                self.logger.warning(
                    "AOI CRS is geographic; reprojecting to EPSG:3857 for buffering"
                )
                aoi_metric = aoi.to_crs("EPSG:3857")
                aoi_metric["geometry"] = aoi_metric.geometry.buffer(buff)
                aoi = aoi_metric.to_crs(aoi.crs)
            else:
                aoi["geometry"] = aoi.geometry.buffer(buff)

        if str(aoi.crs) != str(crs_out):
            self.logger.info("Reprojecting AOI | %s -> %s", aoi.crs, crs_out)
            aoi = aoi.to_crs(crs_out)

        self.logger.info("Successfully loaded AOI  | rows=%d | crs=%s", len(aoi), aoi.crs)

        return aoi

    def connect_db(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL s3;      LOAD s3;") 

        if self.config.overture.enabled:
            con.execute(f"SET s3_region='{self.config.overture.s3_region}';")
            con.execute("SET s3_url_style='path';")
            con.execute("SET s3_endpoint='s3.us-west-2.amazonaws.com';")
            con.execute("SET s3_use_ssl=true;")
        elif self.config.gba.enabled:
            con.execute(f"SET s3_region='{self.config.gba.s3_region}';")
            con.execute("SET s3_url_style='path';")
        return con
    
    def get_overture_base_path(self) -> str:
        # Overture STAC root exposes ".latest" and is designed to keep scripts stable across releases
        rel = str(self.config.overture.release)
        if rel == "latest":
            self.logger.info("Resolving latest Overture release from STAC")
            j = requests.get(OVERTURE_STAC_ROOT, timeout=30).json()
            rel = j["latest"]
        return f"{self.config.overture.s3_url}{rel}".rstrip("/")
    
    def run_connection(self):
        con = self.connect_db()
        try:
            theme = self.config.overture.theme
            types = self.config.overture.types
            base_path = self.get_overture_base_path() if self.config.overture.enabled else None

            self.logger.info(
                "Running Download Pipeline for aois=%d | overwrite=%s",
                len(self.aoi),
                self.overwrite,
            )
            id_col = self.config.aoi.id_col
            outputs = []
            for i, row in enumerate(self.aoi.itertuples(index=False), start=1):
                aoi_id = str(getattr(row, id_col))
                geom = getattr(row, "geometry")

                for typ in types:
                    out_path = self.out_root / f"aoi={aoi_id}" / f"{theme}_{typ}.parquet"
                    try:
                        out_file = self.extract_data_for_aoi(
                            con=con,
                            aoi_geom=geom,
                            out_path=out_path,
                            overwrite=self.overwrite,
                            base_path=base_path,
                            theme=theme,
                            typ=typ,
                        )
                        outputs.append(out_file)
                        self.logger.info("Overture wrote: %s", out_file)
                    except Exception:
                        self.logger.exception("Overture failed | aoi_id=%s | type=%s", aoi_id, typ)
            self.logger.info("Data Download completed | outputs=%d", len(outputs))
            return outputs
        
        finally: 
            con.close()

    def extract_data_for_aoi(
            self,
            *,
            con: duckdb.DuckDBPyConnection,
            base_path: str,
            theme: str,
            typ: str,
            aoi_geom,
            out_path: Path,
            overwrite: bool = False,
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            self.logger.debug("Overture skip existing: %s", out_path)
            return str(out_path)

        minx, miny, maxx, maxy = aoi_geom.bounds
        aoi_geojson = json.dumps(mapping(aoi_geom)).replace("'", "''")
        if self.config.gba.enabled:
            parquet_glob = f"{self.config.gba.s3_url}/*.parquet"
        elif self.config.overture.enabled:
            parquet_glob = f"{base_path}/theme={theme}/type={typ}/*.parquet"

        self.logger.debug("Parquet glob: %s", parquet_glob)

        sql = f"""
            COPY (
                WITH aoi AS (
                    SELECT ST_GeomFromGeoJSON('{aoi_geojson}') AS geom
                )
                SELECT b.*
                FROM read_parquet('{parquet_glob}') AS b, aoi
                WHERE
                    ST_XMin(ST_Envelope(b.geometry)) <= {maxx} AND ST_XMax(ST_Envelope(b.geometry)) >= {minx}
                    AND ST_YMin(ST_Envelope(b.geometry)) <= {maxy} AND ST_YMax(ST_Envelope(b.geometry)) >= {miny}
                    AND ST_Intersects(b.geometry, aoi.geom)
                            )
            TO '{str(out_path)}'
            (FORMAT PARQUET);
            """
        self.logger.info("Parquet glob: %s", parquet_glob)
        con.execute(sql)
        return str(out_path)
    

if __name__ == "__main__":
    config_path = "configs/config.yaml"
    UrbanVectorDownloader(config_path).run_connection()

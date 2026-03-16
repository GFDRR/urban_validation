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
from src.utils import load_aoi

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

        # self.aoi = self.read_aoi()
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

    def infer_out_root(self) -> Path:
        """
        Build output root dynamically from the AOI path.

        Expected AOI layout:
            <root>/aoi/<file>

        Output layout:
            <root>/vector/
        """
        aoi_path = Path(self.config.aoi.path)

        if aoi_path.parent.name.lower() == "aoi":
            root = aoi_path.parent.parent
        else:
            root = aoi_path.parent

        root_slug = root.name.replace("-", "_").replace(" ", "_")

        gba_on = getattr(self.config, "gba", None) and getattr(self.config.gba, "enabled", False)
        ov_on  = getattr(self.config, "overture", None) and getattr(self.config.overture, "enabled", False)

        if gba_on and ov_on:
            ds_tag = "vector"
        elif gba_on:
            ds_tag = "gba"
        elif ov_on:
            ds_tag = "overture"
        else:
            ds_tag = "vector"

        # store prefix for filenames
        self.file_prefix = f"{root_slug}_{ds_tag}"

        return root / "vector"

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
        # If country partition is known, scope the glob to that partition only
        country = getattr(self.config.gba, "country_iso", None)
        if country:
            self.logger.info("GBA glob scoped to country=%s", country)
            return f"{s}/country={country.upper()}/*.parquet"
        return s if "*" in s else s + "/*.parquet"
        
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
            self.logger.info(
                "Running Download Pipeline for aois=%d | overwrite=%s",
                len(self.aoi),
                self.overwrite,
            )

            id_col = self.config.aoi.id_col
            
            # fall back to row index if id_col not present in AOI
            if id_col not in self.aoi.columns:
                self.logger.warning(
                    "id_col='%s' not found in AOI columns %s — falling back to row index",
                    id_col, list(self.aoi.columns)
                )
                id_col = None  # to use enumerate index below
            outputs = []
            
            gba_enabled = getattr(self.config, "gba", None) and getattr(self.config.gba, "enabled", False)
            overture_enabled = getattr(self.config, "overture", None) and getattr(self.config.overture, "enabled", False)

            self.logger.info("Pipeline flags | gba=%s | overture=%s", gba_enabled, overture_enabled)

            if not gba_enabled and not overture_enabled:
                raise ValueError("No data source enabled in config. Set gba.enabled or overture.enabled to true.")

            # --- GBA run (one output per AOI) ---
            if gba_enabled:
                parquet_glob = self._gba_parquet_glob()
                self.logger.info("GBA enabled | parquet_glob=%s", parquet_glob)

                for i, row in enumerate(self.aoi.itertuples(index=False), start=1):
                    # aoi_id = str(getattr(row, id_col))
                    aoi_id = str(i) if id_col is None else str(getattr(row, id_col))
                    geom = getattr(row, "geometry")
                    out_path = self.out_root / f"{self.file_prefix}_{aoi_id}_gba_lod1.parquet"

                    try:
                        out_file = self.extract_data_for_aoi(
                            con=con,
                            aoi_geom=geom,
                            out_path=out_path,
                            overwrite=self.overwrite,
                            base_path=None,
                            theme="",
                            typ="",
                            parquet_glob_override=parquet_glob,   # add this param
                        )
                        outputs.append(out_file)
                        self.logger.info("GBA wrote: %s", out_file)
                    except Exception:
                        self.logger.exception("GBA failed | aoi_id=%s", aoi_id)

            # --- Overture run (per type) ---
            if overture_enabled:
                theme = self.config.overture.theme
                types = self.config.overture.types or []
                self.logger.info("Overture enabled | types=%s", types)

                base_path = self.get_overture_base_path()
                for i, row in enumerate(self.aoi.itertuples(index=False), start=1):
                    # aoi_id = str(getattr(row, id_col))
                    aoi_id = str(i) if id_col is None else str(getattr(row, id_col))
                    geom = getattr(row, "geometry")

                    for typ in types:
                        out_path = self.out_root / f"{self.file_prefix}_{aoi_id}_{typ}.parquet"
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
        
    def extract_data_for_aoi(self, *, con, base_path, theme, typ, aoi_geom,
                                out_path, overwrite=False, parquet_glob_override=None):
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not overwrite:
                self.logger.debug("Skip existing: %s", out_path)
                return str(out_path)

            minx, miny, maxx, maxy = aoi_geom.bounds
            aoi_geojson = json.dumps(mapping(aoi_geom)).replace("'", "''")

            if parquet_glob_override:
                parquet_glob = parquet_glob_override
            elif getattr(self.config, "overture", None) and self.config.overture.enabled:
                parquet_glob = f"{base_path}/theme={theme}/type={typ}/*.parquet"
            else:
                raise ValueError("Cannot determine parquet_glob.")

            # GBA parquet files store explicit bbox columns: bbox.xmin, bbox.ymin etc.
            # Use these for cheap column-statistic pushdown BEFORE the expensive ST_Intersects
            sql = f"""
                COPY (
                    WITH aoi AS (
                        SELECT ST_GeomFromGeoJSON('{aoi_geojson}') AS geom
                    ),
                    bbox_filtered AS (
                        SELECT b.*
                        FROM read_parquet('{parquet_glob}', hive_partitioning=true) AS b
                        WHERE
                            b.bbox.xmin <= {maxx}
                            AND b.bbox.xmax >= {minx}
                            AND b.bbox.ymin <= {maxy}
                            AND b.bbox.ymax >= {miny}
                    )
                    SELECT f.*
                    FROM bbox_filtered f, aoi
                    WHERE ST_Intersects(f.geometry, aoi.geom)
                )
                TO '{str(out_path)}' (FORMAT PARQUET);
            """
            self.logger.info("Executing spatial query | bounds=[%.4f,%.4f,%.4f,%.4f]", minx, miny, maxx, maxy)
            con.execute(sql)
            return str(out_path)
    

if __name__ == "__main__":
    config_path = "configs/config.yaml"
    UrbanVectorDownloader(config_path).run_connection()

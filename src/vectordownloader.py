from __future__ import annotations
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import geopandas as gpd
import requests
from shapely.geometry import mapping

from src.config import load_config
from src.utils import load_all_aois, ensure_world_grid, get_grid_ids_for_geometry, download_globfp_grid_tile

logger = logging.getLogger("Urban_Vector_Downloader")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
sh = logging.StreamHandler()
sh.setFormatter(fmt)
if not logger.handlers:
    logger.addHandler(sh)

OVERTURE_STAC_ROOT = "https://stac.overturemaps.org/catalog.json"


class UrbanVectorDownloader:
    """
    Unified downloader for Overture / GBA / GloBFP building datasets.
    Accepts either a single AOI file or a CSV reference inventory as input.
    """

    def __init__(self, config_path: str, *, log: Optional[logging.Logger] = None):
        self.logger = log or logging.getLogger("Urban_Vector_Downloader")
        self.config = load_config(config_path)
        self.overwrite = bool(self.config.output.overwrite)

        # Load the dataset inventory: list of (dataset_id, dissolved_aoi_gdf, out_root)
        self.datasets = load_all_aois(self.config)
        self.logger.info("Loaded %d dataset(s) from inventory", len(self.datasets))


    def _enabled_sources(self):
        gba_on    = getattr(self.config, "gba",     None) and self.config.gba.enabled
        ov_on     = getattr(self.config, "overture", None) and self.config.overture.enabled
        globfp_on = getattr(self.config, "globfp",  None) and self.config.globfp.enabled
        return gba_on, ov_on, globfp_on

    def _dataset_prefix(self, slug: str, ds_tag: str) -> str:
        return f"{slug}_{ds_tag}"

    def _gba_parquet_glob(self) -> str:
        s = str(self.config.gba.s3_url).rstrip("/")
        return s if "*" in s else s + "/*.parquet"

    def get_overture_base_path(self) -> str:
        rel = str(self.config.overture.release)
        if rel == "latest":
            self.logger.info("Resolving latest Overture release from STAC")
            j = requests.get(OVERTURE_STAC_ROOT, timeout=30).json()
            rel = j["latest"]
        return f"{self.config.overture.s3_url}{rel}".rstrip("/")

    def connect_db(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("INSTALL httpfs;  LOAD httpfs;")
        con.execute("INSTALL s3;      LOAD s3;")
        if getattr(self.config, "overture", None) and self.config.overture.enabled:
            con.execute(f"SET s3_region='{self.config.overture.s3_region}';")
        elif getattr(self.config, "gba", None) and self.config.gba.enabled:
            con.execute(f"SET s3_region='{self.config.gba.s3_region}';")
        con.execute("SET s3_url_style='path';")
        con.execute("SET s3_endpoint='s3.us-west-2.amazonaws.com';")
        con.execute("SET s3_use_ssl=true;")
        return con

    def run_connection(self) -> Dict[str, List[str]]:
        """
        Run the full download pipeline across all datasets.
        Returns a dict of {dataset_id: [output_paths]}.
        """
        gba_on, ov_on, globfp_on = self._enabled_sources()
        if not any([gba_on, ov_on, globfp_on]):
            raise ValueError(
                "No data source enabled. Set gba.enabled, overture.enabled, "
                "or globfp.enabled to true in your config."
            )

        # Resolve Overture base path once (makes one HTTP call if release=latest)
        overture_base = self.get_overture_base_path() if ov_on else None
        globfp_grid   = ensure_world_grid(self.config) if globfp_on else None

        all_outputs: Dict[str, List[str]] = {}
        con = self.connect_db()
        try:
            for ds in self.datasets:
                self.logger.info(
                    "── Dataset: %s | sources: gba=%s overture=%s globfp=%s",
                    ds["id"], gba_on, ov_on, globfp_on,
                )
                outputs = []

                if gba_on:
                    outputs += self._run_gba(con, ds)
                if ov_on:
                    outputs += self._run_overture(con, ds, overture_base)
                if globfp_on:
                    outputs += self._run_globfp(con, ds, globfp_grid)

                all_outputs[ds["id"]] = outputs
                self.logger.info("Dataset %s complete | outputs=%d", ds["id"], len(outputs))
        finally:
            con.close()

        total = sum(len(v) for v in all_outputs.values())
        self.logger.info("All datasets complete | total outputs=%d", total)
        return all_outputs

    def _run_gba(self, con, ds: Dict) -> List[str]:
        parquet_glob = self._gba_parquet_glob()
        prefix       = self._dataset_prefix(ds["slug"], "gba")
        self.logger.info("GBA | dataset=%s | glob=%s", ds["id"], parquet_glob)
        return self._extract_for_dataset(
            con, ds,
            out_name=f"{prefix}.parquet",
            parquet_glob=parquet_glob,
        )

    def _run_overture(self, con, ds: Dict, base_path: str) -> List[str]:
        theme   = self.config.overture.theme
        types   = self.config.overture.types or []
        prefix  = self._dataset_prefix(ds["slug"], "overture")
        self.logger.info("Overture | dataset=%s | types=%s", ds["id"], types)
        outputs = []
        for typ in types:
            outputs += self._extract_for_dataset(
                con, ds,
                out_name=f"{prefix}_{typ}.parquet",
                base_path=base_path,
                theme=theme,
                typ=typ,
            )
        return outputs

    def _run_globfp(self, con, ds: Dict, world_grid_shp: Path) -> List[str]:
        prefix  = self._dataset_prefix(ds["slug"], "globfp")
        outputs = []
        aoi     = ds["aoi"]

        for row in aoi.itertuples(index=False):
            for grid_id in get_grid_ids_for_geometry(world_grid_shp, row.geometry):
                out_path = ds["out_root"] / f"{prefix}_grid={grid_id}.parquet"
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
                    self.logger.info("GloBFP OK | dataset=%s grid=%d -> %s",
                                     ds["id"], grid_id, out_file)
                except Exception:
                    self.logger.exception("GloBFP failed | dataset=%s grid=%d", ds["id"], grid_id)
        return outputs

    def _extract_for_dataset(self, con, ds: Dict, out_name: str, **extract_kwargs) -> List[str]:
        """Iterate AOI rows for one dataset and call extract_data_for_aoi."""
        outputs = []
        for row in ds["aoi"].itertuples(index=False):
            out_path = ds["out_root"] / out_name
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
                self.logger.exception("Failed | dataset=%s out=%s", ds["id"], out_path)
        return outputs

    def extract_data_for_aoi(self, *, con, aoi_geom, out_path, overwrite=False,
                              shp_path=None, parquet_glob=None, base_path=None,
                              theme="", typ=""):
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not overwrite:
            self.logger.debug("Skip existing: %s", out_path)
            return str(out_path)

        source_expr, bbox_filter = self.filter_by_source(
            shp_path=shp_path, parquet_glob=parquet_glob,
            base_path=base_path, theme=theme, typ=typ,
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
        self.logger.info("Spatial query | source=%s | bounds=[%.4f,%.4f,%.4f,%.4f]",
                         source_expr[:60], minx, miny, maxx, maxy)
        con.execute(sql)
        return str(out_path)

    def filter_by_source(self, *, shp_path, parquet_glob, base_path, theme, typ):
        if shp_path is not None:
            return (
                f"ST_Read('{shp_path}')",
                "ST_XMin(ST_Envelope(b.geom)) <= {maxx} AND ST_XMax(ST_Envelope(b.geom)) >= {minx}"
                " AND ST_YMin(ST_Envelope(b.geom)) <= {maxy} AND ST_YMax(ST_Envelope(b.geom)) >= {miny}",
            )
        if parquet_glob is not None:
            glob = parquet_glob
        elif base_path is not None:
            glob = f"{base_path}/theme={theme}/type={typ}/*.parquet"
        else:
            raise ValueError("Provide exactly one of: shp_path, parquet_glob, or base_path.")
        return (
            f"read_parquet('{glob}', hive_partitioning=true)",
            "b.bbox.xmin <= {maxx} AND b.bbox.xmax >= {minx}"
            " AND b.bbox.ymin <= {maxy} AND b.bbox.ymax >= {miny}",
        )

    @staticmethod
    def _geom_col(shp_path) -> str:
        return "geom" if shp_path is not None else "geometry"


if __name__ == "__main__":
    BASE   = "/content/drive/MyDrive/Gates Foundation/Building Dataset Validation"
    CONFIG = f"{BASE}/configs/overture.yaml"
    UrbanVectorDownloader(CONFIG).run_connection()

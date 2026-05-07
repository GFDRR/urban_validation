"""
Abstract base classes for download runners.

BaseRunner:        defines the runner contract (run for one dataset).
BaseVectorRunner:  owns the shared DuckDB spatial-extract logic and
                   sub-AOI building tagging.
BaseRasterRunner:  owns the shared multi-AOI raster masking helper.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import geopandas as gpd
from shapely import wkb
from shapely.geometry import mapping

from src.utils.raster_io import mask_raster_to_sub_aois
from src.utils.aoi_inventory import tag_buildings_with_sub_aoi

log = logging.getLogger("UrbanDownloader.runner")


class BaseRunner(ABC):
    """
    Abstract runner. Each concrete subclass downloads exactly one source
    (e.g. Overture, GHSL) for a single dataset.
    """

    name: str = "base"

    def __init__(self, config, source_cfg, *, overwrite: bool = False):
        self.config = config
        self.source_cfg = source_cfg
        self.overwrite = overwrite

    @abstractmethod
    def run(self, ds: dict, out_root: Path) -> List[str]:
        """Run this source for a single dataset and return output paths."""
        raise NotImplementedError


class BaseVectorRunner(BaseRunner):
    """
    Vector runners share:
      - a DuckDB connection (passed at run time)
      - the spatial bbox+intersect extract pattern
      - sub-AOI tagging on the resulting parquet
    """

    def __init__(self, config, source_cfg, *, overwrite: bool = False):
        super().__init__(config, source_cfg, overwrite=overwrite)
        self.con = None  # set by main downloader before calling run()

    def bind_connection(self, con) -> None:
        """Attach a shared DuckDB connection for this run cycle."""
        self.con = con

    # -----------------------------------------------------------------
    # Spatial extract — shared by Overture, GBA, GloBFP
    # -----------------------------------------------------------------

    def _extract(
        self,
        aoi_gdf,
        out_path: Path,
        *,
        shp_path=None,
        parquet_glob=None,
        base_path=None,
        theme: str = "",
        typ: str = "",
    ) -> List[str]:
        """
        Spatially query a source dataset clipped to the AOI and write a
        parquet file. Requires bind_connection() to have been called.
        """
        if self.con is None:
            raise RuntimeError(
                f"{self.__class__.__name__}._extract called before bind_connection()."
            )

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not self.overwrite:
            log.debug("Skip existing: %s", out_path)
            return [str(out_path)]

        if shp_path is not None:
            source = f"ST_Read('{shp_path}')"
            geom_col = "geom"
            bbox_where = (
                "ST_XMin(ST_Envelope(b.geom)) <= {maxx} AND ST_XMax(ST_Envelope(b.geom)) >= {minx}"
                " AND ST_YMin(ST_Envelope(b.geom)) <= {maxy} AND ST_YMax(ST_Envelope(b.geom)) >= {miny}"
            )
        else:
            glob = parquet_glob or f"{base_path}/theme={theme}/type={typ}/*.parquet"
            source = f"read_parquet('{glob}', hive_partitioning=true)"
            geom_col = "geometry"
            bbox_where = (
                "b.bbox.xmin <= {maxx} AND b.bbox.xmax >= {minx}"
                " AND b.bbox.ymin <= {maxy} AND b.bbox.ymax >= {miny}"
            )

        aoi_geom = aoi_gdf.union_all()
        minx, miny, maxx, maxy = aoi_geom.bounds
        aoi_json = json.dumps(mapping(aoi_geom)).replace("'", "''")
        bbox = bbox_where.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

        if shp_path is not None:
            sql = f"""
                WITH aoi AS (SELECT ST_GeomFromGeoJSON('{aoi_json}') AS geom)
                SELECT * EXCLUDE ({geom_col}), ST_AsWKB(b.{geom_col}) AS geometry
                FROM {source} AS b, aoi
                WHERE {bbox} AND ST_Intersects(b.{geom_col}, aoi.geom)
            """
            df = self.con.execute(sql).df()
            if not df.empty:
                df["geometry"] = df["geometry"].apply(
                    lambda x: wkb.loads(bytes(x)) if x is not None else None
                )
            gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326").to_parquet(
                out_path,
                index=False,
            )
        else:
            sql = f"""
                COPY (
                    WITH aoi AS (SELECT ST_GeomFromGeoJSON('{aoi_json}') AS geom)
                    SELECT b.*
                    FROM {source} AS b, aoi
                    WHERE {bbox} AND ST_Intersects(b.{geom_col}, aoi.geom)
                ) TO '{out_path}' (FORMAT PARQUET);
            """
            self.con.execute(sql)

        log.info("Wrote: %s", out_path)
        return [str(out_path)]

    # -----------------------------------------------------------------
    # Sub-AOI tagging — applied to every vector output
    # -----------------------------------------------------------------

    def _tag_vector_output(self, parquet_path: Path, ds: dict) -> None:
        if not parquet_path.exists():
            return

        sub_aois = ds.get("sub_aois", [])
        if not sub_aois:
            return

        try:
            gdf = gpd.read_parquet(parquet_path)
        except Exception:
            log.warning("Could not read %s for sub-AOI tagging.", parquet_path)
            return

        if gdf.empty:
            return

        n_before = len(gdf)
        gdf = tag_buildings_with_sub_aoi(gdf, sub_aois)
        n_after = len(gdf)

        if ds.get("is_multi_aoi") and n_before != n_after:
            log.info(
                "Sub-AOI tagging | %s | %s | %d -> %d buildings (%d in gaps removed)",
                ds["id"], parquet_path.name, n_before, n_after, n_before - n_after,
            )

        gdf.to_parquet(parquet_path, index=False)


class BaseRasterRunner(BaseRunner):
    """
    Raster runners share the multi-AOI mask post-processing step
    (clip a downloaded raster to the union of sub-AOIs in place).
    """

    def _mask_if_multi_aoi(
        self,
        raster_path: Path,
        ds: dict,
        nodata: float = 0.0,
    ) -> None:
        """Apply sub-AOI mask to a raster if this is a multi-AOI dataset."""
        if not ds.get("is_multi_aoi"):
            return
        if not Path(raster_path).exists():
            return

        sub_aois = ds.get("sub_aois", [])
        if not sub_aois:
            return

        log.info(
            "Masking raster to %d sub-AOI(s) | %s | %s",
            len(sub_aois),
            ds["id"],
            raster_path,
        )
        mask_raster_to_sub_aois(
            Path(raster_path),
            sub_aois,
            nodata=nodata,
            in_place=True,
        )
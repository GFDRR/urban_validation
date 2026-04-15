"""
Building dataset downloader — vector (Overture / GBA / GloBFP) and
raster (Google OBT / Microsoft TEMPO / GHSL).

Usage:
    downloader = UrbanDownloader("configs/data_configs.yaml")
    downloader.download_vector()   # Overture, GBA, GloBFP
    downloader.download_raster()   # Google OBT, TEMPO, GHSL
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import ee
import geopandas as gpd
import pandas as pd
import rasterio
import requests
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from shapely import wkb
from shapely.geometry import box, mapping

from src.config import load_config
from src.utils import (
    _shapely_to_geojson_dict,
    aoi_gdf_to_ee_geometry,
    download_file,
    download_globfp_grid_tile,
    ensure_world_grid,
    get_grid_ids_for_geometry,
    get_tile_url_col,
    init_earth_engine,
    load_all_aois,
    mask_raster_to_sub_aois,
    reproject_to_4326,
    resolve_out_root,
    tag_buildings_with_sub_aoi,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("UrbanDownloader")

OVERTURE_STAC_ROOT = "https://stac.overturemaps.org/catalog.json"


class UrbanDownloader:
    """
    Downloads building datasets for all cities listed in the AOI inventory.

    Vector sources: Overture Maps, Global Building Atlas (GBA), GloBFP
    Raster sources: Google Open Buildings Temporal (OBT), Microsoft TEMPO, GHSL

    Enable/disable each source in configs/data_configs.yaml.
    """

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.overwrite = self.config.output.overwrite
        self.datasets = load_all_aois(self.config)
        log.info("Loaded %d dataset(s) from inventory.", len(self.datasets))

    # ─────────────────────────────────────────────────────────────────
    # VECTOR
    # ─────────────────────────────────────────────────────────────────

    def download_vector(self) -> Dict[str, List[str]]:
        """Download Overture, GBA, and GloBFP footprints for all datasets."""
        cfg = self.config
        enabled = {
            "overture": cfg.overture.enabled,
            "gba": cfg.gba.enabled,
            "globfp": cfg.globfp.enabled,
        }
        if not any(enabled.values()):
            raise ValueError(
                "No vector source enabled in config. "
                "Set overture.enabled, gba.enabled, or globfp.enabled to true."
            )

        overture_base = self._resolve_overture_base() if enabled["overture"] else None
        globfp_grid = ensure_world_grid(cfg) if enabled["globfp"] else None

        all_outputs: Dict[str, List[str]] = {}
        con = self._connect_db()
        try:
            for ds in self.datasets:
                out_root = resolve_out_root(cfg, ds["id"], "vector")
                outputs: List[str] = []
                if enabled["gba"]:
                    outputs += self._run_gba(con, ds, out_root)
                if enabled["overture"]:
                    outputs += self._run_overture(con, ds, out_root, overture_base)
                if enabled["globfp"]:
                    outputs += self._run_globfp(con, ds, out_root, globfp_grid)
                all_outputs[ds["id"]] = outputs
                log.info("Vector done | %s | %d file(s)", ds["id"], len(outputs))
        finally:
            con.close()

        log.info(
            "Vector download complete | total=%d",
            sum(len(v) for v in all_outputs.values()),
        )
        return all_outputs

    # ─────────────────────────────────────────────────────────────────
    # RASTER
    # ─────────────────────────────────────────────────────────────────

    def download_raster(self) -> Dict[str, List[str]]:
        """Download Google OBT, Microsoft TEMPO, and GHSL rasters for all datasets."""
        cfg = self.config
        sources = {
            "google_open_buildings_temporal": cfg.datasets.google_open_buildings_temporal,
            "microsoft_tempo": cfg.datasets.microsoft_tempo,
            "ghsl": cfg.datasets.ghsl,
        }
        enabled = {name: source_cfg.enabled for name, source_cfg in sources.items()}
        if not any(enabled.values()):
            raise ValueError(
                "No raster source enabled in config. "
                "Set datasets.google_open_buildings_temporal.enabled, "
                "datasets.microsoft_tempo.enabled, or datasets.ghsl.enabled to true."
            )

        needs_ee = enabled.get("google_open_buildings_temporal") or enabled.get("ghsl")
        if needs_ee:
            init_earth_engine(cfg.ee_project)

        runners = {
            "google_open_buildings_temporal": self._run_obt,
            "microsoft_tempo": self._run_tempo,
            "ghsl": self._run_ghsl,
        }

        all_outputs: Dict[str, List[str]] = {}
        for ds in self.datasets:
            out_root = resolve_out_root(cfg, ds["id"], "raster")
            outputs: List[str] = []
            for name, runner in runners.items():
                if not enabled[name]:
                    continue
                log.info("  ── %s | %s ──", name, ds["id"])
                try:
                    outputs += runner(ds, out_root, sources[name])
                except Exception:
                    log.exception("  [%s] FAILED for dataset %s", name, ds["id"])
            all_outputs[ds["id"]] = outputs
            log.info("Raster done | %s | %d file(s)", ds["id"], len(outputs))

        log.info(
            "Raster download complete | total=%d",
            sum(len(v) for v in all_outputs.values()),
        )
        return all_outputs

    # ─────────────────────────────────────────────────────────────────
    # INTERNALS — DB connection
    # ─────────────────────────────────────────────────────────────────

    def _connect_db(self) -> duckdb.DuckDBPyConnection:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("INSTALL httpfs;  LOAD httpfs;")
        con.execute("INSTALL s3;      LOAD s3;")

        region = (
            self.config.overture.s3_region
            if self.config.overture.enabled
            else self.config.gba.s3_region
        )
        con.execute(f"SET s3_region='{region}';")
        con.execute("SET s3_url_style='path';")
        con.execute("SET s3_endpoint='s3.us-west-2.amazonaws.com';")
        con.execute("SET s3_use_ssl=true;")
        return con

    def _resolve_overture_base(self) -> str:
        release = str(self.config.overture.release)
        if release == "latest":
            log.info("Resolving latest Overture release from STAC …")
            j = requests.get(OVERTURE_STAC_ROOT, timeout=30).json()
            release = j["latest"]
        return f"{self.config.overture.s3_url}{release}".rstrip("/")

    # ─────────────────────────────────────────────────────────────────
    # INTERNALS — Earth Engine direct download
    # ─────────────────────────────────────────────────────────────────

    def _ee_geometry_to_region(self, ee_geom: ee.Geometry):
        """Convert an ee.Geometry to a client-side GeoJSON geometry dict."""
        info = ee_geom.getInfo()
        if not isinstance(info, dict) or "type" not in info:
            raise ValueError("Could not convert ee.Geometry to a GeoJSON region.")
        return info

    def _download_ee_image_direct(
        self,
        image: ee.Image,
        out_path: Path,
        region,
        scale: float,
        crs: str = "EPSG:4326",
        file_per_band: bool = False,
        timeout: int = 300,
    ) -> Path:
        """
        Download an Earth Engine image directly using getDownloadURL.

        If Earth Engine returns a ZIP payload, extract the first TIFF inside and
        write it to out_path. Otherwise write the response bytes directly.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        params = {
            "scale": scale,
            "crs": crs,
            "region": region,
            "format": "GEO_TIFF",
            "filePerBand": file_per_band,
        }

        url = image.getDownloadURL(params)
        log.info("EE direct download -> %s", out_path.name)

        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        content = resp.content

        if content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                tif_names = [
                    name for name in zf.namelist()
                    if name.lower().endswith((".tif", ".tiff"))
                ]
                if not tif_names:
                    raise RuntimeError(
                        f"ZIP download for {out_path.name} contained no TIFF files."
                    )

                tif_name = tif_names[0]
                with zf.open(tif_name) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            with open(out_path, "wb") as f:
                f.write(content)

        log.info("Saved EE raster: %s", out_path)
        return out_path

    def _split_bounds_into_tiles(
        self,
        bounds: Tuple[float, float, float, float],
        tile_width_deg: float,
        tile_height_deg: float,
    ) -> List[Tuple[float, float, float, float]]:
        """Split lon/lat bounds into rectangular tiles."""
        minx, miny, maxx, maxy = bounds
        tiles: List[Tuple[float, float, float, float]] = []

        x = minx
        while x < maxx:
            x2 = min(x + tile_width_deg, maxx)
            y = miny
            while y < maxy:
                y2 = min(y + tile_height_deg, maxy)
                tiles.append((x, y, x2, y2))
                y = y2
            x = x2

        return tiles

    def _download_ee_image_direct_tiled(
        self,
        image: ee.Image,
        out_path: Path,
        aoi_gdf: gpd.GeoDataFrame,
        scale: float,
        crs: str = "EPSG:4326",
        tile_width_deg: float = 0.05,
        tile_height_deg: float = 0.05,
    ) -> Path:
        """
        Download an Earth Engine image tile-by-tile over the AOI bbox,
        then mosaic locally into out_path.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        aoi_4326 = aoi_gdf.to_crs("EPSG:4326")
        aoi_union = aoi_4326.union_all()
        bounds = aoi_union.bounds
        tile_bounds = self._split_bounds_into_tiles(
            bounds,
            tile_width_deg=tile_width_deg,
            tile_height_deg=tile_height_deg,
        )

        temp_dir = Path(tempfile.mkdtemp(prefix="ee_tiles_", dir=str(out_path.parent)))
        tile_paths: List[Path] = []

        try:
            for i, (minx, miny, maxx, maxy) in enumerate(tile_bounds):
                tile_box = box(minx, miny, maxx, maxy)
                if aoi_union.intersection(tile_box).is_empty:
                    continue

                tile_geom = ee.Geometry.Rectangle(
                    [minx, miny, maxx, maxy],
                    proj="EPSG:4326",
                    geodesic=False,
                )
                tile_path = temp_dir / f"tile_{i:04d}.tif"

                self._download_ee_image_direct(
                    image=image.clip(tile_geom),
                    out_path=tile_path,
                    region=tile_geom.getInfo(),
                    scale=scale,
                    crs=crs,
                )
                tile_paths.append(tile_path)

            if not tile_paths:
                raise RuntimeError(f"No tiles were downloaded for {out_path.name}")

            readers = [rasterio.open(p) for p in tile_paths]
            try:
                mosaic, transform = rio_merge(readers)
                meta = readers[0].meta.copy()
                meta.update(
                    driver="GTiff",
                    height=mosaic.shape[1],
                    width=mosaic.shape[2],
                    transform=transform,
                    crs=crs,
                )
            finally:
                for r in readers:
                    r.close()

            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(mosaic)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        log.info("Saved tiled EE raster: %s", out_path)
        return out_path

    def _download_ee_with_fallback(
        self,
        image: ee.Image,
        out_path: Path,
        aoi_gdf: gpd.GeoDataFrame,
        region,
        scale: float,
        crs: str = "EPSG:4326",
        tile_width_deg: float = 0.05,
        tile_height_deg: float = 0.05,
    ) -> Path:
        """
        Try single-request EE direct download first.
        If the request is too large, retry with tiled downloads.
        """
        try:
            return self._download_ee_image_direct(
                image=image,
                out_path=out_path,
                region=region,
                scale=scale,
                crs=crs,
            )
        except ee.ee_exception.EEException as exc:
            msg = str(exc)
            if "Total request size" not in msg:
                raise

            log.warning(
                "EE request too large for single download (%s). Retrying tiled export for %s.",
                msg,
                out_path.name,
            )
            return self._download_ee_image_direct_tiled(
                image=image,
                out_path=out_path,
                aoi_gdf=aoi_gdf,
                scale=scale,
                crs=crs,
                tile_width_deg=tile_width_deg,
                tile_height_deg=tile_height_deg,
            )

    # ─────────────────────────────────────────────────────────────────
    # INTERNALS — Vector runners
    # ─────────────────────────────────────────────────────────────────

    def _run_gba(self, con, ds: dict, out_root: Path) -> List[str]:
        glob = str(self.config.gba.s3_url).rstrip("/")
        if "*" not in glob:
            glob += "/*.parquet"
        out_path = out_root / f"{ds['slug']}_gba.parquet"
        results = self._extract(con, ds["aoi"], out_path, parquet_glob=glob)
        self._tag_vector_output(out_path, ds)
        return results

    def _run_overture(self, con, ds: dict, out_root: Path, base: str) -> List[str]:
        theme = self.config.overture.theme
        types = self.config.overture.types or []
        outputs: List[str] = []
        for typ in types:
            out_path = out_root / f"{ds['slug']}_overture_{typ}.parquet"
            results = self._extract(
                con,
                ds["aoi"],
                out_path,
                base_path=base,
                theme=theme,
                typ=typ,
            )
            self._tag_vector_output(out_path, ds)
            outputs += results
        return outputs

    def _run_globfp(self, con, ds: dict, out_root: Path, world_grid: Path) -> List[str]:
        final_path = out_root / f"{ds['slug']}_globfp.parquet"
        if final_path.exists() and not self.overwrite:
            log.debug("Skip existing: %s", final_path)
            return [str(final_path)]

        aoi_union = ds["aoi"].union_all()
        grid_ids = get_grid_ids_for_geometry(world_grid, aoi_union)
        log.info("GloBFP | %s | %d grid tile(s) intersect AOI.", ds["id"], len(grid_ids))

        if len(grid_ids) == 0:
            log.info("GloBFP | %s | no intersecting grid tiles for AOI; skipping.", ds["id"])
            return []

        per_grid_paths: List[Path] = []
        for grid_id in grid_ids:
            tmp_path = out_root / f"{ds['slug']}_globfp_grid{grid_id}.parquet"
            try:
                shp = download_globfp_grid_tile(self.config, grid_id)
                self._extract(con, ds["aoi"], tmp_path, shp_path=shp)
                per_grid_paths.append(tmp_path)
                log.info("GloBFP OK | %s | grid=%d -> %s", ds["id"], grid_id, tmp_path.name)
            except Exception:
                log.exception("GloBFP failed | %s | grid=%d", ds["id"], grid_id)

        if not per_grid_paths:
            log.warning("GloBFP | %s | intersecting tiles found, but none downloaded successfully.", ds["id"])
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

    # ─────────────────────────────────────────────────────────────────
    # INTERNALS — Vector post-processing
    # ─────────────────────────────────────────────────────────────────

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

    # ─────────────────────────────────────────────────────────────────
    # INTERNALS — Spatial extract
    # ─────────────────────────────────────────────────────────────────

    def _extract(
        self,
        con,
        aoi_gdf,
        out_path: Path,
        *,
        shp_path=None,
        parquet_glob=None,
        base_path=None,
        theme="",
        typ="",
    ) -> List[str]:
        """
        Spatially query a source dataset clipped to the AOI and write a parquet file.
        """
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
            df = con.execute(sql).df()
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
            con.execute(sql)

        log.info("Wrote: %s", out_path)
        return [str(out_path)]

    # ─────────────────────────────────────────────────────────────────
    # INTERNALS — Raster helpers
    # ─────────────────────────────────────────────────────────────────

    def _mask_if_multi_aoi(self, raster_path: Path, ds: dict, nodata: float = 0.0) -> None:
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

    # ─────────────────────────────────────────────────────────────────
    # INTERNALS — Raster runners
    # ─────────────────────────────────────────────────────────────────

    def _run_obt(self, ds: dict, out_root: Path, cfg) -> List[str]:
        """Download yearly rasters from Google Open Buildings Temporal using direct EE export with tiled fallback."""
        aoi_ee = aoi_gdf_to_ee_geometry(ds["aoi"])
        region = self._ee_geometry_to_region(aoi_ee)
        collection = ee.ImageCollection(cfg.ee_collection_id)
        outputs: List[str] = []

        for year in cfg.years:
            out_path = out_root / f"{ds['slug']}_obt_{year}.tif"
            if out_path.exists() and not self.overwrite:
                log.info("[OBT] %s %d — exists, skipping.", ds["id"], year)
                outputs.append(str(out_path))
                continue

            start, end = f"{year}-01-01", f"{year + 1}-01-01"
            year_ic = collection.filterDate(start, end).filterBounds(aoi_ee)
            count = year_ic.size().getInfo()
            if count == 0:
                log.warning("[OBT] %s %d — no data for AOI.", ds["id"], year)
                continue

            image = year_ic.mosaic().select(cfg.bands).clip(aoi_ee)

            self._download_ee_with_fallback(
                image=image,
                out_path=out_path,
                aoi_gdf=ds["aoi"],
                region=region,
                scale=cfg.scale,
                crs="EPSG:4326",
            )
            log.info("[OBT] %s %d — saved %s", ds["id"], year, out_path)

            self._mask_if_multi_aoi(out_path, ds)
            outputs.append(str(out_path))

        return outputs

    def _run_tempo(self, ds: dict, out_root: Path, cfg) -> List[str]:
        """Download Microsoft TEMPO tiles and create a clipped mosaic."""
        target_crs = "EPSG:4326"
        tile_cache = Path(cfg.tile_cache_dir)
        reproj_cache = tile_cache / "reproj_4326"
        tile_cache.mkdir(parents=True, exist_ok=True)
        reproj_cache.mkdir(parents=True, exist_ok=True)

        aoi_union = ds["aoi"].union_all()

        tile_index_cache = Path(cfg.tile_index_cache)
        download_file(cfg.tile_index_url, tile_index_cache)
        tile_index = gpd.read_file(tile_index_cache).to_crs(target_crs)
        selected = tile_index[tile_index.intersects(aoi_union)].copy().reset_index(drop=True)
        log.info("[TEMPO] %s — %d tile(s) intersect AOI.", ds["id"], len(selected))

        if selected.empty:
            log.warning("[TEMPO] %s — no tiles intersect AOI.", ds["id"])
            return []

        selected.to_file(
            out_root / f"{ds['slug']}_tempo_tile_footprints.geojson",
            driver="GeoJSON",
        )

        url_col = get_tile_url_col(selected.columns)
        reproj_files: List[Path] = []
        for _, row in selected.iterrows():
            url = row[url_col]
            tile_name = os.path.basename(url)
            raw_path = tile_cache / tile_name
            reproj_path = reproj_cache / tile_name

            download_file(url, raw_path)
            if not reproj_path.exists():
                log.info("    Reprojecting %s -> EPSG:4326 …", tile_name)
                reproject_to_4326(raw_path, reproj_path)
            reproj_files.append(reproj_path)

        if not cfg.make_mosaic:
            return [str(f) for f in reproj_files]

        mosaic_path = out_root / f"{ds['slug']}_tempo_2023q4.tif"
        if mosaic_path.exists() and not self.overwrite:
            log.info("[TEMPO] %s — mosaic exists, skipping.", ds["id"])
            return [str(mosaic_path)]

        readers = [rasterio.open(fp) for fp in reproj_files]
        try:
            mosaic, transform = rio_merge(readers)
            meta = readers[0].meta.copy()
            meta.update(
                crs=target_crs,
                height=mosaic.shape[1],
                width=mosaic.shape[2],
                transform=transform,
                driver="GTiff",
            )
        finally:
            for reader in readers:
                reader.close()

        temp = out_root / "_temp_mosaic.tif"
        with rasterio.open(temp, "w", **meta) as writer:
            writer.write(mosaic)

        sub_aois = ds.get("sub_aois", [])
        if ds.get("is_multi_aoi") and sub_aois:
            clip_shapes = [mapping(s["geometry"]) for s in sub_aois]
        else:
            clip_shapes = [_shapely_to_geojson_dict(aoi_union)]

        try:
            with rasterio.open(temp) as reader:
                clipped, tf = rio_mask(reader, clip_shapes, crop=True, all_touched=True)
                clipped_meta = reader.meta.copy()
                clipped_meta.update(
                    crs=target_crs,
                    height=clipped.shape[1],
                    width=clipped.shape[2],
                    transform=tf,
                )
            with rasterio.open(mosaic_path, "w", **clipped_meta) as writer:
                writer.write(clipped)
        except Exception as exc:
            log.warning("[TEMPO] %s — clip failed (%s); saving unclipped mosaic.", ds["id"], exc)
            shutil.copy2(temp, mosaic_path)

        temp.unlink(missing_ok=True)
        log.info("[TEMPO] %s — saved %s", ds["id"], mosaic_path)
        return [str(mosaic_path)]

    def _run_ghsl(self, ds: dict, out_root: Path, cfg) -> List[str]:
        """Download GHSL products for the AOI using direct EE export with tiled fallback."""
        aoi_ee = aoi_gdf_to_ee_geometry(ds["aoi"])
        region = self._ee_geometry_to_region(aoi_ee)
        outputs: List[str] = []

        for product_name, prod in cfg.products.items():
            for year in prod.years:
                out_path = out_root / f"{ds['slug']}_ghsl_{product_name.lower()}_{year}.tif"
                if out_path.exists() and not self.overwrite:
                    log.info("[GHSL/%s] %s %d — exists, skipping.", product_name, ds["id"], year)
                    outputs.append(str(out_path))
                    continue

                image_id = f"{prod.ee_id}/{year}"
                try:
                    image = ee.Image(image_id).select(prod.band).clip(aoi_ee)
                    image.bandNames().getInfo()
                except Exception as exc:
                    log.error(
                        "[GHSL/%s] %s %d — cannot load '%s': %s",
                        product_name,
                        ds["id"],
                        year,
                        image_id,
                        exc,
                    )
                    continue

                try:
                    self._download_ee_with_fallback(
                        image=image,
                        out_path=out_path,
                        aoi_gdf=ds["aoi"],
                        region=region,
                        scale=prod.scale,
                        crs="EPSG:4326",
                    )
                    log.info(
                        "[GHSL/%s] %s %d — saved %s",
                        product_name,
                        ds["id"],
                        year,
                        out_path,
                    )

                    self._mask_if_multi_aoi(out_path, ds)
                    outputs.append(str(out_path))
                except Exception as exc:
                    log.error(
                        "[GHSL/%s] %s %d — download failed: %s",
                        product_name,
                        ds["id"],
                        year,
                        exc,
                        exc_info=True,
                    )

        return outputs
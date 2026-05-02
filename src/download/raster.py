"""
Raster runners: OBT, TEMPO, GHSL, WSF Tracker.

Each runner downloads / mosaics one source. All inherit from
BaseRasterRunner, which supplies the multi-AOI mask post-processing.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List

import ee
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from shapely.geometry import mapping

from src.download.base import BaseRasterRunner
from src.utils.ee_utils import (
    download_ee_with_fallback,
    ee_geometry_to_region,
)
from src.utils.wsf_utils import (
    index_wsf_tracker_tiles,
    mosaic_and_clip_wsf_tracker,
)
from src.utils import (
    _shapely_to_geojson_dict,
    aoi_gdf_to_ee_geometry,
    download_file,
    get_tile_url_col,
    reproject_to_4326,
)

log = logging.getLogger("UrbanDownloader.raster")


class OBTRunner(BaseRasterRunner):
    """Yearly Google Open Buildings Temporal rasters via EE direct download."""

    name = "google_open_buildings_temporal"

    def run(self, ds: dict, out_root: Path) -> List[str]:
        cfg = self.source_cfg
        aoi_ee = aoi_gdf_to_ee_geometry(ds["aoi"])
        region = ee_geometry_to_region(aoi_ee)
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

            download_ee_with_fallback(
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


class TEMPORunner(BaseRasterRunner):
    """Microsoft TEMPO tiles -> reprojected -> mosaic -> clipped to AOI/sub-AOIs."""

    name = "microsoft_tempo"

    def run(self, ds: dict, out_root: Path) -> List[str]:
        cfg = self.source_cfg
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


class GHSLRunner(BaseRasterRunner):
    """GHSL products (per product × per year) via EE direct download."""

    name = "ghsl"

    def run(self, ds: dict, out_root: Path) -> List[str]:
        cfg = self.source_cfg
        aoi_ee = aoi_gdf_to_ee_geometry(ds["aoi"])
        region = ee_geometry_to_region(aoi_ee)
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
                    download_ee_with_fallback(
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


class WSFTrackerRunner(BaseRasterRunner):
    """
    Select WSF Tracker tiles from a Drive mount that intersect the AOI,
    optionally copy them and/or mosaic+clip them into one product.
    """

    name = "wsf_tracker"

    def run(self, ds: dict, out_root: Path) -> List[str]:
        cfg = self.source_cfg
        drive_root = Path(cfg.drive_root)
        if not drive_root.exists():
            raise FileNotFoundError(f"WSF Tracker drive_root does not exist: {drive_root}")

        aoi_4326 = ds["aoi"].to_crs("EPSG:4326")
        aoi_union = aoi_4326.union_all()

        tile_index = index_wsf_tracker_tiles(
            drive_root=drive_root,
            tile_degree_size=cfg.tile_degree_size,
        )

        if tile_index.empty:
            log.warning("[WSF Tracker] %s — no TIFFs found under %s", ds["id"], drive_root)
            return []

        selected = tile_index[tile_index.intersects(aoi_union)].copy().reset_index(drop=True)
        log.info("[WSF Tracker] %s — %d tile(s) intersect AOI.", ds["id"], len(selected))

        if selected.empty:
            return []

        outputs: List[str] = []
        out_root.mkdir(parents=True, exist_ok=True)

        selected.to_file(
            out_root / f"{ds['slug']}_{cfg.output_prefix}_tile_footprints.geojson",
            driver="GeoJSON",
        )

        if cfg.keep_tile_copies:
            tile_dir = out_root / f"{cfg.output_prefix}_tiles"
            tile_dir.mkdir(parents=True, exist_ok=True)

            for _, row in selected.iterrows():
                src = Path(row["path"])
                dst = tile_dir / src.name
                if not dst.exists() or self.overwrite:
                    shutil.copy2(src, dst)
                outputs.append(str(dst))

        if cfg.make_mosaic:
            mosaic_path = out_root / f"{ds['slug']}_{cfg.output_prefix}.tif"
            if mosaic_path.exists() and not self.overwrite:
                log.info("[WSF Tracker] %s — mosaic exists, skipping.", ds["id"])
                outputs.append(str(mosaic_path))
            else:
                mosaic_and_clip_wsf_tracker(
                    selected=selected,
                    ds=ds,
                    out_path=mosaic_path,
                    nodata=cfg.nodata,
                )
                outputs.append(str(mosaic_path))

        return outputs
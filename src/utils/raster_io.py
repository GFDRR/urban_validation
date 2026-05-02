"""
Generic raster I/O and download helpers.

Streaming HTTP download with idempotent skip, tile-index URL column
detection, full-band reprojection to EPSG:4326, and sub-AOI raster
masking (used by the multi-AOI cities to zero out pixels in inter-AOI
gaps).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List

import rasterio
import requests
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping

log = logging.getLogger(__name__)


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream a file to out_path, skipping if it already exists."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    log.info("Downloading %s …", url)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
    log.info("Saved: %s", out_path)


def get_tile_url_col(columns) -> str:
    """Pick the column in a tile index that holds the per-tile URL."""
    if "data" in columns:
        return "data"
    candidates = [
        c for c in columns
        if any(k in c.lower() for k in ("url", "href", "path", "link"))
    ]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"Cannot find a URL column in tile index. Columns present: {list(columns)}"
    )


def reproject_to_4326(src_path: Path, dst_path: Path) -> None:
    """Reproject all bands of a raster to EPSG:4326 (bilinear)."""
    with rasterio.open(src_path) as reader:
        if reader.crs and reader.crs.to_epsg() == 4326:
            shutil.copy2(src_path, dst_path)
            return
        transform, width, height = calculate_default_transform(
            reader.crs, "EPSG:4326", reader.width, reader.height, *reader.bounds
        )
        meta = reader.meta.copy()
        meta.update(
            crs="EPSG:4326",
            transform=transform,
            width=width,
            height=height,
            driver="GTiff",
        )
        with rasterio.open(dst_path, "w", **meta) as writer:
            for i in range(1, reader.count + 1):
                reproject(
                    source=rasterio.band(reader, i),
                    destination=rasterio.band(writer, i),
                    src_transform=reader.transform,
                    src_crs=reader.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear,
                )


def get_sub_aoi_geojson_geometries(sub_aois: List[Dict]) -> List[dict]:
    """Convert sub-AOI geometries to GeoJSON dicts for rasterio masking."""
    return [mapping(s["geometry"]) for s in sub_aois]


def mask_raster_to_sub_aois(
    raster_path: Path,
    sub_aois: List[Dict],
    *,
    nodata: float = 0.0,
    all_touched: bool = True,
    in_place: bool = True,
) -> Path:
    """Mask a raster so that only pixels within sub-AOI polygons are kept.

    Pixels outside any sub-AOI polygon are set to `nodata`.
    If in_place is True, the file is overwritten; otherwise a new file
    with suffix '_masked' is written alongside the original.

    Parameters
    ----------
    raster_path : Path
        Path to the raster file (GeoTIFF).
    sub_aois : list[dict]
        The sub_aois list from the dataset dict.
    nodata : float
        The nodata value to assign to masked-out pixels.
    all_touched : bool
        If True, all pixels touched by sub-AOI boundaries are kept.
    in_place : bool
        If True, overwrite the original file. Otherwise write *_masked.tif.

    Returns
    -------
    Path
        Path to the (possibly new) masked raster.
    """
    geojson_geoms = get_sub_aoi_geojson_geometries(sub_aois)

    with rasterio.open(raster_path) as src:
        masked_data, masked_transform = rio_mask(
            src,
            geojson_geoms,
            crop=False,          # keep original extent
            all_touched=all_touched,
            nodata=nodata,
            filled=True,         # fill outside with nodata
        )
        meta = src.meta.copy()
        meta.update(nodata=nodata)

    if in_place:
        out_path = raster_path
    else:
        out_path = raster_path.with_name(
            raster_path.stem + "_masked" + raster_path.suffix
        )

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(masked_data)

    log.info(
        "Masked raster -> %s  (nodata=%s, %d sub-AOI polygons)",
        out_path, nodata, len(sub_aois),
    )
    return out_path

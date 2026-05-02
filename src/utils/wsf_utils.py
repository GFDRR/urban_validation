"""
WSF Tracker tile utilities.

WSF Tracker rasters are pre-downloaded onto a Drive mount; these helpers
parse tile filenames into footprint geometries, build a spatial index of
local tiles, and mosaic+clip selected tiles to an AOI (or its sub-AOIs
for multi-AOI cities).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge as rio_merge
from shapely.geometry import box, mapping

log = logging.getLogger("UrbanDownloader.wsf")

WSF_TRACKER_REGEX = re.compile(
    r"WSFtracker_(\d{8})[-_](\d{8})_(-?\d+)_(-?\d+)\.tif$"
)


def parse_wsf_tracker_filename(
    tif_path: Path,
    tile_degree_size: int = 2,
    pad_deg: float = 0.011,
) -> Optional[dict]:
    """
    Parse:
        WSFtracker_<start>-<end>_<lon>_<lat>.tif

    Verified from sample rasters:
      - lon/lat correspond to the nominal lower-left corner
      - tiles are nominally 2° x 2°
      - actual bounds are padded by about 0.01° on all sides
    """
    m = WSF_TRACKER_REGEX.match(tif_path.name)
    if not m:
        return None

    start, end, lon, lat = m.groups()
    lon = int(lon)
    lat = int(lat)

    geom = box(
        lon - pad_deg,
        lat - pad_deg,
        lon + tile_degree_size + pad_deg,
        lat + tile_degree_size + pad_deg,
    )

    return {
        "start": start,
        "end": end,
        "lon": lon,
        "lat": lat,
        "geometry": geom,
    }


def index_wsf_tracker_tiles(
    drive_root: Path,
    tile_degree_size: int = 2,
) -> gpd.GeoDataFrame:
    """
    Fast index of WSF Tracker tiles using filename parsing only.
    Assumes filenames follow:
      WSFtracker_<start>_<end>_<lon>_<lat>.tif
    """
    records = []

    for tif in drive_root.rglob("*.tif"):
        parsed = parse_wsf_tracker_filename(tif, tile_degree_size=tile_degree_size)
        if parsed is None:
            continue

        records.append(
            {
                "path": str(tif),
                "name": tif.name,
                "start": parsed["start"],
                "end": parsed["end"],
                "lon": parsed["lon"],
                "lat": parsed["lat"],
                "geometry": parsed["geometry"],
            }
        )

    if not records:
        return gpd.GeoDataFrame(
            columns=["path", "name", "start", "end", "lon", "lat", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def mosaic_and_clip_wsf_tracker(
    selected: gpd.GeoDataFrame,
    ds: dict,
    out_path: Path,
    nodata: int = 0,
) -> Path:
    """
    Mosaic selected WSF Tracker tiles and clip to AOI or sub-AOIs.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    readers = [rasterio.open(p) for p in selected["path"].tolist()]
    try:
        mosaic, transform = rio_merge(readers, nodata=nodata)
        meta = readers[0].meta.copy()
        meta.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            nodata=nodata,
        )
    finally:
        for r in readers:
            r.close()

    temp_path = out_path.parent / f"_{out_path.stem}_temp.tif"
    with rasterio.open(temp_path, "w", **meta) as dst:
        dst.write(mosaic)

    aoi_union = ds["aoi"].to_crs("EPSG:4326").union_all()
    sub_aois = ds.get("sub_aois", [])

    if ds.get("is_multi_aoi") and sub_aois:
        clip_shapes = [mapping(s["geometry"]) for s in sub_aois]
    else:
        clip_shapes = [mapping(aoi_union)]

    try:
        with rasterio.open(temp_path) as src:
            clipped, tf = rio_mask(
                src,
                clip_shapes,
                crop=True,
                nodata=nodata,
                all_touched=True,
            )
            clipped_meta = src.meta.copy()
            clipped_meta.update(
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=tf,
                nodata=nodata,
            )

        with rasterio.open(out_path, "w", **clipped_meta) as dst:
            dst.write(clipped)
    finally:
        temp_path.unlink(missing_ok=True)

    log.info("[WSF Tracker] %s — saved %s", ds["id"], out_path)
    return out_path

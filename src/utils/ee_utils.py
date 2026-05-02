"""
Earth Engine direct-download helpers.

Used by both OBT and GHSL runners to pull imagery via getDownloadURL
with a tiled fallback for AOIs that exceed EE's single-request size cap.
"""
from __future__ import annotations

import io
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple

import ee
import geopandas as gpd
import rasterio
import requests
from rasterio.merge import merge as rio_merge
from shapely.geometry import box

log = logging.getLogger("UrbanDownloader.ee")


def ee_geometry_to_region(ee_geom: ee.Geometry) -> dict:
    """Convert an ee.Geometry to a client-side GeoJSON geometry dict."""
    info = ee_geom.getInfo()
    if not isinstance(info, dict) or "type" not in info:
        raise ValueError("Could not convert ee.Geometry to a GeoJSON region.")
    return info


def split_bounds_into_tiles(
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


def download_ee_image_direct(
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


def download_ee_image_direct_tiled(
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
    tile_bounds = split_bounds_into_tiles(
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

            download_ee_image_direct(
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


def download_ee_with_fallback(
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
        return download_ee_image_direct(
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
        return download_ee_image_direct_tiled(
            image=image,
            out_path=out_path,
            aoi_gdf=aoi_gdf,
            scale=scale,
            crs=crs,
            tile_width_deg=tile_width_deg,
            tile_height_deg=tile_height_deg,
        )

# ---------------------------------------------------------------------
# Earth Engine initialization
# ---------------------------------------------------------------------

def init_earth_engine(project: str | None = None) -> None:
    """
    Initialize Earth Engine.

    Behavior:
      - First try existing credentials.
      - If unavailable and running in an interactive notebook kernel, prompt auth.
      - If unavailable and running as a plain script, raise a helpful error
        instead of crashing inside Colab widget auth.
    """
    import ee

    def _in_ipython_kernel() -> bool:
        try:
            from IPython import get_ipython
            ip = get_ipython()
            return ip is not None and getattr(ip, "kernel", None) is not None
        except Exception:
            return False

    kwargs = {}
    if project:
        kwargs["project"] = project

    try:
        ee.Initialize(**kwargs)
        print("EE already initialized.")
        return

    except Exception as e:
        print("EE not initialised – attempting authentication …")

        # Only do interactive auth when a real notebook kernel exists
        if _in_ipython_kernel():
            ee.Authenticate()
            ee.Initialize(**kwargs)
            print("EE authenticated and initialized.")
            return

        raise RuntimeError(
            "Earth Engine credentials are not available for this script run.\n\n"
            "Authenticate first in an interactive environment, then rerun.\n"
            "Options:\n"
            "  1. In a notebook cell, run:\n"
            "       import ee\n"
            "       ee.Authenticate()\n"
            "       ee.Initialize(project={!r})\n"
            "  2. Or in a shell, run:\n"
            "       earthengine authenticate\n\n"
            "After that, rerun:\n"
            "  python main.py --data-config configs/data_configs.yaml "
            "--val-config configs/validation_configs.yaml --skip-vector"
            .format(project)
        ) from e
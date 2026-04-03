""" Utility functions for loading and exploring datasets within the Global Satellite Derived Urban Dataset Validation project.
"""
from __future__ import annotations
import gc
import json
import os
import re
import requests
from typing import Optional, Union, Dict, List, Tuple
from pathlib import Path
import zipfile
from dataclasses import dataclass

import logging
import shutil

import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
import numpy as np
from src.preprocess import validate_aoi_geometry
import pyproj

FIGSHARE_API = "https://api.figshare.com/v2"
GLOBFP_PARTS: List[Tuple[int, int, int]] = [
    (0,    400,  28879733),
    (401,  699,  28881749),
    (700,  899,  28882700),
    (900,  1299, 28889813),
    (1300, 1699, 28890593),
    (1700, 1799, 28891631),
    (1800, 1899, 28903454),
    (1900, 1999, 28903853),
    (2000, 2299, 28904453),
    (2300, 2599, 28906499),
]

# rows per chunk when computing UTM areas
_AREA_CHUNK_SIZE = int(os.environ.get("AREA_CHUNK_SIZE", 50_000))

@dataclass(frozen=True)
class FigshareFile:
    id: int
    name: str
    download_url: str
    size: int

def get_article_id(grid_id: int) -> int:
    for lo, hi, aid in GLOBFP_PARTS:
        if lo <= grid_id <= hi:
            return aid
    raise ValueError(f"grid_id {grid_id} not covered by any GloBFP PART range")

def _unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def globfp_local_dir(config) -> Path:
    """Root cache directory for all GloBFP downloads."""
    p = Path(config.globfp.local_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def download_globfp_file(url: str, dest: Path, timeout: int = 180) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=16 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)
    return dest

def get_grid_ids_for_geometry(world_grid_shp: Path, aoi_geom_4326) -> List[int]:
        """Return sorted list of grid IDs that intersect the given geometry."""
        grid = gpd.read_file(world_grid_shp).to_crs("EPSG:4326")

        cand_fields = ["gridID", "grid_id", "grid_ID", "GRIDID", "GRID_ID", "id", "ID"]
        grid_field  = next((c for c in cand_fields if c in grid.columns), None)
        if grid_field is None:
            raise ValueError(
                f"Cannot find grid-ID field in world_grid. Columns: {list(grid.columns)}"
            )

        hits = grid[grid.intersects(aoi_geom_4326)]
        return sorted({int(v) for v in hits[grid_field]})

def get_figshare_list_files(article_id: int) -> List[FigshareFile]:
    url = f"{FIGSHARE_API}/articles/{article_id}"
    r   = requests.get(url, timeout=60)
    r.raise_for_status()
    return [
            FigshareFile(
                id=int(f["id"]),
                name=str(f["name"]),
                download_url=str(f["download_url"]),
                size=int(f.get("size", 0)),
            )
            for f in r.json().get("files", [])
        ]

def select_globfp_tile_files(
        files: List[FigshareFile], grid_id: int
    ) -> List[FigshareFile]:
    gid = str(grid_id)
    zips = [f for f in files
            if f.name.lower().endswith(".zip")
            and re.match(rf"^{gid}[_\-]", f.name)]
    if zips:
        return zips

    components = [
        f for f in files
        if re.match(rf"^{gid}[_\-].*\.(shp|dbf|shx|prj|cpg)$",
                    f.name, flags=re.IGNORECASE)
    ]
    if components:
        return components

    return [f for f in files if f"{gid}_" in f.name]


def download_globfp_grid_tile(config, grid_id: int) -> Path:
    tiles_dir = globfp_local_dir(config) / "tiles" / f"grid_id={grid_id}"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    existing = list(tiles_dir.glob("*.shp"))
    if existing:
        return existing[0]

    article_id = get_article_id(grid_id)
    all_files  = get_figshare_list_files(article_id)
    selected   = select_globfp_tile_files(all_files, grid_id)

    if not selected:
        raise FileNotFoundError(
            f"No Figshare files matched grid_id={grid_id} in article {article_id}. "
            f"Available: {[f.name for f in all_files[:10]]}"
        )

    for f in selected:
        dest = tiles_dir / f.name
        print(f"Downloading tile file: {f.name} ({f.size / 1e6:.1f} MB)")
        download_globfp_file(f.download_url, dest, timeout=600)
        if dest.suffix.lower() == ".zip":
           _unzip(dest, tiles_dir)

    shp_candidates = list(tiles_dir.rglob("*.shp"))
    if not shp_candidates:
        raise FileNotFoundError(
            f"No .shp found after downloading tile grid_id={grid_id} in {tiles_dir}"
        )

    gid = str(grid_id)
    shp_candidates.sort(
        key=lambda p: (0 if p.name.startswith(gid + "_") else 1, len(p.name))
    )
    return shp_candidates[0]


def ensure_world_grid(config) -> Path:
    local_dir = globfp_local_dir(config)
    record_id = config.globfp.zenodo_record
    zip_path  = local_dir / "world_grid.zip"
    grid_dir  = local_dir / "world_grid"
    shp_path  = grid_dir  / "world_grid.shp"

    if shp_path.exists():
        return shp_path

    url = f"https://zenodo.org/records/{record_id}/files/world_grid.zip?download=1"
    print(f"Downloading world_grid.zip from Zenodo record {record_id}")
    download_globfp_file(url, zip_path, timeout=180)
    _unzip(zip_path, grid_dir)

    if not shp_path.exists():
        candidates = list(grid_dir.rglob("world_grid.shp"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"world_grid.shp not found after unzip in {grid_dir}")
    return shp_path


def _read_gdf(path: Union[str, Path]) -> gpd.GeoDataFrame:
    path = Path(path)
    if path.suffix.lower() in [".parquet", ".geoparquet"]:
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def load_buildings(
    path: Union[str, Path],
    *,
    crs_work: str,
    min_area_m2: float,
    fix_invalid_geoms: bool = False,
    compute_area_mode: str = "auto",
    logger=None,
) -> gpd.GeoDataFrame:
    """
    Load building footprints, reproject, compute area, filter by min area.

    Changes vs. original
    --------------------
    1.  Null / empty geometries are dropped BEFORE any CRS operation,
        preventing ``estimate_utm_crs`` from crashing on NaN bounds
        (the GloBFP / globfp parquet bug).

    2.  ``compute_area_mode`` default changed from ``"utm"`` to ``"auto"``:
        - If ``crs_work`` is already projected (metres), areas are computed
          directly — no chunked reproject needed, much faster.
        - If ``crs_work`` is geographic (degrees), falls back to the
          chunked-UTM approach.

    3.  The chunked-UTM path now also drops null geometries before calling
        ``estimate_utm_crs``, as a safety net.
    """
    path = Path(path)
    gdf = (gpd.read_parquet(path) if path.suffix.lower() in {".parquet", ".geoparquet"}
           else gpd.read_file(path))

    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS defined.")

    # ── FIX 1: drop null / empty geometries BEFORE any reprojection ──────────
    n_before = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    n_dropped = n_before - len(gdf)
    if n_dropped > 0:
        msg = f"[{path.name}] Dropped {n_dropped} null/empty geometries ({n_before} → {len(gdf)})"
        if logger:
            logger.warning(msg)
        else:
            print(msg)

    if gdf.empty:
        gdf = gdf.to_crs(crs_work)
        gdf["area_m2"] = np.float64()
        return gdf

    gdf = gdf.to_crs(crs_work)

    if fix_invalid_geoms:
        gdf = validate_aoi_geometry(gdf, label=path.name)

    # ── FIX 2: decide area computation strategy based on crs_work ────────────
    if compute_area_mode == "auto":
        # If crs_work is projected (units = metres), compute area directly
        crs_obj = pyproj.CRS(crs_work)
        if crs_obj.is_projected:
            compute_area_mode = "work_crs"
        else:
            compute_area_mode = "utm"

    if compute_area_mode == "work_crs":
        # crs_work is already in metres — just use .area directly
        gdf["area_m2"] = gdf.geometry.area

    elif compute_area_mode == "utm":
        # crs_work is geographic — need to reproject to UTM in chunks
        # Safety: drop any remaining null geometries before estimate_utm_crs
        valid_mask = gdf.geometry.notna() & ~gdf.geometry.is_empty
        if not valid_mask.all():
            gdf = gdf[valid_mask].copy()

        metric_crs = gdf.estimate_utm_crs()
        n = len(gdf)
        areas = np.empty(n, dtype=np.float64)

        chunk = _AREA_CHUNK_SIZE
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            chunk_gdf = gdf.iloc[start:end].to_crs(metric_crs)
            areas[start:end] = chunk_gdf.geometry.area.values
            del chunk_gdf

        del metric_crs
        gc.collect()

        gdf["area_m2"] = areas
        del areas
    else:
        raise ValueError(f"Unknown compute_area_mode={compute_area_mode!r}")

    gdf = gdf[gdf["area_m2"] >= float(min_area_m2)].copy()
    gdf.reset_index(drop=True, inplace=True)

    if logger:
        logger.info("Loaded buildings | n=%d | path=%s", len(gdf), path)

    return gdf

def load_aoi(
    path: Union[str, Path],
    *,
    crs_out: str = "EPSG:4326",
    buffer_meters: float = 0.0,
    dissolve: bool = False,
    logger=None,
) -> gpd.GeoDataFrame:
    path = Path(path)
    aoi = _read_gdf(path)

    if aoi.crs is None:
        if logger:
            logger.warning("AOI CRS missing; assuming %s", crs_out)
        aoi = aoi.set_crs("EPSG:4326")

    if dissolve and len(aoi) > 1:
        aoi = aoi.dissolve().reset_index(drop=True)

    buffer_meters = float(buffer_meters or 0.0)
    if buffer_meters > 0:
        if not aoi.crs.is_projected:
            if logger:
                logger.warning("AOI CRS is geographic; reprojecting to EPSG:3857 for buffering")
            aoi_metric = aoi.to_crs("EPSG:3857") # TODO: this should not be harcoded — ideally we would use a local UTM zone
            aoi_metric["geometry"] = aoi_metric.geometry.buffer(buffer_meters)
            aoi = aoi_metric.to_crs(aoi.crs)
        else:
            aoi["geometry"] = aoi.geometry.buffer(buffer_meters)

    if str(aoi.crs) != str(crs_out):
        if logger:
            logger.info("Reprojecting AOI | %s -> %s", aoi.crs, crs_out)
        aoi = aoi.to_crs(crs_out)

    if logger:
        logger.info("Loaded AOI | rows=%d | crs=%s | path=%s", len(aoi), aoi.crs, path)

    return aoi

def make_tiles(
    aoi: gpd.GeoDataFrame,
    tile_size_m: float,
    *,
    clip_to_aoi: bool = False,
    snap_origin: bool = False,
) -> gpd.GeoDataFrame:
    if aoi.empty:
        return gpd.GeoDataFrame({"tile_id": [], "geometry": []}, crs=aoi.crs)

    aoi_union = aoi.geometry.union_all()
    minx, miny, maxx, maxy = aoi_union.bounds
    tile = float(tile_size_m)

    if snap_origin:
        minx = np.floor(minx / tile) * tile
        miny = np.floor(miny / tile) * tile
        maxx = np.ceil(maxx / tile) * tile
        maxy = np.ceil(maxy / tile) * tile

    nx = int(np.ceil((maxx - minx) / tile))
    ny = int(np.ceil((maxy - miny) / tile))

    tiles = []
    for ix in range(nx):
        x0 = minx + ix * tile
        x1 = x0 + tile
        for iy in range(ny):
            y0 = miny + iy * tile
            y1 = y0 + tile
            poly = box(x0, y0, x1, y1)

            if not poly.intersects(aoi_union):
                continue

            tiles.append(poly.intersection(aoi_union) if clip_to_aoi else poly)

    tiles_gdf = gpd.GeoDataFrame({"geometry": tiles}, crs=aoi.crs)
    tiles_gdf.reset_index(drop=True, inplace=True)
    tiles_gdf["tile_id"] = tiles_gdf.index.astype(int)
    return tiles_gdf[["tile_id", "geometry"]]

def subset_by_tile(buildings: gpd.GeoDataFrame,
                   sindex,
                   tile_geom):
    possible_idx = list(sindex.intersection(tile_geom.bounds))
    if not possible_idx:
        return buildings.iloc[[]].copy()

    subset = buildings.iloc[possible_idx]
    subset = subset[subset.intersects(tile_geom)].copy()
    return subset

def resolve_out_root(config, dataset_id: str, subdir: str = "vector") -> Path:
    """Resolve the output directory for a dataset.

    If output.use_base_dir_for_output is true (default):
        <aoi.base_dir>/<dataset_id>/<subdir>/
    Otherwise:
        <output.root_dir>/<dataset_id>/<subdir>/
    """
    use_base = config.output.use_base_dir_for_output
    if use_base:
        p = Path(config.aoi.base_dir) / dataset_id / subdir
    else:
        root = config.output.root_dir or "data/outputs"
        p = Path(root) / dataset_id / subdir
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_all_aois(config) -> List[Dict]:
    """Parse the AOI inventory and return a list of dataset dicts.

    Returns list of {"id": str, "slug": str, "aoi": GeoDataFrame}.
    Output directories are NOT included — callers resolve those per download type.
    """
    aoi_path = Path(config.aoi.path)
    if aoi_path.suffix.lower() == ".csv":
        return _load_aois_from_csv(config)
    else:
        return _load_aoi_from_file(config, aoi_path)


def _load_aois_from_csv(config) -> List[Dict]:
    aoi_cfg  = config.aoi
    csv_path = Path(aoi_cfg.path)
    base_dir = Path(aoi_cfg.base_dir)
    id_col   = aoi_cfg.id_col
    crs_out  = aoi_cfg.crs_out

    df = pd.read_csv(csv_path, dtype=str)

    # Drop rows with no AOI file on disk
    if "has_aoi_file" in df.columns:
        df = df[df["has_aoi_file"].str.strip().str.upper() == "TRUE"]

    # Suitable filter
    if aoi_cfg.filter_suitable and "Suitable" in df.columns:
        before = len(df)
        df = df[df["Suitable"].str.strip().str.lower() == "yes"]
        print(f"Filtered inventory: {before} -> {len(df)} suitable rows")

    # Optional high-quality filter
    if aoi_cfg.high_quality_only and "is_high_quality" in df.columns:
        before = len(df)
        df = df[df["is_high_quality"].str.strip().str.upper() == "TRUE"]
        print(f"Filtered inventory: {before} -> {len(df)} high-quality rows")

    df = df.dropna(subset=[id_col, "aoi_file_name"])
    df = df[df["aoi_file_name"].str.strip() != ""]

    datasets: List[Dict] = []
    for dataset_id, group in df.groupby(id_col, sort=False):
        dataset_id = str(dataset_id)

        # Collect AOI paths — supports pipe-separated values and multiple rows
        aoi_paths: List[Path] = []
        for _, row in group.iterrows():
            for part in str(row["aoi_file_name"]).split("|"):
                part = part.strip()
                if part:
                    aoi_paths.append(base_dir / dataset_id / aoi_cfg.aoi_subdir / part)

        existing = [p for p in aoi_paths if p.exists()]
        if not existing:
            print(f"Dataset {dataset_id}: no AOI files found on disk, skipping.")
            continue

        aoi = _load_and_dissolve_aois(existing, dataset_id,
                                       crs_out=crs_out,
                                       buffer_meters=aoi_cfg.buffer_meters)
        if aoi is None or aoi.empty:
            print(f"Dataset {dataset_id}: empty AOI after loading, skipping.")
            continue

        slug = dataset_id.replace("-", "_").replace(" ", "_")
        datasets.append({"id": dataset_id, "slug": slug, "aoi": aoi})

    return datasets


def _load_aoi_from_file(config, aoi_path: Path) -> List[Dict]:
    aoi_cfg    = config.aoi
    aoi        = load_aoi(aoi_path, crs_out=aoi_cfg.crs_out,
                          buffer_meters=aoi_cfg.buffer_meters, dissolve=True)
    dataset_id = aoi_path.parent.parent.name
    slug       = dataset_id.replace("-", "_").replace(" ", "_")
    return [{"id": dataset_id, "slug": slug, "aoi": aoi}]


def _load_and_dissolve_aois(
    paths: List[Path],
    dataset_id: str,
    crs_out: str = "EPSG:4326",
    buffer_meters: float = 0.0,
) -> Optional[gpd.GeoDataFrame]:
    parts = []
    for p in paths:
        try:
            parts.append(load_aoi(p, crs_out=crs_out, buffer_meters=buffer_meters))
        except Exception as exc:
            print(f"Dataset {dataset_id}: failed to load AOI {p}: {exc}")

    if not parts:
        return None

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    return combined.dissolve().reset_index(drop=True) if len(combined) > 1 else combined

def get_projected_crs(gdf: gpd.GeoDataFrame) -> str:
    """
    Return the best UTM EPSG code for the centroid of a GeoDataFrame.

    Parameters
    ----------
    gdf : GeoDataFrame
        Must be in EPSG:4326 (or any geographic CRS).

    Returns
    -------
    str
        EPSG string like "EPSG:32637" (UTM zone 37N).
    """
    # make it is in lon/lat
    if gdf.crs and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)

    # get centroid of the union of all geometries
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    lon = (bounds[0] + bounds[2]) / 2
    lat = (bounds[1] + bounds[3]) / 2

    # use pyproj to find the best UTM zone
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon,
            south_lat_degree=lat,
            east_lon_degree=lon,
            north_lat_degree=lat,
        ),
    )

    if utm_crs_list:
        utm = utm_crs_list[0]
        return f"EPSG:{utm.code}"

    # fallback: calculate manually
    zone_number = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:{32600 + zone_number}"
    else:
        return f"EPSG:{32700 + zone_number}"

def init_earth_engine(project: str = "") -> None:
    import ee  # optional dependency: pip install earthengine-api
    kwargs = {"project": project} if project else {}
    try:
        ee.Initialize(**kwargs)
        print("Earth Engine initialised.")
    except Exception:
        print("EE not initialised – authenticating …")
        ee.Authenticate()
        ee.Initialize(**kwargs)

def _shapely_to_geojson_dict(geom) -> dict:
    return json.loads(
        gpd.GeoSeries([geom], crs="EPSG:4326").to_json()
    )["features"][0]["geometry"]

def aoi_gdf_to_ee_geometry(gdf):
    import ee  # optional dependency: pip install earthengine-api
    return ee.Geometry(_shapely_to_geojson_dict(gdf.union_all()))


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Stream-download a file, skipping if it already exists."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    logging.getLogger(__name__).info("Downloading %s …", url)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
    logging.getLogger(__name__).info("Saved: %s", out_path)


def get_tile_url_col(columns) -> str:
    """Detect the URL/COG column in a tile index (e.g. TEMPO tile index)."""
    if "data" in columns:
        return "data"
    candidates = [c for c in columns if any(k in c.lower() for k in ("url", "href", "path", "link"))]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"Cannot find a URL column in tile index. Columns present: {list(columns)}"
    )


def reproject_to_4326(src_path: Path, dst_path: Path) -> None:
    """Reproject a raster to EPSG:4326 and write to dst_path."""
    with rasterio.open(src_path) as reader:
        if reader.crs and reader.crs.to_epsg() == 4326:
            shutil.copy2(src_path, dst_path)
            return
        transform, width, height = calculate_default_transform(
            reader.crs, "EPSG:4326", reader.width, reader.height, *reader.bounds
        )
        meta = reader.meta.copy()
        meta.update(crs="EPSG:4326", transform=transform,
                    width=width, height=height, driver="GTiff")
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
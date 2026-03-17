""" Utility functions for loading and exploring datasets within the Global Satellite Derived Urban Dataset Validation 
"""
from __future__ import annotations
import os
import re
import requests
from typing import Optional, Union
import datetime
import zipfile
from typing import List, Tuple
from dataclasses import dataclass

import yaml
from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import box
import numpy as np
from src.preprocess import validate_aoi_geometry

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

        # Tolerate various field name conventions
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
    """
    Match Figshare files to a specific grid_id.
    Prefers .zip bundles; falls back to individual shapefile components.
    """
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

    # last resort
    return [f for f in files if f"{gid}_" in f.name]


def download_globfp_grid_tile(config, grid_id: int) -> Path:
    """
    Download tile files for *grid_id* from Figshare (if not cached).
    Returns path to the tile .shp file.
    """
    tiles_dir = globfp_local_dir(config) / "tiles" / f"grid_id={grid_id}"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    # if already cached
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
    """
    Download + unzip world_grid.zip from Zenodo if not already cached.
    Returns path to world_grid.shp.
    """
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
    compute_area_mode: str = "utm",   # "utm" (robust) | "work_crs" (fast)
    logger=None,
) -> gpd.GeoDataFrame:
    """
    Load building footprints, reproject, compute area in m², and filter.

    Parameters
    ----------
    crs_work : str
        CRS used for spatial ops downstream (tiling, sjoin, etc.)
    compute_area_mode : str
        - "utm": compute area in meters using estimate_utm_crs() (recommended)
        - "work_crs": compute area directly in crs_work (only safe if projected in meters)
    """
    path = Path(path)
    gdf = _read_gdf(path)

    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS defined. (GeoParquet should store CRS; check writer.)")

    # Reproject to working CRS for downstream ops
    gdf = gdf.to_crs(crs_work)

    # Optionally repair invalid geometries
    if fix_invalid_geoms:
        gdf = validate_aoi_geometry(gdf, label=path.name)

    # Compute area in true m²
    try:
        if compute_area_mode == "utm":
            metric_crs = gdf.estimate_utm_crs()
            gdf_metric = gdf.to_crs(metric_crs)
            gdf["area_m2"] = gdf_metric.geometry.area
        elif compute_area_mode == "work_crs":
            # Only correct if crs_work is projected in meters
            gdf["area_m2"] = gdf.geometry.area
        else:
            raise ValueError(f"Unknown compute_area_mode={compute_area_mode!r}. Use 'utm' or 'work_crs'.")
    except Exception as e:
        raise ValueError(
            f"Failed to compute area in meters for {path}. "
            f"Check CRS and geometry validity. Underlying error: {e}"
        )

    gdf = gdf[gdf["area_m2"] >= float(min_area_m2)].copy()
    gdf.reset_index(drop=True, inplace=True)

    if logger:
        logger.info("Loaded buildings | n=%d | min_area_m2=%.2f | path=%s", len(gdf), float(min_area_m2), path)

    return gdf

def load_aoi(
    path: Union[str, Path],
    *,
    crs_out: str = "EPSG:4326",
    buffer_meters: float = 0.0,
    dissolve: bool = False,
    logger=None,
) -> gpd.GeoDataFrame:
    """
    Load AOI polygon(s), optionally buffer, reproject, and optionally dissolve.

    Parameters
    ----------
    path : str | Path
        AOI file path (geojson/gpkg/parquet/etc.)
    crs_out : str
        Desired CRS output (e.g., EPSG:4326).
    buffer_meters : float
        Buffer distance in meters (applied in a projected CRS).
    dissolve : bool
        If True, dissolve all features into one geometry (single AOI).
    assume_crs_if_missing : str
        CRS to assume if input is missing CRS metadata.
    logger : logging.Logger | None
        Optional logger for messages.

    Returns
    -------
    GeoDataFrame
    """
    path = Path(path)
    # aoi = gpd.read_file(path)
    aoi = _read_gdf(path)

    if aoi.crs is None: # assume crs if missing
        if logger:
            logger.warning("AOI CRS missing; assuming %s", crs_out)
        aoi = aoi.set_crs("EPSG:4326")

    # optional dissolve early (keeps semantics consistent across pipeline)
    if dissolve and len(aoi) > 1:
        aoi = aoi.dissolve().reset_index(drop=True)

    # buffering in meters requires a projected CRS
    buffer_meters = float(buffer_meters or 0.0)
    if buffer_meters > 0:
        # If AOI CRS is not projected, reproject to a metric CRS for buffer then return
        if not aoi.crs.is_projected:
            if logger:
                logger.warning("AOI CRS is geographic; reprojecting to EPSG:3857 for buffering")
            aoi_metric = aoi.to_crs("EPSG:3857") # why pick EPSG:3857
            aoi_metric["geometry"] = aoi_metric.geometry.buffer(buffer_meters)
            aoi = aoi_metric.to_crs(aoi.crs)
        else:
            aoi["geometry"] = aoi.geometry.buffer(buffer_meters)

    # reproject to desired CRS
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
    """
    Create a regular grid of square tiles covering the AOI.

    Assumes AOI is already in a projected CRS with meters.
    """
    if aoi.empty:
        return gpd.GeoDataFrame({"tile_id": [], "geometry": []}, crs=aoi.crs)

    aoi_union = aoi.geometry.union_all()
    minx, miny, maxx, maxy = aoi_union.bounds
    tile = float(tile_size_m)

    # Optional: snap grid origin to multiples of tile size for consistency across runs/cities
    if snap_origin:
        minx = np.floor(minx / tile) * tile
        miny = np.floor(miny / tile) * tile
        maxx = np.ceil(maxx / tile) * tile
        maxy = np.ceil(maxy / tile) * tile

    # Robust stepping via counts (avoids int truncation + float accumulation)
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
    """
    Return subset of buildings intersecting a tile geometry.

    - uses spatial index for bbox prefilter
    - preserves original indices (no reset_index)
    """
    possible_idx = list(sindex.intersection(tile_geom.bounds))
    if not possible_idx:
        return buildings.iloc[[]].copy()

    subset = buildings.iloc[possible_idx]
    subset = subset[subset.intersects(tile_geom)].copy()
    # IMPORTANT: do NOT reset index here – we want original indices
    return subset

def resolve_out_root(config, dataset_id: str, base_dir: Path) -> Path:
    """
    Output layout mirrors the source layout:
        <base_dir>/<dataset_folder>/vector/
    Falls back to config.output.root_dir / dataset_id if set.
    """
    if config.output.root_dir:
        return Path(config.output.root_dir) / dataset_id / "vector"
    return base_dir / dataset_id / "vector"


def load_all_aois(config) -> List[Dict]:
    """
    Returns a list of dicts, one per dataset:
        {"id": str, "aoi": GeoDataFrame, "out_root": Path, "slug": str}

    Handles two input modes:
      - CSV reference inventory (detected by .csv suffix)
      - Single AOI file (any other extension — legacy mode)
    """
    aoi_config  = config.aoi
    aoi_path = Path(aoi_config.path)

    if aoi_path.suffix.lower() == ".csv":
        return read_csv(config, aoi_path)
    else:
        return read_json(config, aoi_path) # single AOI file, one dataset

def read_csv(config, csv_path: Path) -> List[Dict]:
    """
    Parse the reference inventory CSV.
    Each unique `id_col` value becomes one dataset whose AOI is the
    union of all `|`-delimited paths in `aoi_file_path`.
    """
    aoi_config  = config.aoi
    base_dir = Path(aoi_config.base_dir) if aoi_config.base_dir else csv_path.parent
    id_col   = aoi_config.id_col          # "Dataset code"

    df = pd.read_csv(csv_path)

    if aoi_config.filter_suitable:
        before = len(df)
        df = df[df["Suitable (yes/N)"].str.strip().str.lower() == "yes"]
        print("Filtered inventory: %d -> %d suitable rows", before, len(df))

    # Deduplicate by dataset id — keep first occurrence
    df = df.drop_duplicates(subset=[id_col])

    datasets = []
    for _, row in df.iterrows():
        dataset_id  = str(row[id_col])
        raw_paths   = str(row.get("aoi_file_path", ""))

        # Resolve each `|`-delimited relative path
        aoi_paths = [
            base_dir / p.strip()
            for p in raw_paths.split("|")
            if p.strip()
        ]
        existing = [p for p in aoi_paths if p.exists()]
        if not existing:
            print("Dataset %s: no AOI files found, skipping. Paths tried: %s",
                                dataset_id, aoi_paths[:3])
            continue

        # Load and dissolve all AOI files for this dataset into one geometry
        aoi = load_and_dissolve_aois(config, existing, dataset_id)
        if aoi is None or aoi.empty:
            print("Dataset %s: empty AOI after loading, skipping", dataset_id)
            continue

        slug = dataset_id.replace("-", "_").replace(" ", "_")
        out_root = resolve_out_root(config, dataset_id, base_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        datasets.append({"id": dataset_id, "slug": slug, "aoi": aoi, "out_root": out_root})

    return datasets

def read_json(config, aoi_path: Path) -> List[Dict]:
    """_datasets_from_single_aoi: Legacy mode: wrap a single AOI file as one dataset entry."""
    aoi = load_aoi(
        aoi_path,
        crs_out=config.aoi.crs_out,
        buffer_meters=config.aoi.buffer_meters,
        dissolve=True,
        # logger=self.logger,
    )
    slug     = aoi_path.parent.parent.name.replace("-", "_").replace(" ", "_")
    out_root = aoi_path.parent.parent / "vector"
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_id = aoi_path.parent.parent.name
    return [{"id": dataset_id, "slug": slug, "aoi": aoi, "out_root": out_root}]

def load_and_dissolve_aois(config, paths: List[Path], dataset_id: str) -> Optional[gpd.GeoDataFrame]:
    """Load multiple AOI files and dissolve into a single GeoDataFrame."""
    parts = []
    for p in paths:
        try:
            gdf = load_aoi(p, crs_out=config.aoi.crs_out,
                   buffer_meters=config.aoi.buffer_meters,
                    #    logger=self.logger)
            )
            parts.append(gdf)
        except Exception:
            print("Dataset %s: failed to load AOI %s", dataset_id, p)

    if not parts:
        return None

    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        crs=parts[0].crs,
    )
    return combined.dissolve().reset_index(drop=True) if len(combined) > 1 else combined

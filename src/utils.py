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
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box, mapping
import numpy as np
from shapely import make_valid as _make_valid
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

log = logging.getLogger(__name__)


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

def validate_aoi_geometry(gdf: gpd.GeoDataFrame, label: str = "") -> gpd.GeoDataFrame:
    """
    Fix invalid geometries (self-intersections, rings, etc.) and drop empty geometries.
    Uses shapely.make_valid when available; falls back to buffer(0).
    """
    gdf = gdf.copy()

    # Drop missing/empty early
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    invalid = ~gdf.geometry.is_valid
    n_invalid = int(invalid.sum())

    if n_invalid == 0:
        return gdf

    print(f"[{label}] fixing {n_invalid} invalid geometries...")

    try:
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(_make_valid)
    except Exception:
        # Fallback (less robust, but often works)
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    # Drop anything that became empty after fixing
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    still_invalid = int((~gdf.geometry.is_valid).sum())
    if still_invalid:
        print(f"[{label}] warning: {still_invalid} geometries are still invalid after fixing.")

    return gdf


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
    """
    path = Path(path)
    gdf = (gpd.read_parquet(path) if path.suffix.lower() in {".parquet", ".geoparquet"}
           else gpd.read_file(path))

    if gdf.crs is None:
        raise ValueError(f"{path} has no CRS defined.")

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

    if compute_area_mode == "auto":
        crs_obj = pyproj.CRS(crs_work)
        if crs_obj.is_projected:
            compute_area_mode = "work_crs"
        else:
            compute_area_mode = "utm"

    if compute_area_mode == "work_crs":
        gdf["area_m2"] = gdf.geometry.area

    elif compute_area_mode == "utm":
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
            aoi_metric = aoi.to_crs("EPSG:3857")
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
    """Resolve the output directory for a dataset."""
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

    Returns list of:
        {
            "id":        str,
            "slug":      str,
            "aoi":       GeoDataFrame,          # dissolved union (as before)
            "sub_aois":  list[dict],             # NEW — per-file sub-AOI info
            "is_multi_aoi": bool,                # NEW — convenience flag
        }

    Each entry in sub_aois is:
        {
            "sub_aoi_id": str,                   # e.g. "accra_17589"
            "file_name":  str,                   # original filename
            "geometry":   shapely.geometry.Base,  # individual sub-AOI geometry (EPSG:4326)
        }

    For single-AOI datasets, sub_aois has one entry and is_multi_aoi is False.
    """
    aoi_path = Path(config.aoi.path)
    if aoi_path.suffix.lower() == ".csv":
        return _load_aois_from_csv(config)
    else:
        return _load_aoi_from_file(config, aoi_path)


def _extract_sub_aoi_id(filename: str) -> str:
    """Derive a sub-AOI identifier from the AOI filename.

    Examples:
        "accra_17589_aoi.geojson"  -> "accra_17589"
        "rohingya_3939_aoi.geojson" -> "rohingya_3939"
        "curacao_aoi.geojson"       -> "curacao"
    """
    stem = Path(filename).stem                    # "accra_17589_aoi"
    stem = re.sub(r"_aoi$", "", stem, flags=re.I) # "accra_17589"
    return stem


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
        aoi_entries: List[Tuple[str, Path]] = []   # (filename, full_path)
        for _, row in group.iterrows():
            for part in str(row["aoi_file_name"]).split("|"):
                part = part.strip()
                if part:
                    full_path = base_dir / dataset_id / aoi_cfg.aoi_subdir / part
                    aoi_entries.append((part, full_path))

        existing = [(fname, p) for fname, p in aoi_entries if p.exists()]
        if not existing:
            print(f"Dataset {dataset_id}: no AOI files found on disk, skipping.")
            continue

        # ── Load each sub-AOI individually ──────────────────────────
        sub_aois: List[Dict] = []
        parts_gdf: List[gpd.GeoDataFrame] = []
        for fname, p in existing:
            try:
                gdf_part = load_aoi(p, crs_out=crs_out,
                                    buffer_meters=aoi_cfg.buffer_meters)
                parts_gdf.append(gdf_part)
                # Store individual sub-AOI geometry (dissolved per file)
                sub_geom = gdf_part.union_all()
                sub_aois.append({
                    "sub_aoi_id": _extract_sub_aoi_id(fname),
                    "file_name":  fname,
                    "geometry":   sub_geom,
                })
            except Exception as exc:
                print(f"Dataset {dataset_id}: failed to load AOI {p}: {exc}")

        if not parts_gdf:
            print(f"Dataset {dataset_id}: empty AOI after loading, skipping.")
            continue

        # Build the dissolved union (existing behaviour)
        combined = gpd.GeoDataFrame(
            pd.concat(parts_gdf, ignore_index=True), crs=parts_gdf[0].crs
        )
        aoi = combined.dissolve().reset_index(drop=True) if len(combined) > 1 else combined

        if aoi.empty:
            print(f"Dataset {dataset_id}: empty AOI after dissolve, skipping.")
            continue

        slug = dataset_id.replace("-", "_").replace(" ", "_")
        datasets.append({
            "id":           dataset_id,
            "slug":         slug,
            "aoi":          aoi,
            "sub_aois":     sub_aois,
            "is_multi_aoi": len(sub_aois) > 1,
        })

    return datasets


def _load_aoi_from_file(config, aoi_path: Path) -> List[Dict]:
    aoi_cfg    = config.aoi
    aoi        = load_aoi(aoi_path, crs_out=aoi_cfg.crs_out,
                          buffer_meters=aoi_cfg.buffer_meters, dissolve=True)
    dataset_id = aoi_path.parent.parent.name
    slug       = dataset_id.replace("-", "_").replace(" ", "_")

    sub_geom = aoi.union_all()
    return [{
        "id":           dataset_id,
        "slug":         slug,
        "aoi":          aoi,
        "sub_aois":     [{
            "sub_aoi_id": _extract_sub_aoi_id(aoi_path.name),
            "file_name":  aoi_path.name,
            "geometry":   sub_geom,
        }],
        "is_multi_aoi": False,
    }]


# ─────────────────────────────────────────────────────────────────────
#  NEW: tag buildings with their sub-AOI membership
# ─────────────────────────────────────────────────────────────────────

def build_sub_aoi_gdf(sub_aois: List[Dict], crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from the sub_aois list for spatial joins."""
    return gpd.GeoDataFrame(
        [{"sub_aoi_id": s["sub_aoi_id"], "geometry": s["geometry"]}
         for s in sub_aois],
        crs=crs,
    )

def tag_buildings_with_sub_aoi(
    buildings_gdf: gpd.GeoDataFrame,
    sub_aois: list,
) -> gpd.GeoDataFrame:
    """
    Spatially tag each building with the sub-AOI it falls in.

    Returns a GeoDataFrame with:
      - all original building columns
      - geometry
      - exactly one `sub_aoi_id` column

    It will not leak join artifact columns such as:
      - index_left
      - index_right
      - sub_aoi_id_left
      - sub_aoi_id_right

    For multi-AOI datasets, buildings falling in gaps between sub-AOIs are dropped.
    """

    if buildings_gdf is None or buildings_gdf.empty:
        out = buildings_gdf.copy()
        if "sub_aoi_id" not in out.columns:
            out["sub_aoi_id"] = pd.Series(dtype="object")
        return out

    if not sub_aois:
        out = buildings_gdf.copy()
        if "sub_aoi_id" not in out.columns:
            out["sub_aoi_id"] = pd.Series([None] * len(out), index=out.index, dtype="object")
        return out

    gdf = buildings_gdf.copy()

    # Remove any previous join-artifact columns so repeated calls stay clean
    artifact_cols = [
        c for c in gdf.columns
        if c in {"index_left", "index_right", "sub_aoi_id_left", "sub_aoi_id_right"}
        or c.startswith("sub_aoi_id_")
    ]
    if artifact_cols:
        gdf = gdf.drop(columns=artifact_cols, errors="ignore")

    # Also remove duplicate column names before joining
    if gdf.columns.duplicated().any():
        gdf = gdf.loc[:, ~gdf.columns.duplicated()].copy()

    # Preserve original column order
    original_cols = list(gdf.columns)

    # Build sub-AOI GeoDataFrame
    sub_aoi_gdf = gpd.GeoDataFrame(
        [
            {
                "sub_aoi_id": s["sub_aoi_id"],
                "geometry": s["geometry"],
            }
            for s in sub_aois
        ],
        geometry="geometry",
        crs=gdf.crs,
    )

    if sub_aoi_gdf.crs != gdf.crs:
        sub_aoi_gdf = sub_aoi_gdf.to_crs(gdf.crs)

    # Spatial join: keep only buildings that intersect a sub-AOI
    joined = gpd.sjoin(
        gdf,
        sub_aoi_gdf[["sub_aoi_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )

    # Clean up join artifacts
    joined = joined.drop(columns=["index_right"], errors="ignore")

    # Normalize sub_aoi_id to exactly one column
    if "sub_aoi_id_right" in joined.columns:
        joined = joined.rename(columns={"sub_aoi_id_right": "sub_aoi_id"})
    elif "sub_aoi_id_left" in joined.columns and "sub_aoi_id" not in joined.columns:
        joined = joined.rename(columns={"sub_aoi_id_left": "sub_aoi_id"})

    joined = joined.drop(columns=["sub_aoi_id_left", "sub_aoi_id_right"], errors="ignore")

    # Remove duplicate column names if any remain
    if joined.columns.duplicated().any():
        joined = joined.loc[:, ~joined.columns.duplicated()].copy()

    # Keep exactly original columns + one sub_aoi_id
    final_cols = [c for c in original_cols if c in joined.columns and c != "sub_aoi_id"]
    final_cols += ["sub_aoi_id"]

    joined = joined[final_cols].copy()

    # Re-wrap as GeoDataFrame to preserve geometry metadata
    joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=gdf.crs)

    # Optional logging
    dropped = len(gdf) - len(joined)
    print(
        f"Tagged buildings: {len(gdf)} total, {len(joined)} matched sub-AOIs, {dropped} dropped (in gaps)"
    )

    return joined

# ─────────────────────────────────────────────────────────────────────
#  NEW: mask raster to sub-AOI geometries
# ─────────────────────────────────────────────────────────────────────

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

    log.info("Masked raster -> %s  (nodata=%s, %d sub-AOI polygons)",
             out_path, nodata, len(sub_aois))
    return out_path


# ─────────────────────────────────────────────────────────────────────
#  Unchanged utility functions
# ─────────────────────────────────────────────────────────────────────

def load_validation_datasets(cfg: dict, data_dir: Path) -> List[Dict]:
    """Read the AOI tracker CSV and return datasets ready for validation.

    Mirrors the sub-AOI loading done for downloads so that multi-AOI datasets
    (cities with several scattered AOI files) are handled consistently.

    Each entry in the returned list:
        {
            "id":           str,            # dataset_folder_name from tracker
            "slug":         str,            # id with hyphens/spaces -> underscores
            "aoi":          GeoDataFrame,   # dissolved union of all sub-AOIs
            "sub_aois":     list[dict],     # per-file: {sub_aoi_id, file_name, geometry}
            "is_multi_aoi": bool,
            "ref_path":     Path | None,    # resolved path to reference buildings file
        }
    """
    root         = Path(cfg.get("root_dir", "."))
    tracker_path = Path(cfg["aoi_tracker"])
    if not tracker_path.is_absolute():
        tracker_path = root / tracker_path

    df = pd.read_csv(tracker_path, dtype=str)
    df.columns = df.columns.str.strip()

    # Suitable filter — same logic as downloader
    suitable_col = next((c for c in df.columns if "suitable" in c.lower()), None)
    if suitable_col:
        before = len(df)
        df = df[df[suitable_col].astype(str).str.lower() == "yes"].copy()
        log.info("Validation tracker: %d -> %d suitable rows.", before, len(df))

    id_col = "dataset_folder_name"
    df = df.dropna(subset=[id_col]).copy()
    df = df[df[id_col].str.strip() != ""]

    # Identify reference file column
    ref_col = next(
        (c for c in df.columns if "reference" in c.lower() and "file" in c.lower()), None
    )

    datasets: List[Dict] = []

    for dataset_id, group in df.groupby(id_col, sort=False):
        dataset_id = str(dataset_id)

        # Collect all AOI file paths (pipe-separated values and/or multiple rows)
        aoi_entries: List[Tuple[str, Path]] = []
        for _, row in group.iterrows():
            raw_aoi = str(row.get("aoi_file_name", "") or "")
            for part in raw_aoi.split("|"):
                part = part.strip()
                if part:
                    aoi_entries.append((part, data_dir / dataset_id / "aoi" / part))

        existing = [(fname, p) for fname, p in aoi_entries if p.exists()]
        if not existing:
            log.warning("Validation | %s: no AOI files found on disk, skipping.", dataset_id)
            continue

        # Load each sub-AOI individually (same pattern as download pipeline)
        sub_aois: List[Dict] = []
        parts_gdf: List[gpd.GeoDataFrame] = []
        for fname, p in existing:
            try:
                gdf_part = load_aoi(p, crs_out="EPSG:4326")
                parts_gdf.append(gdf_part)
                sub_aois.append({
                    "sub_aoi_id": _extract_sub_aoi_id(fname),
                    "file_name":  fname,
                    "geometry":   gdf_part.union_all(),
                })
            except Exception as exc:
                log.warning("Validation | %s: failed to load AOI %s: %s", dataset_id, p, exc)

        if not parts_gdf:
            log.warning("Validation | %s: empty AOI after loading, skipping.", dataset_id)
            continue

        combined = gpd.GeoDataFrame(
            pd.concat(parts_gdf, ignore_index=True), crs=parts_gdf[0].crs
        )
        aoi = combined.dissolve().reset_index(drop=True) if len(combined) > 1 else combined

        # Resolve reference file path(s) — the tracker cell may contain a
        # pipe-separated list for multi-AOI datasets (e.g. bgd-rohingya).
        ref_paths: List[Path] = []
        if ref_col:
            for _, row in group.iterrows():
                raw_ref = str(row.get(ref_col, "") or "")
                for part in raw_ref.split("|"):
                    part = part.strip()
                    if part:
                        p = data_dir / dataset_id / "vector" / part
                        if p not in ref_paths:
                            ref_paths.append(p)

        slug = dataset_id.replace("-", "_").replace(" ", "_")
        datasets.append({
            "id":           dataset_id,
            "slug":         slug,
            "aoi":          aoi,
            "sub_aois":     sub_aois,
            "is_multi_aoi": len(sub_aois) > 1,
            "ref_paths":    ref_paths,
            # Back-compat: single-file ref as before; None for multi-file cases
            "ref_path":     ref_paths[0] if len(ref_paths) == 1 else None,
        })

    log.info("Loaded %d dataset(s) for validation.", len(datasets))
    return datasets


def get_projected_crs(gdf: gpd.GeoDataFrame) -> str:
    if gdf.crs and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)
    bounds = gdf.total_bounds
    lon = (bounds[0] + bounds[2]) / 2
    lat = (bounds[1] + bounds[3]) / 2
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=lon, south_lat_degree=lat,
            east_lon_degree=lon, north_lat_degree=lat,
        ),
    )
    if utm_crs_list:
        return f"EPSG:{utm_crs_list[0].code}"
    zone_number = int((lon + 180) / 6) + 1
    return f"EPSG:{32600 + zone_number}" if lat >= 0 else f"EPSG:{32700 + zone_number}"

def init_earth_engine(project: str | None = None) -> None:
    """
    Initialize Earth Engine.

    Behavior:
    - First try existing credentials.
    - If unavailable and running in an interactive notebook kernel, prompt auth.
    - If unavailable and running as a plain script, raise a helpful error instead
      of crashing inside Colab widget auth.
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

def _shapely_to_geojson_dict(geom) -> dict:
    return json.loads(
        gpd.GeoSeries([geom], crs="EPSG:4326").to_json()
    )["features"][0]["geometry"]

def aoi_gdf_to_ee_geometry(gdf):
    import ee
    return ee.Geometry(_shapely_to_geojson_dict(gdf.union_all()))

def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
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
    if "data" in columns:
        return "data"
    candidates = [c for c in columns if any(k in c.lower() for k in ("url", "href", "path", "link"))]
    if candidates:
        return candidates[0]
    raise ValueError(f"Cannot find a URL column in tile index. Columns present: {list(columns)}")

def reproject_to_4326(src_path: Path, dst_path: Path) -> None:
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


def consolidate_match_chunks(
    metrics_dir: Path, ds_name: str, final_path: Path
) -> None:
    """Read all temp match chunks for ds_name, concat, write final file, delete temps."""
    chunk_files = sorted(metrics_dir.glob(f"_tmp_matches_{ds_name}_*.parquet"))
    if chunk_files:
        df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
        df.to_parquet(final_path, index=False)
        del df
        for f in chunk_files:
            f.unlink()
    else:
        pd.DataFrame(columns=[
            "ref_id", "cand_id", "iou", "area_ref", "area_cand",
            "rel_area_error", "city", "dataset", "tile_id",
        ]).to_parquet(final_path, index=False)


def log_memory(label: str = "") -> None:
    """Log current process RSS — useful for spotting memory leaks."""
    try:
        import psutil
        rss_mb = psutil.Process().memory_info().rss / 1024 ** 2
        log.info("MEM [%s] RSS = %.0f MB", label, rss_mb)
    except Exception:
        pass
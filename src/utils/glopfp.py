"""
GloBFP download helpers.

Resolves which Figshare article hosts a given grid ID, lists files,
selects the right files for a tile, downloads them (with .zip
extraction), and bootstraps the Zenodo world grid shapefile.
"""
from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import requests

log = logging.getLogger(__name__)

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
    """Map a GloBFP grid ID to its Figshare article ID."""
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
    """Stream a single Figshare/Zenodo file to dest with a .part swap."""
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
    grid_field = next((c for c in cand_fields if c in grid.columns), None)
    if grid_field is None:
        raise ValueError(
            f"Cannot find grid-ID field in world_grid. Columns: {list(grid.columns)}"
        )

    hits = grid[grid.intersects(aoi_geom_4326)]
    return sorted({int(v) for v in hits[grid_field]})


def get_figshare_list_files(article_id: int) -> List[FigshareFile]:
    """List files attached to a Figshare article."""
    url = f"{FIGSHARE_API}/articles/{article_id}"
    r = requests.get(url, timeout=60)
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
    files: List[FigshareFile],
    grid_id: int,
) -> List[FigshareFile]:
    """Pick the file(s) belonging to a given grid_id, preferring zips."""
    gid = str(grid_id)
    zips = [
        f for f in files
        if f.name.lower().endswith(".zip")
        and re.match(rf"^{gid}[_\-]", f.name)
    ]
    if zips:
        return zips

    components = [
        f for f in files
        if re.match(
            rf"^{gid}[_\-].*\.(shp|dbf|shx|prj|cpg)$",
            f.name,
            flags=re.IGNORECASE,
        )
    ]
    if components:
        return components

    return [f for f in files if f"{gid}_" in f.name]


def download_globfp_grid_tile(config, grid_id: int) -> Path:
    """Download (and unzip if needed) the GloBFP tile shapefile for grid_id."""
    tiles_dir = globfp_local_dir(config) / "tiles" / f"grid_id={grid_id}"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    existing = list(tiles_dir.glob("*.shp"))
    if existing:
        return existing[0]

    article_id = get_article_id(grid_id)
    all_files = get_figshare_list_files(article_id)
    selected = select_globfp_tile_files(all_files, grid_id)

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
    """Ensure the GloBFP world-grid shapefile is locally available; return its path."""
    local_dir = globfp_local_dir(config)
    record_id = config.globfp.zenodo_record
    zip_path = local_dir / "world_grid.zip"
    grid_dir = local_dir / "world_grid"
    shp_path = grid_dir / "world_grid.shp"

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
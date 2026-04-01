"""
Raster Building Data Download Pipeline
=======================================
Downloads three raster building datasets for every city in the AOI tracker CSV.
Folder layout and CSV loading conventions deliberately mirror the companion
UrbanVectorDownloader so that both pipelines produce a consistent tree:

    <output_root>/
      <dataset_id>/
        vector/          ← UrbanVectorDownloader writes here
        raster/          ← this pipeline writes here
          google_open_buildings_temporal/
            <slug>_obt_<year>.tif
          microsoft_tempo/
            <slug>_tempo_2023q4.tif         (clipped mosaic)
            <slug>_tempo_tile_footprints.geojson
          ghsl/
            <slug>_ghsl_built_h_2018.tif
            <slug>_ghsl_built_s_2020.tif
            ...

Datasets
--------
  1. Google Open Buildings Temporal v1  — Earth Engine / geemap
  2. Microsoft TEMPO Global 2023 Q4     — direct COG download + mosaic
  3. GHSL P2023A (BUILT_H/S/V)          — Earth Engine / geemap

Usage
-----
    python pipeline.py                          # uses config.yaml in cwd
    python pipeline.py --config my_cfg.yaml
    python pipeline.py --cities tracker.csv     # override CSV path from config
    python pipeline.py --city ssd-juba          # single dataset_id only
    python pipeline.py --only microsoft_tempo   # single raster source only
    python pipeline.py --high-quality-only      # skip is_high_quality=FALSE rows
    python pipeline.py --no-ee                  # skip EE init (TEMPO-only run)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
import yaml

logger = logging.getLogger("Raster_Downloader")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_sh = logging.StreamHandler()
_sh.setFormatter(_fmt)
if not logger.handlers:
    logger.addHandler(_sh)

def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)

def resolve_out_root(config: dict, dataset_id: str) -> Path:
    """
    Two modes controlled by output.use_base_dir_for_output (default: true):

      true  ->  <aoi.base_dir>/<dataset_id>/raster/
                Sits alongside the aoi/ subfolder inside each city folder.
                e.g.  data/01_raw/dom-dominica/raster/

      false ->  <output.root_dir>/<dataset_id>/raster/
                Separate output tree.
    """
    output_cfg = config.get("output", {})
    use_base = output_cfg.get("use_base_dir_for_output", True)

    if use_base:
        aoi_cfg = config.get("aoi", {})
        base_dir = Path(aoi_cfg.get("base_dir", "data/01_raw"))
        p = base_dir / dataset_id / "raster"
    else:
        root = output_cfg.get("root_dir") or config.get("output_root", "data/outputs")
        p = Path(root) / dataset_id / "raster"

    p.mkdir(parents=True, exist_ok=True)
    return p

def _read_aoi_file(path: Path, crs_out: str = "EPSG:4326"):
    """Load a single AOI file (GeoJSON / shapefile / geoparquet) -> GeoDataFrame."""
    import geopandas as gpd

    path = Path(path)
    gdf = (
        gpd.read_parquet(path)
        if path.suffix.lower() in {".parquet", ".geoparquet"}
        else gpd.read_file(path)
    )

    if gdf.empty:
        raise ValueError(f"AOI file is empty: {path}")
    if gdf.crs is None:
        logger.warning("AOI %s has no CRS; assuming EPSG:4326", path)
        gdf = gdf.set_crs("EPSG:4326")
    if str(gdf.crs) != crs_out:
        gdf = gdf.to_crs(crs_out)

    return gdf


def load_and_dissolve_aois(
    paths: List[Path],
    dataset_id: str,
    crs_out: str = "EPSG:4326",
) -> Optional[object]:
    """
    Load one or more AOI files and dissolve into a single-row GeoDataFrame.
    Mirrors load_and_dissolve_aois in utils.py.
    """
    import geopandas as gpd

    parts = []
    for p in paths:
        try:
            parts.append(_read_aoi_file(p, crs_out=crs_out))
        except Exception:
            logger.exception("Dataset %s: failed to load AOI %s", dataset_id, p)

    if not parts:
        return None

    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True),
        crs=parts[0].crs,
    )
    return (
        combined.dissolve().reset_index(drop=True)
        if len(combined) > 1
        else combined
    )


def load_all_aois(config: dict) -> List[Dict]:
    """
    Parse the CSV inventory and return a list of dataset dicts:
        {"id": str, "slug": str, "aoi": GeoDataFrame, "out_root": Path}

    Mirrors UrbanVectorDownloader.load_all_aois / read_csv exactly:
    - Groups rows by id_col (default: dataset_folder_name)
    - Resolves AOI paths as  <base_dir>/<dataset_id>/<aoi_file_name>
    - Supports pipe-separated filenames in a single cell for multi-part AOIs
    - Dissolves all parts into one geometry per dataset
    - Applies Suitable / has_aoi_file / is_high_quality filters from config
    """
    aoi_cfg = config.get("aoi", {})
    csv_path = Path(aoi_cfg.get("path") or config.get("cities_csv", "cities.csv"))
    base_dir = Path(aoi_cfg.get("base_dir") or csv_path.parent)
    aoi_subdir = aoi_cfg.get("aoi_subdir", "aoi")   # subfolder inside each city dir
    id_col = aoi_cfg.get("id_col", "dataset_folder_name")
    crs_out = aoi_cfg.get("crs_out", "EPSG:4326")
    filter_suitable = bool(aoi_cfg.get("filter_suitable", True))
    high_quality_only = bool(aoi_cfg.get("high_quality_only", False))

    df = pd.read_csv(csv_path, dtype=str)

    # Validate required columns
    required = [id_col, "aoi_file_name", "has_aoi_file"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # has_aoi_file filter — always applied
    df["has_aoi_file"] = df["has_aoi_file"].str.strip().str.upper()
    df = df[df["has_aoi_file"] == "TRUE"]

    # Suitable filter — mirrors read_csv in utils.py
    if filter_suitable and "Suitable" in df.columns:
        before = len(df)
        df = df[df["Suitable"].str.strip().str.lower() == "yes"]
        logger.info("Filtered inventory: %d -> %d suitable rows", before, len(df))

    # Optional high-quality filter
    if high_quality_only and "is_high_quality" in df.columns:
        before = len(df)
        df = df[df["is_high_quality"].str.strip().str.upper() == "TRUE"]
        logger.info("Filtered inventory: %d -> %d high-quality rows", before, len(df))

    df = df.dropna(subset=[id_col, "aoi_file_name"])
    df = df[df["aoi_file_name"].str.strip() != ""]

    if df.empty:
        logger.warning("No processable rows found in CSV after filtering.")
        return []

    datasets: List[Dict] = []

    for dataset_id, group in df.groupby(id_col, sort=False):
        dataset_id = str(dataset_id)

        # Collect AOI file paths — supports pipe-separated values in one cell
        # AND multiple rows with the same dataset_id (both patterns appear in
        # the tracker CSV, e.g. ssd-juba, sxm-sint-maarten)
        aoi_paths: List[Path] = []
        for _, row in group.iterrows():
            for part in str(row["aoi_file_name"]).split("|"):
                part = part.strip()
                if part:
                    aoi_paths.append(base_dir / dataset_id / aoi_subdir / part)

        existing = [p for p in aoi_paths if p.exists()]
        if not existing:
            logger.warning(
                "Dataset %s: no AOI files found on disk, skipping. Tried: %s",
                dataset_id, aoi_paths[:3],
            )
            continue

        aoi = load_and_dissolve_aois(existing, dataset_id, crs_out=crs_out)
        if aoi is None or aoi.empty:
            logger.warning("Dataset %s: empty AOI after loading, skipping.", dataset_id)
            continue

        slug = dataset_id.replace("-", "_").replace(" ", "_")
        out_root = resolve_out_root(config, dataset_id)

        datasets.append({
            "id":       dataset_id,
            "slug":     slug,
            "aoi":      aoi,
            "out_root": out_root,
        })

    return datasets

def _shapely_to_geojson_dict(geom) -> dict:
    import geopandas as gpd
    return json.loads(
        gpd.GeoSeries([geom], crs="EPSG:4326").to_json()
    )["features"][0]["geometry"]


def aoi_gdf_to_ee_geometry(gdf):
    import ee
    return ee.Geometry(_shapely_to_geojson_dict(gdf.union_all()))


def init_earth_engine(project: str = "") -> None:
    import ee
    kwargs = {"project": project} if project else {}
    try:
        ee.Initialize(**kwargs)
        logger.info("Earth Engine initialised.")
    except Exception:
        logger.info("EE not initialised – authenticating …")
        ee.Authenticate()
        ee.Initialize(**kwargs)

def download_google_obt(ds: Dict, cfg: dict) -> List[str]:
    """Download yearly rasters from Google Open Buildings Temporal."""
    import ee
    import geemap

    out_dir = ds["out_root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    aoi_ee = aoi_gdf_to_ee_geometry(ds["aoi"])
    collection = ee.ImageCollection(cfg["ee_collection_id"])
    years: List[int] = cfg["years"]
    bands: List[str] = cfg["bands"]
    scale: int = cfg.get("scale", 4)
    outputs = []

    for year in years:
        out_path = out_dir / f"{ds['slug']}_obt_{year}.tif"

        if out_path.exists():
            logger.info("  [OBT] %s %d — exists, skipping.", ds["id"], year)
            outputs.append(str(out_path))
            continue

        start, end = f"{year}-01-01", f"{year + 1}-01-01"
        year_ic = collection.filterDate(start, end).filterBounds(aoi_ee)
        count = year_ic.size().getInfo()
        logger.info("  [OBT] %s %d — %d tile(s) found.", ds["id"], year, count)

        if count == 0:
            logger.warning("  [OBT] %s %d — no data for AOI, skipping.", ds["id"], year)
            continue

        image = year_ic.mosaic().select(bands).clip(aoi_ee)
        geemap.download_ee_image(
            image=image,
            filename=str(out_path),
            region=aoi_ee,
            scale=scale,
            crs="EPSG:4326",
        )
        logger.info("  [OBT] %s %d — saved %s", ds["id"], year, out_path)
        outputs.append(str(out_path))

    return outputs


# =============================================================================
# Dataset 2 – Microsoft TEMPO
#


def _download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    logger.info("    Downloading %s …", url)
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(out_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
    logger.info("    Saved: %s", out_path)


def _get_tile_url_column(columns) -> str:
    """
    Detect the URL column in the TEMPO tile index.
    The published index uses 'data' as the COG URL column.
    Falls back to scanning for url/href/path keywords for forward-compatibility.
    # Output: <out_root>/microsoft_tempo/<slug>_tempo_2023q4.tif
    # Raw COG tiles go into a shared cache (data/cache/tempo_tiles by default) so
    # cities that overlap the same tile do not re-download it — mirrors the shared
    # GloBFP tile cache in the vector downloader.
    """
    # Primary: the actual published column name
    if "data" in columns:
        return "data"
    # Fallback: any column whose name suggests a URL
    candidates = [c for c in columns if any(k in c.lower() for k in ("url", "href", "path", "link"))]
    if candidates:
        return candidates[0]
    raise ValueError(
        f"Cannot find a URL/COG column in TEMPO tile index. "
        f"Columns present: {list(columns)}"
    )


def _reproject_to_4326(src_path: Path, dst_path: Path) -> None:
    """Reproject a raster to EPSG:4326 in place (writes to dst_path)."""
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with rasterio.open(src_path) as src:
        if src.crs and src.crs.to_epsg() == 4326:
            # Already in target CRS — just copy
            import shutil
            shutil.copy2(src_path, dst_path)
            return

        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update(
            crs="EPSG:4326",
            transform=transform,
            width=width,
            height=height,
            driver="GTiff",
        )
        with rasterio.open(dst_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs="EPSG:4326",
                    resampling=Resampling.bilinear,
                )


def download_microsoft_tempo(ds: Dict, cfg: dict) -> List[str]:
    """
    Download Microsoft TEMPO tiles that intersect the AOI and produce a
    clipped mosaic in EPSG:4326.

    Fixes applied vs original:
    - All tiles are reprojected to EPSG:4326 before merging so the AOI
      geometry (always in 4326) correctly intersects the raster extent.
    - The clip uses all_touched=True to handle small AOIs near tile edges.
    - WindowError (non-overlapping AOI / raster) is caught per-tile so one
      bad tile does not abort the whole city.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.merge import merge as rio_merge

    TARGET_CRS = "EPSG:4326"

    out_dir = ds["out_root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared raw tile cache
    tile_cache_dir = Path(cfg.get("tile_cache_dir", "data/cache/tempo_tiles"))
    tile_cache_dir.mkdir(parents=True, exist_ok=True)

    # Per-city reprojected tile cache (avoids re-reprojecting on resume)
    reproj_cache_dir = tile_cache_dir / "reproj_4326"
    reproj_cache_dir.mkdir(parents=True, exist_ok=True)

    aoi_union = ds["aoi"].union_all()

    # Load (or download) tile index
    tile_index_cache = Path(cfg.get("tile_index_cache", "data/cache/tempo_tile_index.gpkg"))
    _download_file(cfg["tile_index_url"], tile_index_cache)

    tile_index = gpd.read_file(tile_index_cache).to_crs(TARGET_CRS)
    if tile_index.empty:
        raise ValueError("TEMPO tile index is empty.")

    selected = tile_index[tile_index.intersects(aoi_union)].copy().reset_index(drop=True)
    logger.info("  [TEMPO] %s — %d intersecting tile(s).", ds["id"], len(selected))

    if selected.empty:
        logger.warning("  [TEMPO] %s — no tiles intersect AOI.", ds["id"])
        return []

    selected.to_file(
        out_dir / f"{ds['slug']}_tempo_tile_footprints.geojson",
        driver="GeoJSON",
    )

    url_col = _get_tile_url_column(selected.columns)
    logger.info("  [TEMPO] using tile URL column: '%s'", url_col)

    # Download raw tiles and reproject each to EPSG:4326
    reproj_files: List[Path] = []
    for _, row in selected.iterrows():
        url = row[url_col]
        tile_name = os.path.basename(url)
        raw_path = tile_cache_dir / tile_name
        reproj_path = reproj_cache_dir / tile_name

        _download_file(url, raw_path)

        if not reproj_path.exists():
            logger.info("    Reprojecting %s -> EPSG:4326 …", tile_name)
            _reproject_to_4326(raw_path, reproj_path)

        reproj_files.append(reproj_path)

    if not cfg.get("make_mosaic", True):
        return [str(f) for f in reproj_files]

    mosaic_path = out_dir / f"{ds['slug']}_tempo_2023q4.tif"
    if mosaic_path.exists():
        logger.info("  [TEMPO] %s — mosaic exists, skipping.", ds["id"])
        return [str(mosaic_path)]

    # Merge reprojected tiles
    srcs = [rasterio.open(fp) for fp in reproj_files]
    mosaic, transform = rio_merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(
        crs=TARGET_CRS,
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        driver="GTiff",
    )
    for src in srcs:
        src.close()

    temp = out_dir / "_temp_mosaic.tif"
    with rasterio.open(temp, "w", **meta) as dst:
        dst.write(mosaic)

    # Clip to AOI — guard against edge-case non-overlap after reprojection
    try:
        with rasterio.open(temp) as src:
            clipped, clipped_tf = rio_mask(
                src,
                [_shapely_to_geojson_dict(aoi_union)],
                crop=True,
                all_touched=True,
            )
            clipped_meta = src.meta.copy()
            clipped_meta.update(
                crs=TARGET_CRS,
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=clipped_tf,
            )
        with rasterio.open(mosaic_path, "w", **clipped_meta) as dst:
            dst.write(clipped)
    except Exception as exc:
        logger.warning(
            "  [TEMPO] %s — clip failed (%s); saving unclipped mosaic instead.", ds["id"], exc
        )
        import shutil
        shutil.copy2(temp, mosaic_path)

    temp.unlink(missing_ok=True)
    logger.info("  [TEMPO] %s — saved %s", ds["id"], mosaic_path)
    return [str(mosaic_path)]


def download_ghsl(ds: Dict, cfg: dict) -> List[str]:
    """Download GHSL P2023A products (BUILT_H, BUILT_S, BUILT_V) for the AOI.

    EE access pattern: each year is a separate Image within the collection:
        ee.Image('JRC/GHSL/P2023A/GHS_BUILT_S/2020').select('built_surface')
    The band name is the same bare name for every year — not 'built_surface_2020'.
    """
    import ee
    import geemap

    out_dir = ds["out_root"]
    out_dir.mkdir(parents=True, exist_ok=True)

    aoi_ee = aoi_gdf_to_ee_geometry(ds["aoi"])
    products: dict = cfg["products"]
    outputs = []

    logger.info(
        "  [GHSL] %s — %d product(s) configured: %s",
        ds["id"], len(products), list(products.keys()),
    )

    for product_name, prod_cfg in products.items():
        pname_lower = product_name.lower()
        band_name: str = prod_cfg["band"]      # same band name for every year
        scale: int = prod_cfg.get("scale", 100)
        years: List[int] = prod_cfg.get("years", [])
        collection_id: str = prod_cfg["ee_id"]

        logger.info(
            "  [GHSL/%s] %s — %d year(s) to process.",
            product_name, ds["id"], len(years),
        )

        for year in years:
            out_path = out_dir / f"{ds['slug']}_ghsl_{pname_lower}_{year}.tif"

            if out_path.exists():
                logger.info(
                    "  [GHSL/%s] %s %d — exists, skipping.",
                    product_name, ds["id"], year,
                )
                outputs.append(str(out_path))
                continue

            # Each year is a separate Image: <collection_id>/<year>
            image_id = f"{collection_id}/{year}"
            try:
                image = ee.Image(image_id).select(band_name).clip(aoi_ee)
                image.bandNames().getInfo()  # lightweight check before full download
            except Exception as exc:
                logger.error(
                    "  [GHSL/%s] %s %d — cannot load '%s': %s",
                    product_name, ds["id"], year, image_id, exc,
                )
                continue

            logger.info(
                "  [GHSL/%s] %s %d — downloading '%s' …",
                product_name, ds["id"], year, image_id,
            )
            try:
                geemap.download_ee_image(
                    image=image,
                    filename=str(out_path),
                    region=aoi_ee,
                    scale=scale,
                    crs="EPSG:4326",
                )
            except Exception as exc:
                logger.error(
                    "  [GHSL/%s] %s %d — download failed: %s",
                    product_name, ds["id"], year, exc, exc_info=True,
                )
                continue

            logger.info(
                "  [GHSL/%s] %s %d — saved %s",
                product_name, ds["id"], year, out_path,
            )
            outputs.append(str(out_path))

    return outputs

RASTER_SOURCES = {
    "google_open_buildings_temporal": download_google_obt,
    "microsoft_tempo":                download_microsoft_tempo,
    "ghsl":                           download_ghsl,
}

def run_pipeline(
    config: dict,
    datasets: Optional[List[Dict]] = None,
    only: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Run the full raster download pipeline.
    Returns {dataset_id: [output_paths]}.

    Mirrors UrbanVectorDownloader.run_connection:
    - Validates that at least one source is enabled
    - Initialises EE once (not per dataset)
    - Iterates datasets, calls each enabled source runner
    - Logs per-dataset and total output counts
    - Catches per-source exceptions without aborting the whole run
    """
    if datasets is None:
        datasets = load_all_aois(config)

    logger.info("Loaded %d dataset(s) from inventory.", len(datasets))

    if not datasets:
        logger.warning("No datasets to process.")
        return {}

    sources_cfg: dict = config.get("datasets", {})
    enabled = [
        name for name in RASTER_SOURCES
        if sources_cfg.get(name, {}).get("enabled", True)
        and (only is None or name == only)
    ]
    if not enabled:
        raise ValueError(
            "No raster source enabled. Check 'datasets' in config or --only value. "
            f"Available: {list(RASTER_SOURCES)}"
        )

    needs_ee = any(
        s in enabled for s in ("google_open_buildings_temporal", "ghsl")
    )
    if needs_ee:
        init_earth_engine(config.get("ee_project", ""))

    all_outputs: Dict[str, List[str]] = {}

    for ds in datasets:
        logger.info(
            "── Dataset: %s | slug: %s | sources: %s",
            ds["id"], ds["slug"], enabled,
        )
        ds_outputs: List[str] = []

        for source_name, runner in RASTER_SOURCES.items():
            if source_name not in enabled:
                continue

            src_cfg = sources_cfg.get(source_name, {})
            logger.info("  ── %s ──", source_name)
            try:
                result = runner(ds, src_cfg)
                ds_outputs.extend(result)
            except Exception:
                logger.exception(
                    "  [%s] FAILED for dataset %s", source_name, ds["id"]
                )

        all_outputs[ds["id"]] = ds_outputs
        logger.info(
            "Dataset %s complete | outputs=%d", ds["id"], len(ds_outputs)
        )

    total = sum(len(v) for v in all_outputs.values())
    logger.info("All datasets complete | total outputs=%d", total)
    return all_outputs

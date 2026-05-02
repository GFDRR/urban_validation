"""
AOI inventory loaders and sub-AOI tagging.

Reads the dataset inventory (CSV or single AOI file), assembles per-city
records with dissolved AOIs and per-file sub-AOI metadata, and provides
the spatial join used to tag each downloaded building with its sub-AOI.

Multi-AOI cities are first-class: cities like bgd-rohingya, ken-nairobi,
or gha-accra have several scattered sub-AOIs and are returned with
is_multi_aoi=True plus a sub_aois list of {sub_aoi_id, file_name, geometry}.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import pandas as pd

from src.utils.buildings import load_aoi

log = logging.getLogger(__name__)


def load_all_aois(config) -> List[Dict]:
    """Parse the AOI inventory and return a list of dataset dicts.

    Returns list of:
        {
            "id":           str,
            "slug":         str,
            "aoi":          GeoDataFrame,           # dissolved union
            "sub_aois":     list[dict],             # per-file sub-AOI info
            "is_multi_aoi": bool,                   # convenience flag
        }

    Each entry in sub_aois is:
        {
            "sub_aoi_id": str,                      # e.g. "accra_17589"
            "file_name":  str,                      # original filename
            "geometry":   shapely.geometry.Base,    # individual sub-AOI (EPSG:4326)
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
        "accra_17589_aoi.geojson"   -> "accra_17589"
        "rohingya_3939_aoi.geojson" -> "rohingya_3939"
        "curacao_aoi.geojson"       -> "curacao"
    """
    stem = Path(filename).stem                     # "accra_17589_aoi"
    stem = re.sub(r"_aoi$", "", stem, flags=re.I)  # "accra_17589"
    return stem


def _load_aois_from_csv(config) -> List[Dict]:
    aoi_cfg = config.aoi
    csv_path = Path(aoi_cfg.path)
    base_dir = Path(aoi_cfg.base_dir)
    id_col = aoi_cfg.id_col
    crs_out = aoi_cfg.crs_out

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

        # Load each sub-AOI individually
        sub_aois: List[Dict] = []
        parts_gdf: List[gpd.GeoDataFrame] = []
        for fname, p in existing:
            try:
                gdf_part = load_aoi(p, crs_out=crs_out,
                                    buffer_meters=aoi_cfg.buffer_meters)
                parts_gdf.append(gdf_part)
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
    aoi_cfg = config.aoi
    aoi = load_aoi(
        aoi_path,
        crs_out=aoi_cfg.crs_out,
        buffer_meters=aoi_cfg.buffer_meters,
        dissolve=True,
    )
    dataset_id = aoi_path.parent.parent.name
    slug = dataset_id.replace("-", "_").replace(" ", "_")

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


# -----------------------------------------------------------------
# Sub-AOI tagging
# -----------------------------------------------------------------

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
        f"Tagged buildings: {len(gdf)} total, {len(joined)} matched sub-AOIs, "
        f"{dropped} dropped (in gaps)"
    )

    return joined


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
            "ref_paths":    list[Path],     # all resolved reference paths
        }
    """
    root = Path(cfg.get("root_dir", "."))
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

        # Load each sub-AOI individually
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

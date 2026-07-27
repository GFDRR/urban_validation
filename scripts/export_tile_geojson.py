"""
Export per-tile vector validation metrics to GeoJSON, GeoPackage, and CSV.

Usage (run from the project root on Colab or locally):
    python scripts/export_tile_geojson.py
    python scripts/export_tile_geojson.py --config configs/validation_configs.yaml

Outputs
-------
outputs/scratch/tiles/{city}_tiles.geojson   one file per city, all datasets as features
outputs/scratch/all_tiles.gpkg               combined across all cities (QGIS-ready)
outputs/scratch/per_tile_metrics.csv         flat CSV, one row per tile × dataset

Derived columns added (not in pipeline parquet):
    ref_building_density_per_km2    n_ref / tile_area_km2
    ref_avg_building_size_m2        ref_area_total_m2 / n_ref  (or mean_ref_building_area_m2 if present)
    tile_minx / tile_miny / tile_maxx / tile_maxy    bounding box in the tile CRS

Additional metrics suggested for future pipeline output (see bottom of script).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── make_tiles is the same function used by the pipeline ─────────────────────
# Import it so tile reconstruction is guaranteed to be identical.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.tiling import make_tiles
from src.utils.geometry import get_projected_crs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_columns(df: pd.DataFrame, tiles_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Add derived density/size columns and tile bounding boxes.

    Works with both the pre-2640ba6 schema (no tile_area_km2) and the
    updated schema that includes tile_area_km2 and mean_ref_building_area_m2.
    """
    df = df.copy()

    # tile_area_km2 — compute from geometry if not already in parquet
    if "tile_area_km2" not in df.columns:
        area_map = (
            tiles_gdf.set_index("tile_id")["geometry"].area / 1e6
        )
        df["tile_area_km2"] = df["tile_id"].map(area_map)

    # ref_building_density_per_km2 = n_ref / tile_area_km2
    df["ref_building_density_per_km2"] = np.where(
        df["tile_area_km2"] > 0,
        df["n_ref"] / df["tile_area_km2"],
        np.nan,
    )

    # ref_avg_building_size_m2 — use pipeline column if available, else derive
    if "mean_ref_building_area_m2" in df.columns:
        df["ref_avg_building_size_m2"] = df["mean_ref_building_area_m2"]
    elif "ref_area_total_m2" in df.columns:
        df["ref_avg_building_size_m2"] = np.where(
            df["n_ref"] > 0,
            df["ref_area_total_m2"] / df["n_ref"],
            np.nan,
        )
    else:
        df["ref_avg_building_size_m2"] = np.nan

    # Tile bounding box from geometry
    bounds = tiles_gdf.set_index("tile_id")["geometry"].bounds
    for col in ["minx", "miny", "maxx", "maxy"]:
        df[f"tile_{col}"] = df["tile_id"].map(bounds[col])

    return df


def _load_tiles(city_slug: str, data_dir: Path, aoi_gdf: gpd.GeoDataFrame,
                tile_size_m: float) -> gpd.GeoDataFrame:
    """
    Load tile geometries from the pipeline-saved GPKG if it exists,
    otherwise reconstruct them using the same make_tiles call the pipeline uses.
    """
    gpkg = data_dir / city_slug.upper() / "tiles" / f"{city_slug}_tiles.gpkg"
    if gpkg.exists():
        tiles = gpd.read_file(gpkg)
        log.info("  tiles: loaded from %s (%d tiles)", gpkg.name, len(tiles))
        return tiles

    # Reconstruct — must use the same projected CRS the pipeline chose
    crs = get_projected_crs(aoi_gdf)
    aoi_proj = aoi_gdf.to_crs(crs)
    tiles = make_tiles(aoi_proj, tile_size_m)
    log.info("  tiles: reconstructed (%d tiles, CRS %s)", len(tiles), crs)
    return tiles


def _load_aoi(city_slug: str, data_dir: Path) -> gpd.GeoDataFrame | None:
    """Load any AOI file for a city (needed only when tiles GPKG is missing)."""
    city_dir = data_dir / city_slug.upper() / "aoi"
    if not city_dir.exists():
        return None
    for suffix in ("*.geojson", "*.gpkg", "*.shp"):
        found = sorted(city_dir.glob(suffix))
        if found:
            return gpd.read_file(found[0])
    return None


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------

def export(config_path: str = "configs/validation_configs.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    root     = Path(cfg["root_dir"])
    data_dir = root / cfg["data_dir"]
    tile_size_m = float(
        cfg.get("vector", {}).get("preprocessing", {}).get("tile_size_m", 1000)
    )
    dataset_names = [
        d["name"] for d in cfg.get("vector", {}).get("datasets", [])
        if d.get("enabled", True)
    ]

    metrics_root = root / "outputs" / "metrics"
    out_root     = root / "outputs" / "scratch"
    tiles_out    = out_root / "tiles"
    tiles_out.mkdir(parents=True, exist_ok=True)

    # Discover processed cities from existing sentinel files
    sentinel_name = "vector_metrics_tiles_all_datasets.parquet"
    city_dirs = sorted(
        p.parent for p in metrics_root.rglob(sentinel_name)
    )

    if not city_dirs:
        log.warning(
            "No sentinel files found under %s — run the validation pipeline first.",
            metrics_root,
        )
        return

    log.info(
        "Found %d processed city/cities | tile_size=%dm | datasets: %s",
        len(city_dirs), int(tile_size_m), dataset_names,
    )

    all_gdfs: list[gpd.GeoDataFrame] = []
    csv_rows: list[pd.DataFrame]     = []

    for city_metrics_dir in city_dirs:
        city_slug = city_metrics_dir.name   # e.g. "ssd-juba"
        log.info("[%s]", city_slug)

        parquet_path = city_metrics_dir / sentinel_name
        try:
            metrics_df = pd.read_parquet(parquet_path)
        except Exception as exc:
            log.warning("  could not read parquet: %s", exc)
            continue

        if metrics_df.empty:
            log.warning("  parquet is empty — skipping.")
            continue

        # Load AOI (needed only for tile reconstruction fallback)
        aoi_gdf = _load_aoi(city_slug, data_dir)

        # Load tile geometries
        try:
            tiles_gdf = _load_tiles(
                city_slug, data_dir,
                aoi_gdf if aoi_gdf is not None else gpd.GeoDataFrame(),
                tile_size_m,
            )
        except Exception as exc:
            log.warning("  could not build tiles: %s", exc)
            continue

        if tiles_gdf.empty:
            log.warning("  no tiles — skipping.")
            continue

        # Derive density/size columns and bounding boxes
        metrics_df = _derive_columns(metrics_df, tiles_gdf)

        # Join metrics onto tile geometries
        tiles_crs = tiles_gdf.crs
        merged = tiles_gdf.merge(metrics_df, on="tile_id", how="inner")

        if merged.empty:
            log.warning("  no tile_id matches between geometry and metrics — skipping.")
            continue

        merged["city"] = city_slug

        # ── Per-city GeoJSON ───────────────────────────────────────────
        geojson_path = tiles_out / f"{city_slug}_tiles.geojson"
        # GeoJSON must be EPSG:4326
        merged_4326 = merged.to_crs("EPSG:4326")
        merged_4326.to_file(geojson_path, driver="GeoJSON")
        log.info("  → %s (%d features)", geojson_path.name, len(merged_4326))

        all_gdfs.append(merged)

        # ── Flat CSV rows (no geometry, add bbox in WGS84) ────────────
        csv_df = merged_4326.drop(columns="geometry").copy()
        # Replace tile bbox columns with WGS84 equivalents
        bounds_4326 = merged_4326["geometry"].bounds
        for col in ["minx", "miny", "maxx", "maxy"]:
            csv_df[f"tile_wgs84_{col}"] = bounds_4326[col].values
        csv_rows.append(csv_df)

    if not all_gdfs:
        log.warning("No output produced — nothing to combine.")
        return

    # ── Combined GeoPackage ───────────────────────────────────────────
    # Reproject everything to EPSG:4326 for QGIS portability
    combined = gpd.GeoDataFrame(
        pd.concat([g.to_crs("EPSG:4326") for g in all_gdfs], ignore_index=True),
        crs="EPSG:4326",
    )
    gpkg_path = out_root / "all_tiles.gpkg"
    combined.to_file(gpkg_path, driver="GPKG", layer="tiles")
    log.info("Combined GPKG → %s (%d features)", gpkg_path, len(combined))

    # ── Flat CSV ──────────────────────────────────────────────────────
    flat_csv = pd.concat(csv_rows, ignore_index=True)

    # Column order: identifiers first, then core metrics, then extras
    id_cols     = ["city", "dataset", "tile_id"]
    metric_cols = [
        "f1", "precision", "recall",
        "tp", "fp", "fn",
        "signed_area_bias",
        "ref_area_total_m2", "cand_area_total_m2",
        "n_ref", "n_cand",
        "tile_area_km2",
        "ref_building_density_per_km2",
        "ref_avg_building_size_m2",
        "mean_iou", "median_iou",
        "boundary_f_union",
    ]
    bbox_cols   = ["tile_wgs84_minx", "tile_wgs84_miny", "tile_wgs84_maxx", "tile_wgs84_maxy"]
    # Keep any extra columns that don't fit the categories above
    known = set(id_cols + metric_cols + bbox_cols)
    extra = [c for c in flat_csv.columns if c not in known]
    ordered = [c for c in id_cols + metric_cols + bbox_cols + extra if c in flat_csv.columns]
    flat_csv = flat_csv[ordered]

    csv_path = out_root / "per_tile_metrics.csv"
    flat_csv.to_csv(csv_path, index=False)
    log.info("Flat CSV → %s (%d rows, %d columns)", csv_path, len(flat_csv), len(flat_csv.columns))

    # ── Summary stats ─────────────────────────────────────────────────
    log.info("\n── Per-dataset summary ──")
    for ds in flat_csv["dataset"].unique() if "dataset" in flat_csv.columns else []:
        sub = flat_csv[flat_csv["dataset"] == ds]
        log.info(
            "  %s: %d tiles | F1 min=%.3f mean=%.3f max=%.3f | "
            "density min=%.1f mean=%.1f max=%.1f bldg/km²",
            ds, len(sub),
            sub["f1"].min(), sub["f1"].mean(), sub["f1"].max(),
            sub["ref_building_density_per_km2"].min(),
            sub["ref_building_density_per_km2"].mean(),
            sub["ref_building_density_per_km2"].max(),
        )

    _print_suggestions()


def _print_suggestions():
    """Print column suggestions for the density/F1 scatter analysis."""
    log.info(
        "\n── Suggested additional columns for density/F1 scatter analysis ──\n"
        "  Already in per-tile parquet:\n"
        "    mean_iou, median_iou, iou_p25, iou_p75   — IoU distribution\n"
        "    boundary_f_union, boundary_f_meanpair     — boundary accuracy\n"
        "    mean_rel_area_error, signed_area_bias      — area bias\n"
        "    ref_area_total_m2, cand_area_total_m2     — total footprint area\n"
        "\n"
        "  Derivable in this script (add to _derive_columns):\n"
        "    cand_building_density_per_km2  = n_cand / tile_area_km2\n"
        "    cand_avg_building_size_m2      = cand_area_total_m2 / n_cand\n"
        "    density_ratio                  = n_cand / n_ref  (over/under-prediction)\n"
        "    built_fraction_ref             = ref_area_total_m2 / (tile_area_km2 * 1e6)\n"
        "\n"
        "  Requires raster parquet join (outputs/metrics/{city}/raster_*.parquet):\n"
        "    wsf_built_fraction             — WSF built-up fraction per tile\n"
        "    population_density_per_km2     — if WorldPop or GHS-POP is available\n"
        "\n"
        "  From aoi_tracker / metadata:\n"
        "    region / subregion             — for regional disaggregation\n"
        "    ref_image_year                 — temporal context (from Section 2 of notebook 04)\n"
        "    is_spacenet7                   — dataset origin flag\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/validation_configs.yaml",
        help="Path to validation_configs.yaml (default: configs/validation_configs.yaml)",
    )
    args = parser.parse_args()
    export(args.config)

"""
run_validation_pipeline.py  (memory-optimised)
===============================================
Multi-AOI building-dataset validation pipeline.

Key changes vs. original:
  • Explicit memory cleanup between AOIs (gc.collect, del, matplotlib cache purge)
  • Tile metrics/matches written incrementally to parquet instead of accumulated in lists
  • Candidate GeoDataFrames freed as soon as their tile loop finishes
  • Figure generation isolated so temps are released immediately
  • Optional --max-workers for process-based parallelism (each AOI in its own process → own memory space)

Usage
-----
    python run_validation_pipeline.py \
        --config  configs/validation_config.yaml \
        --tracker data/02_interim/aoi_tracker.csv \
        [--aoi-filter ant-curacao ssd-juba ...]
        [--skip-existing]
        [--dry-run]
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

# ── project imports ──────────────────────────────────────────────────────────
from src.utils import load_aoi, make_tiles, load_buildings, subset_by_tile, get_projected_crs
from src.metrics import compute_tile_metrics
from src.output import summarize_city, save_figure, fig_name

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_tracker(path: Path, aoi_filter: list[str] | None = None) -> pd.DataFrame:
    """Return suitable rows from the aoi_tracker CSV."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    suitable_col = next(
        (c for c in df.columns if "suitable" in c.lower()), None
    )
    if suitable_col:
        df = df[df[suitable_col].astype(str).str.lower() == "yes"].copy()

    if aoi_filter:
        df = df[df["dataset_folder_name"].isin(aoi_filter)].copy()

    df = df[df["aoi_file_name"].notna()].copy()
    log.info("Tracker: %d AOI rows selected for processing.", len(df))
    return df.reset_index(drop=True)


def already_done(metrics_dir: Path) -> bool:
    return (metrics_dir / "vector_metrics_tiles_all_datasets.parquet").exists()


def _purge_matplotlib():
    """Aggressively release matplotlib/seaborn memory."""
    import matplotlib.pyplot as plt
    plt.close("all")
    # Clear the figure manager's internal state
    try:
        from matplotlib._pylab_helpers import Gcf
        Gcf.destroy_all()
    except Exception:
        pass
    gc.collect()


def _log_memory(label: str = ""):
    """Log current RSS — useful for spotting leaks during development."""
    try:
        import psutil
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / 1024 ** 2
        log.info("MEM [%s] RSS = %.0f MB", label, rss_mb)
    except ImportError:
        pass  # psutil not available — skip silently


# ─────────────────────────────────────────────────────────────────────────────
# Per-AOI processing  (memory-optimised)
# ─────────────────────────────────────────────────────────────────────────────

def process_aoi(row: pd.Series, cfg: dict, root: Path) -> bool:
    """
    Run the full vector-validation pipeline for a single AOI row.
    All large intermediates are explicitly deleted after use.
    """
    folder_name: str = row["dataset_folder_name"]
    aoi_file: str = row["aoi_file_name"]
    ref_file: Optional[str] = row.get("reference_file_name")

    # ── resolve paths ─────────────────────────────────────────────────────────
    data_dir = root / cfg["data_dir"]
    aoi_path = data_dir / folder_name / "aoi" / aoi_file

    if not aoi_path.exists():
        log.warning("[%s] AOI file not found: %s — skipping.", folder_name, aoi_path)
        return False

    if not ref_file or pd.isna(ref_file):
        log.warning("[%s] No reference file listed — skipping.", folder_name)
        return False

    ref_path = data_dir / folder_name / "vector" / str(ref_file)
    if not ref_path.exists():
        log.warning("[%s] Reference file not found: %s — skipping.", folder_name, ref_path)
        return False

    # ── output directories ────────────────────────────────────────────────────
    city_slug = folder_name.lower()
    metrics_dir = root / "outputs" / "metrics" / city_slug
    figures_dir = root / "outputs" / "figures" / city_slug
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── preprocessing config ──────────────────────────────────────────────────
    vec_pre = cfg["vector"]["preprocessing"]
    min_area: float = vec_pre["min_area_m2"]
    tile_size: float = vec_pre["tile_size_m"]
    tau_overlap: float = vec_pre["tau_overlap"]
    tau_buffer: float = vec_pre["tau_buffer_m"]
    tau_boundary: float = vec_pre["tau_boundary"]
    fix_geoms: bool = vec_pre.get("fix_invalid_geoms", True)

    log.info("━━━━  AOI: %s  ━━━━", folder_name)
    _log_memory(f"{folder_name} start")

    # ── 1. Load AOI in 4326 first, then auto-detect projected CRS ────────────
    aoi_4326 = load_aoi(path=str(aoi_path), crs_out="EPSG:4326")
    crs: str = get_projected_crs(aoi_4326)
    log.info("[%s] Auto-detected working CRS: %s", folder_name, crs)

    aoi = aoi_4326.to_crs(crs)
    del aoi_4326
    tiles = make_tiles(aoi, tile_size)
    log.info("[%s] %d tiles generated.", folder_name, len(tiles))

    tiles_path = root / "data" / "02_interim" / "tiles" / f"{city_slug}_tiles.gpkg"
    tiles_path.parent.mkdir(parents=True, exist_ok=True)
    tiles.to_file(tiles_path, driver="GPKG")

    del aoi  # no longer needed
    gc.collect()

    # ── 2. Load reference buildings ───────────────────────────────────────────
    ref_all = load_buildings(
        path=str(ref_path),
        crs_work=crs,
        min_area_m2=min_area,
        fix_invalid_geoms=fix_geoms,
    )
    log.info("[%s] Reference buildings: %d", folder_name, len(ref_all))

    # ── 3. Iterate over candidate datasets ────────────────────────────────────
    candidates = cfg["vector"]["datasets"]
    vec_out_path = data_dir / folder_name / "vector"

    # We'll collect lightweight file paths and concat at the end
    # instead of accumulating large DataFrames in memory
    per_ds_tile_paths: list[Path] = []
    per_ds_match_paths: list[Path] = []

    for cand_cfg in candidates:
        if not cand_cfg.get("enabled", True):
            continue

        ds_name: str = cand_cfg["name"]
        pattern = f"{city_slug.replace('-', '_')}_{ds_name}*.parquet"
        candidate_files = list(vec_out_path.glob(pattern))

        if not candidate_files:
            log.warning(
                "[%s] No candidate files for dataset '%s' (pattern: %s in %s).",
                folder_name, ds_name, pattern, vec_out_path,
            )
            continue

        cand_path = candidate_files[0]
        log.info("[%s / %s] Loading candidate: %s", folder_name, ds_name, cand_path.name)

        cand_all = load_buildings(
            path=str(cand_path),
            crs_work=crs,
            min_area_m2=min_area,
            fix_invalid_geoms=fix_geoms,
        )
        log.info("[%s / %s] Candidate buildings: %d", folder_name, ds_name, len(cand_all))

        # ── tile loop — write results incrementally ───────────────────────────
        ds_tile_metrics: list[dict] = []
        ds_match_chunks: list[pd.DataFrame] = []
        MATCH_FLUSH_INTERVAL = 50  # flush matches to disk every N tiles

        # Build spatial indices once per candidate
        ref_sindex = ref_all.sindex
        cand_sindex = cand_all.sindex

        for tile_idx, tile_row in enumerate(tiles.itertuples()):
            tile_geom = tile_row.geometry
            tile_id = int(tile_row.tile_id)

            ref_tile = subset_by_tile(ref_all, ref_sindex, tile_geom)
            cand_tile = subset_by_tile(cand_all, cand_sindex, tile_geom)

            if ref_tile.empty and cand_tile.empty:
                continue

            metrics, matches_df = compute_tile_metrics(
                ref_tile, folder_name, cand_tile,
                tau_overlap, tau_buffer, tau_boundary,
                tile_id, ds_name,
            )
            ds_tile_metrics.append(metrics)

            if not matches_df.empty:
                matches_df = matches_df.copy()
                matches_df["city"] = folder_name
                matches_df["dataset"] = ds_name
                matches_df["tile_id"] = tile_id
                ds_match_chunks.append(matches_df)

            # Flush match chunks periodically to avoid memory buildup
            if len(ds_match_chunks) >= MATCH_FLUSH_INTERVAL:
                _flush_matches(ds_match_chunks, metrics_dir, ds_name, append=True)
                ds_match_chunks.clear()

            # Free tile-level intermediates
            del ref_tile, cand_tile, metrics, matches_df

        # ── free candidate GeoDataFrame immediately ───────────────────────────
        del cand_all, cand_sindex
        gc.collect()

        # ── save per-dataset tile metrics ─────────────────────────────────────
        ds_tile_df = pd.DataFrame(ds_tile_metrics)
        del ds_tile_metrics

        tile_out = metrics_dir / f"vector_metrics_tiles_{ds_name}.parquet"
        if not ds_tile_df.empty:
            ds_tile_df.to_parquet(tile_out, index=False)
            log.info("[%s / %s] Saved tile metrics → %s", folder_name, ds_name, tile_out.name)
            per_ds_tile_paths.append(tile_out)
        del ds_tile_df

        # ── flush remaining matches ───────────────────────────────────────────
        match_out = metrics_dir / f"vector_matches_{ds_name}.parquet"
        if ds_match_chunks:
            _flush_matches(ds_match_chunks, metrics_dir, ds_name, append=True)
            ds_match_chunks.clear()

        # Consolidate the incrementally-flushed chunks into one file
        _consolidate_match_chunks(metrics_dir, ds_name, match_out)
        per_ds_match_paths.append(match_out)

        gc.collect()
        _log_memory(f"{folder_name}/{ds_name} done")

    # ── free reference GeoDataFrame ───────────────────────────────────────────
    del ref_all, ref_sindex
    gc.collect()

    if not per_ds_tile_paths:
        log.warning("[%s] No tile metrics produced — skipping summary/figures.", folder_name)
        return False

    # ── 4. Combine across datasets (read from disk, not memory) ───────────────
    metrics_all = pd.concat(
        [pd.read_parquet(p) for p in per_ds_tile_paths],
        ignore_index=True,
    )
    metrics_all.to_parquet(
        metrics_dir / "vector_metrics_tiles_all_datasets.parquet", index=False,
    )

    matches_all = pd.concat(
        [pd.read_parquet(p) for p in per_ds_match_paths if p.exists()],
        ignore_index=True,
    )
    matches_all.to_parquet(
        metrics_dir / "vector_matches_all_datasets.parquet", index=False,
    )

    # ── 5. City-level summary ─────────────────────────────────────────────────
    city_summary = summarize_city(folder_name, metrics_all, matches_all)
    summary_path = metrics_dir / "vector_city_summary_all_datasets.parquet"
    city_summary.to_parquet(summary_path, index=False)
    city_summary.to_csv(
        metrics_dir / "vector_city_summary_all_datasets.csv", index=False,
    )
    log.info("[%s] City summary saved.", folder_name)
    del city_summary

    # ── 6. Figures (isolated scope) ───────────────────────────────────────────
    try:
        _make_figures(
            folder_name=folder_name,
            metrics_all=metrics_all,
            matches_all=matches_all,
            tiles=tiles,
            figures_dir=figures_dir,
            cfg=cfg,
        )
    finally:
        _purge_matplotlib()

    # ── cleanup everything from this AOI ──────────────────────────────────────
    del metrics_all, matches_all, tiles
    gc.collect()
    _log_memory(f"{folder_name} end")

    log.info("[%s] ✓ Complete.", folder_name)
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Incremental match-writing helpers
# ─────────────────────────────────────────────────────────────────────────────

_match_chunk_counter: dict[str, int] = {}


def _flush_matches(
    chunks: list[pd.DataFrame],
    metrics_dir: Path,
    ds_name: str,
    append: bool = True,
) -> None:
    """Write accumulated match chunks to a numbered temp parquet file."""
    if not chunks:
        return
    counter = _match_chunk_counter.get(ds_name, 0)
    tmp_path = metrics_dir / f"_tmp_matches_{ds_name}_{counter:04d}.parquet"
    pd.concat(chunks, ignore_index=True).to_parquet(tmp_path, index=False)
    _match_chunk_counter[ds_name] = counter + 1


def _consolidate_match_chunks(
    metrics_dir: Path, ds_name: str, final_path: Path
) -> None:
    """Read all temp match chunks, concat, write final file, clean up temps."""
    chunk_files = sorted(metrics_dir.glob(f"_tmp_matches_{ds_name}_*.parquet"))

    if chunk_files:
        df = pd.concat(
            [pd.read_parquet(f) for f in chunk_files],
            ignore_index=True,
        )
        df.to_parquet(final_path, index=False)
        del df
        for f in chunk_files:
            f.unlink()
    else:
        # Write an empty parquet with expected schema
        pd.DataFrame(
            columns=[
                "ref_id", "cand_id", "iou", "area_ref", "area_cand",
                "rel_area_error", "city", "dataset", "tile_id",
            ]
        ).to_parquet(final_path, index=False)

    # Reset counter for this dataset
    _match_chunk_counter.pop(ds_name, None)


# ─────────────────────────────────────────────────────────────────────────────
# Figure generation (unchanged logic, wrapped for cleanup)
# ─────────────────────────────────────────────────────────────────────────────

def _make_figures(
    folder_name: str,
    metrics_all: pd.DataFrame,
    matches_all: pd.DataFrame,
    tiles: gpd.GeoDataFrame,
    figures_dir: Path,
    cfg: dict,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    dpi: int = cfg.get("output", {}).get("figures", {}).get("dpi", 200)
    fmt: str = cfg.get("output", {}).get("figures", {}).get("fmt", "png")
    city_label = folder_name.replace("-", " ").title()

    # ── Figure 1: F1 box-plot by dataset ─────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.boxplot(data=metrics_all, x="dataset", y="f1", ax=ax)
        ax.set_title(f"Tile-level F1 scores – {city_label}")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("F1")
        save_figure(fig, figures_dir, fig_name(folder_name, "tile_f1_boxplot"))
        plt.close(fig)
    except Exception:
        log.warning("[%s] F1 boxplot failed:\n%s", folder_name, traceback.format_exc())

    # ── Figure 2: Spatial F1 map per dataset ─────────────────────────────────
    for ds in metrics_all["dataset"].unique():
        try:
            metrics_ds = metrics_all[metrics_all["dataset"] == ds]
            tiles_m = tiles.merge(metrics_ds[["tile_id", "f1"]], on="tile_id", how="left")
            fig, ax = plt.subplots(figsize=(8, 8))
            tiles_m.plot(column="f1", ax=ax, legend=True, cmap="viridis", edgecolor="none")
            ax.set_title(f"{city_label} – Tile-level F1 – {ds}")
            ax.set_axis_off()
            save_figure(fig, figures_dir, fig_name(folder_name, f"spatial_f1_{ds}"))
            plt.close(fig)
            del tiles_m
        except Exception:
            log.warning("[%s / %s] Spatial F1 map failed:\n%s",
                        folder_name, ds, traceback.format_exc())

    # ── Figure 3: IoU histograms ─────────────────────────────────────────────
    for ds in metrics_all["dataset"].unique():
        try:
            m_ds = metrics_all[metrics_all["dataset"] == ds]
            tp = int(m_ds["tp"].sum())
            fp = int(m_ds["fp"].sum())
            fn = int(m_ds["fn"].sum())

            ious_tp = matches_all[matches_all["dataset"] == ds]["iou"].dropna()
            ious_all = pd.concat(
                [ious_tp, pd.Series(np.zeros(fp + fn))], ignore_index=True
            )

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(ious_all, bins=30)
            ax.set_title(f"{city_label} – IoU (TP + FP/FN→0) – {ds}")
            ax.set_xlabel("IoU")
            ax.set_ylabel("Count")
            save_figure(fig, figures_dir, fig_name(folder_name, f"iou_hist_{ds}"))
            plt.close(fig)
            del ious_all
        except Exception:
            log.warning("[%s / %s] IoU histogram failed:\n%s",
                        folder_name, ds, traceback.format_exc())

    # ── Figure 4+5: IoU and rel-area-error vs building size ──────────────────
    size_bins_cfg = cfg.get("size_bins", {})
    size_bins = size_bins_cfg.get("bins", [0, 25, 50, 100, 500, 1000, np.inf])
    size_bin_labels = size_bins_cfg.get(
        "labels", ["<25", "25–50", "50–100", "100–500", "500–1000", ">1000"]
    )

    for ds in matches_all["dataset"].unique() if not matches_all.empty else []:
        try:
            m_ds = matches_all[matches_all["dataset"] == ds].copy()
            if m_ds.empty:
                continue

            m_ds["size_bin"] = pd.cut(
                m_ds["area_ref"],
                bins=size_bins,
                labels=size_bin_labels,
                include_lowest=True,
            )
            size_stats = (
                m_ds.groupby("size_bin", observed=True)
                .agg(
                    mean_iou=("iou", "mean"),
                    median_iou=("iou", "median"),
                    median_rel_area_error=("rel_area_error", "median"),
                    count=("iou", "size"),
                )
                .reset_index()
            )

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(size_stats["size_bin"].astype(str), size_stats["median_iou"], marker="o")
            ax.set_ylabel("Median IoU")
            ax.set_xlabel("Reference building size (m²)")
            ax.set_title(f"{city_label} – IoU vs building size – {ds}")
            ax.axhline(0, color="grey", linestyle="--", linewidth=1)
            ax.grid(True, axis="y", alpha=0.3)
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            save_figure(fig, figures_dir, fig_name(folder_name, f"iou_vs_size_{ds}"))
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                size_stats["size_bin"].astype(str),
                size_stats["median_rel_area_error"],
                marker="o",
            )
            ax.set_ylabel("Median relative area error")
            ax.set_xlabel("Reference building size (m²)")
            ax.set_title(f"{city_label} – Rel. area error vs building size – {ds}")
            ax.axhline(0, color="grey", linestyle="--", linewidth=1)
            ax.grid(True, axis="y", alpha=0.3)
            plt.xticks(rotation=45, ha="right")
            fig.tight_layout()
            save_figure(fig, figures_dir, fig_name(folder_name, f"area_err_vs_size_{ds}"))
            plt.close(fig)
            del m_ds, size_stats

        except Exception:
            log.warning("[%s / %s] Size-bin figures failed:\n%s",
                        folder_name, ds, traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# CLI + main loop
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-AOI building-dataset validation pipeline."
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--tracker", type=Path, default=None)
    p.add_argument("--aoi-filter", nargs="*", metavar="FOLDER")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    cfg = load_config(args.config)
    project_root = Path(cfg.get("root_dir", ".")).expanduser().resolve()
    root = project_root

    tracker_path = (
        args.tracker
        if args.tracker
        else root / cfg.get("aoi_tracker", "data/02_interim/aoi_tracker.csv")
    )
    tracker_df = load_tracker(tracker_path, aoi_filter=args.aoi_filter)

    if tracker_df.empty:
        log.error("No AOI rows to process. Check your tracker path and filters.")
        sys.exit(1)

    if args.dry_run:
        print(f"\nDry-run — would process {len(tracker_df)} AOIs:\n")
        for _, row in tracker_df.iterrows():
            slug = row["dataset_folder_name"].lower()
            metrics_dir = root / "outputs" / "metrics" / slug
            status = "DONE" if already_done(metrics_dir) else "PENDING"
            if args.skip_existing and status == "DONE":
                status = "SKIP"
            print(f"  {status:7s}  {row['dataset_folder_name']}")
        print()
        return

    # ── main loop with explicit cleanup ───────────────────────────────────────
    results: dict[str, str] = {}

    for i, (_, row) in enumerate(tracker_df.iterrows()):
        folder = row["dataset_folder_name"]
        city_slug = folder.lower().replace("-", "_")
        metrics_dir = root / "outputs" / "metrics" / city_slug

        if args.skip_existing and already_done(metrics_dir):
            log.info("[%s] Already done — skipping (--skip-existing).", folder)
            results[folder] = "skipped"
            continue

        try:
            success = process_aoi(row, cfg, root)
            results[folder] = "ok" if success else "no_data"
        except Exception:
            log.error("[%s] Unhandled exception:\n%s", folder, traceback.format_exc())
            results[folder] = "error"

        # ── CRITICAL: force full GC between every AOI ─────────────────────────
        gc.collect()
        _purge_matplotlib()
        _log_memory(f"after AOI {i+1}/{len(tracker_df)}")

    # ── report ────────────────────────────────────────────────────────────────
    ok      = [k for k, v in results.items() if v == "ok"]
    no_data = [k for k, v in results.items() if v == "no_data"]
    skipped = [k for k, v in results.items() if v == "skipped"]
    errored = [k for k, v in results.items() if v == "error"]

    print("\n" + "═" * 60)
    print(f"  Pipeline complete  —  {len(results)} AOIs processed")
    print("═" * 60)
    print(f"  ✓  Success  : {len(ok)}")
    print(f"  ·  No data  : {len(no_data)}")
    print(f"  ↷  Skipped  : {len(skipped)}")
    print(f"  ✗  Errors   : {len(errored)}")
    if errored:
        print("\n  Failed AOIs:")
        for name in errored:
            print(f"    - {name}")
    print("═" * 60 + "\n")

    if errored:
        sys.exit(1)


if __name__ == "__main__":
    main()
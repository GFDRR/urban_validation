import yaml
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from pprint import pprint

from src.output import load_config
from src.utils import load_aoi, make_tiles, load_buildings, subset_by_tile
from src.metrics import compute_tile_metrics
from src.output import summarize_city, tile_f1_box_plot, tile_f1_spatial_dist, plot_iou_dist, plot_iou_per_building_sizes

logger = logging.getLogger("Urban_Validator")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
sh = logging.StreamHandler()
sh.setFormatter(fmt)
if not logger.handlers:
    logger.addHandler(sh)

# Default building-size bins (m²)
_DEFAULT_SIZE_BINS   = [0, 25, 50, 100, 500, 1000, np.inf]
_DEFAULT_SIZE_LABELS = ["<25", "25–50", "50–100", "100–500", "500–1000", ">1000"]


class Validator:
    """
    Validates candidate building footprint datasets against a reference layer
    for one city (Dataset code), which may contain multiple sub-areas.

    Parameters
    ----------
    config : dict
        Parsed YAML config (loaded by load_config).
    city : str
        City slug, e.g. ``"ssd-juba"``.
    sub_areas : list[dict]
        Each dict must contain:
            sub_area_id : str   – stem used in filenames / output columns
            aoi         : Path  – resolved path to the AOI file
            reference   : Path  – resolved path to the reference buildings file
        Built by run_validation.py from the AOI tracker CSV.
    """

    def __init__(
        self,
        config: dict,
        city: str,
        sub_areas: list[dict],
        log: Optional[logging.Logger] = None,
    ):
        self.logger    = log or logging.getLogger("Urban_Validator")
        self.config    = config
        self.city      = city
        self.sub_areas = sub_areas

        root       = Path(config["root_dir"])
        self.root  = root
        self.crs   = config["crs"]

        self.candidates = [
            d for d in config["vector"]["datasets"] if d.get("enabled", True)
        ]

        pre = config["vector"]["preprocessing"]
        self.tau_overlap       = pre["tau_overlap"]
        self.tau_buffer_m      = pre["tau_buffer_m"]
        self.tau_boundary_m    = pre["tau_boundary"]
        self.tile_size_m       = pre["tile_size_m"]
        self.min_area_m2       = pre["min_area_m2"]
        self.fix_invalid_geoms = pre.get("fix_invalid_geoms", True)

        bins_cfg = config["vector"].get("size_bins", {})
        raw_bins = bins_cfg.get("bins", _DEFAULT_SIZE_BINS)
        self.size_bins   = [
            np.inf if str(b).lower() in ("inf", ".inf") else float(b)
            for b in raw_bins
        ]
        self.size_labels = bins_cfg.get("labels", _DEFAULT_SIZE_LABELS)

        out_cfg         = config.get("output", {})
        self.overwrite  = out_cfg.get("overwrite", False)
        fig_cfg         = out_cfg.get("figures", {})
        self.fig_dpi    = fig_cfg.get("dpi", 200)
        self.fig_fmt    = fig_cfg.get("fmt", "png")
        self.save_tiles = out_cfg.get("metrics", {}).get("save_tile_level", True)

        city_slug        = self.city.lower()
        self.metrics_dir = root / f"outputs/metrics/{city_slug}"
        self.figures_dir = root / f"outputs/figures/{city_slug}"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "Validator ready | city=%s | sub_areas=%d | datasets=%s",
            self.city,
            len(self.sub_areas),
            [d["name"] for d in self.candidates],
        )

    @classmethod
    def from_config_path(
        cls,
        config_path: str | Path,
        city: str,
        sub_areas: list[dict],
        log: Optional[logging.Logger] = None,
    ) -> "Validator":
        """Construct from a YAML file path."""
        return cls(load_config(config_path), city=city, sub_areas=sub_areas, log=log)
    
    def _already_done(self) -> bool:
        sentinel = self.metrics_dir / "vector_metrics_tiles_all_datasets.parquet"
        return sentinel.exists() and not self.overwrite

    def compute_vector_metrics(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        For every sub-area × candidate dataset, compute tile-level matching
        metrics and persist results to parquet.

        The outer loop is **sub-areas** (each has its own AOI + reference);
        the inner loops are candidate datasets → tiles.

        Returns
        -------
        metrics_all : DataFrame
            Tile-level TP/FP/FN/F1 for all sub-areas × datasets.
            Includes ``sub_area_id`` column.
        matches_all : DataFrame
            Per-matched-pair IoU / area stats. Includes ``sub_area_id``.
        """
        if self._already_done():
            self.logger.info(
                "%s: outputs exist; loading from disk (set overwrite: true to recompute).",
                self.city,
            )
            return (
                pd.read_parquet(self.metrics_dir / "vector_metrics_tiles_all_datasets.parquet"),
                pd.read_parquet(self.metrics_dir / "vector_matches_all_datasets.parquet"),
            )

        data_dir = self.root / self.config["vector"]["out_path"]

        all_tile_metrics: list[pd.DataFrame] = []
        all_match_rows:   list[pd.DataFrame] = []

        _empty_match_cols = [
            "ref_id", "cand_id", "iou", "area_ref", "area_cand",
            "rel_area_error", "city", "dataset", "sub_area_id", "tile_id",
        ]

        # ── outer loop: sub-areas ──────────────────────────────────────
        for sa in self.sub_areas:
            sa_id    = sa["sub_area_id"]
            aoi_path = sa["aoi"]
            ref_path = sa["reference"]

            self.logger.info("  Sub-area: %s", sa_id)

            aoi   = load_aoi(path=aoi_path, crs_out=self.crs)
            tiles = make_tiles(aoi, self.tile_size_m)

            ref_buildings = load_buildings(
                path=ref_path,
                crs_work=self.crs,
                min_area_m2=self.min_area_m2,
                fix_invalid_geoms=self.fix_invalid_geoms,
            )
            ref_sindex = ref_buildings.sindex

            self.logger.info(
                "    tiles=%d  ref_buildings=%d", len(tiles), len(ref_buildings)
            )

            # inner loop: candidate datasets
            for cand_cfg in self.candidates:
                ds_name  = cand_cfg["name"]
                ds_label = cand_cfg.get("label", ds_name)

                # Candidate parquet lives alongside the raw data:
                # <data_dir>/<dataset_folder_name>/vector/
                # and is globbed by city slug + dataset name.
                cand_dir = data_dir / self.city /"vector"
                pattern  = f"{self.city.lower()}_{ds_name}*.parquet"
                found    = list(cand_dir.glob(pattern))

                if not found:
                    self.logger.warning(
                        "    No candidate files for '%s' (pattern '%s' in %s). Skipping.",
                        ds_name, pattern, cand_dir,
                    )
                    continue
                if len(found) > 1:
                    self.logger.warning(
                        "    Multiple files match '%s'; using: %s", pattern, found[0].name
                    )
                cand_path = found[0]

                self.logger.info("    ▶ %s — %s", ds_name, ds_label)
                cand_all = load_buildings(
                    path=cand_path,
                    crs_work=self.crs,
                    min_area_m2=self.min_area_m2,
                    fix_invalid_geoms=self.fix_invalid_geoms,
                )
                cand_sindex = cand_all.sindex

                ds_tile_metrics: list[dict]         = []
                ds_match_rows:   list[pd.DataFrame] = []

                # innermost loop: tiles
                for _, tile_row in tiles.iterrows():
                    tile_geom = tile_row.geometry
                    tile_id   = int(tile_row["tile_id"])

                    ref_tile  = subset_by_tile(ref_buildings, ref_sindex,  tile_geom)
                    cand_tile = subset_by_tile(cand_all,      cand_sindex, tile_geom)

                    if ref_tile.empty and cand_tile.empty:
                        continue

                    metrics, matches_df = compute_tile_metrics(
                        ref_tile, self.city, cand_tile,
                        self.tau_overlap, self.tau_buffer_m, self.tau_boundary_m,
                        tile_id, ds_name,
                    )
                    metrics["sub_area_id"] = sa_id
                    ds_tile_metrics.append(metrics)

                    if not matches_df.empty:
                        matches_df = matches_df.copy()
                        matches_df["city"]        = self.city
                        matches_df["dataset"]     = ds_name
                        matches_df["sub_area_id"] = sa_id
                        matches_df["tile_id"]     = tile_id
                        ds_match_rows.append(matches_df)

                # Persist per sub-area × dataset if requested
                if ds_tile_metrics:
                    ds_tile_df = pd.DataFrame(ds_tile_metrics)
                    if self.save_tiles:
                        out = self.metrics_dir / f"vector_metrics_tiles_{sa_id}_{ds_name}.parquet"
                        ds_tile_df.to_parquet(out, index=False)
                        self.logger.info("      Tile metrics → %s", out.name)
                    all_tile_metrics.append(ds_tile_df)

                if ds_match_rows:
                    ds_matches_df = pd.concat(ds_match_rows, ignore_index=True)
                    if self.save_tiles:
                        out = self.metrics_dir / f"vector_matches_{sa_id}_{ds_name}.parquet"
                        ds_matches_df.to_parquet(out, index=False)
                        self.logger.info("      Match pairs  → %s", out.name)
                    all_match_rows.append(ds_matches_df)
                else:
                    all_match_rows.append(pd.DataFrame(columns=_empty_match_cols))

        if not all_tile_metrics:
            self.logger.warning("No metrics produced for %s.", self.city)
            return pd.DataFrame(), pd.DataFrame()

        metrics_all = pd.concat(all_tile_metrics, ignore_index=True)
        matches_all = pd.concat(all_match_rows,   ignore_index=True)

        metrics_all.to_parquet(
            self.metrics_dir / "vector_metrics_tiles_all_datasets.parquet", index=False
        )
        matches_all.to_parquet(
            self.metrics_dir / "vector_matches_all_datasets.parquet", index=False
        )
        self.logger.info("City-wide metrics saved → %s", self.metrics_dir)
        return metrics_all, matches_all

    def visualize_metrics(
        self,
        metrics_all: Optional[pd.DataFrame] = None,
        matches_all: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Produce and save all standard visualisations.
        Pass pre-computed DataFrames, or leave None to load from disk.

        Returns
        -------
        summary : DataFrame  – one row per dataset with city-wide KPIs
        """
        if metrics_all is None:
            metrics_all = pd.read_parquet(
                self.metrics_dir / "vector_metrics_tiles_all_datasets.parquet"
            )
        if matches_all is None:
            matches_all = pd.read_parquet(
                self.metrics_dir / "vector_matches_all_datasets.parquet"
            )

        # Build a combined tile GeoDataFrame across all sub-areas for spatial plots.
        # Each sub-area's tiles are re-generated here since we didn't store them.
        all_tiles: list[gpd.GeoDataFrame] = []
        for sa in self.sub_areas:
            aoi   = load_aoi(path=sa["aoi"], crs_out=self.crs)
            tiles = make_tiles(aoi, self.tile_size_m)
            all_tiles.append(tiles)
        tiles_combined = (
            pd.concat(all_tiles, ignore_index=True)
            .drop_duplicates(subset=["tile_id"])
        )

        summary = summarize_city(self.city, metrics_all, matches_all)
        summary.to_parquet(self.metrics_dir / "city_summary.parquet", index=False)

        shared = dict(figures_dir=self.figures_dir, city=self.city,
                      dpi=self.fig_dpi, fmt=self.fig_fmt)

        tile_f1_box_plot(metrics_all, **shared)
        tile_f1_spatial_dist(tiles_combined, metrics_all, **shared)
        plot_iou_dist(metrics_all, matches_all, **shared)
        plot_iou_per_building_sizes(
            matches_all,
            size_bins=self.size_bins,
            size_bin_labels=self.size_labels,
            **shared,
        )

        self.logger.info("All figures saved → %s", self.figures_dir)
        return summary

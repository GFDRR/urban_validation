# Data dictionary — merged validation CSVs

Reference for the two global outputs produced by `04_aggregate_global_metrics`
(from the per-city summaries written by the validators):

| File | Grain (one row per) | Rows source |
|---|---|---|
| `outputs/global_metrics/vector_all_cities_merged.csv` | **city × dataset** | `summarize_city()` — vector footprint matching |
| `outputs/global_metrics/raster_all_cities_merged.csv` | **city × dataset × grid** | raster built-up area validation |

Both are also written as `.parquet` and `.xlsx` with identical columns.

---

## Conventions used throughout

- **Headline / "overall" F1 = macro F1 = the unweighted mean of the per-city F1**
  (`f1_city` for vector, `f1_tile_mean` for raster). Do **not** sum TP/FP/FN across
  cities first — that yields a micro / area-weighted F1 dominated by the largest
  cities. `f1_area` (raster) is micro and is *not* the headline metric.
- **Per-city IoU threshold (vector).** Matching uses **τ = 0.25 for SpaceNet-reference
  cities and τ = 0.50 for all others** (see `iou_threshold`). A pooled mean of
  `f1_city` therefore mixes two thresholds; split by reference source for a
  like-for-like comparison.
- **Bias sign.** Positive = the candidate/product **over-represents** the reference
  (more buildings, or more built-up area); negative = under-represents.
- **`_tp` suffix (vector).** Computed over matched (true-positive) pairs **only** —
  a geometry-quality measure, independent of how many buildings were matched.
- Ratios are `NaN` when the denominator is 0 (e.g. `count_ratio_total` when
  `n_ref_total = 0`).

---

## vector_all_cities_merged.csv

One row per **city × candidate dataset** (`overture`, `gba`, `globfp`).

| # | Column | Definition |
|---|---|---|
| 1 | `city` | City slug (e.g. `usa-brentwood`, `jam-kingston-sn7`). |
| 2 | `dataset` | Candidate dataset: `overture` (Overture Maps) \| `gba` (**Global Building Atlas** — *not* Google Open Buildings) \| `globfp` (3D-GloBFP). |
| 3 | `iou_threshold` | IoU threshold (τ) used for matching in this city. **0.25** for SpaceNet-reference cities, **0.50** otherwise. |
| 4 | `n_sub_areas` | Number of sub-AOIs the city AOI was split into (1 for single-AOI cities). |
| 5 | `n_tiles` | Number of 1 km evaluation tiles. |
| 6 | `n_ref_buildings_loaded` | Reference buildings read from file **before** tiling/filtering (raw load count). |
| 7 | `n_cand_buildings_loaded` | Candidate buildings read from file before tiling (raw load count). |
| 8 | `count_ratio_loaded` | `n_cand_buildings_loaded / n_ref_buildings_loaded`. |
| 9 | `n_ref_buildings` | **De-duplicated** reference building count (centroid-in-tile rule; avoids double-counting buildings on tile edges). Preferred "how many reference buildings" figure. |
| 10 | `n_cand_buildings` | De-duplicated candidate building count (pre-tiling loaded count; falls back to `tp+fp`). |
| 11 | `building_count_ratio` | `n_cand_buildings / n_ref_buildings` (de-duplicated ratio). |
| 12 | `n_ref_total` | Reference buildings **summed across tiles** (`tp+fn`). May double-count buildings spanning tile boundaries. |
| 13 | `n_cand_total` | Candidate buildings summed across tiles (`tp+fp`). Same edge caveat. |
| 14 | `count_delta_total` | `n_cand_total − n_ref_total`. |
| 15 | `count_ratio_total` | `n_cand_total / n_ref_total` — candidate-to-reference count ratio. |
| 16 | `rel_count_delta_total` | `count_delta_total / n_ref_total`. |
| 17 | `ref_density_per_km2` | Reference building density (`n_ref_total / aoi_area_km2`). |
| 18 | `cand_density_per_km2` | Candidate building density. |
| 19 | `density_delta_per_km2` | `cand_density − ref_density`. |
| 20 | `density_ratio_per_km2` | `cand_density / ref_density`. |
| 21 | `tp_total` | **True positives** — matched reference↔candidate pairs (IoU ≥ τ), summed over tiles. |
| 22 | `fp_total` | **False positives** — candidate buildings with no match. |
| 23 | `fn_total` | **False negatives** — reference buildings with no match. |
| 24 | `precision_city` | `tp / (tp + fp)` from city totals. |
| 25 | `recall_city` | `tp / (tp + fn)`. |
| 26 | `f1_city` | `2·P·R / (P+R)`. **The per-city F1** (at this city's τ). Headline macro F1 = mean of `f1_city` across cities. |
| 27 | `iou_mean_tp` | Mean IoU over matched (TP) pairs. |
| 28 | `iou_median_tp` | Median IoU over TP pairs. |
| 29 | `iou_p25_tp` | 25th percentile IoU over TP pairs. |
| 30 | `iou_p75_tp` | 75th percentile IoU over TP pairs. |
| 31 | `boundary_f_meanpair_tp` | Mean per-pair boundary F-score over TP pairs — delineation/shape quality, independent of detection. |
| 32 | `rel_area_error_mean_tp` | Mean of `(cand_area − ref_area) / ref_area` over TP pairs — signed **per-building** size error. |
| 33 | `rel_area_error_median_tp` | Median of the above. |
| 34 | `total_area_bias` | **City-wide** built-area bias: `(Σ all candidate area − Σ all reference area) / Σ all reference area`. Includes unmatched buildings (FP add area, FN subtract). Positive = candidate over-represents total built area. |
| 35 | `total_area_bias_pct` | `total_area_bias × 100`. |

**On the three count families (cols 6–13):** `*_loaded` = raw counts as read from
disk; `n_ref_buildings` / `n_cand_buildings` = de-duplicated (best for "how many
buildings"); `*_total` = tile-summed TP+FN / TP+FP (best paired with TP/FP/FN, but
edge-double-counts). Use `n_ref_buildings` for reporting counts, `n_ref_total` only
alongside the confusion matrix.

---

## raster_all_cities_merged.csv

One row per **city × raster product × evaluation grid**. Products: `obt_2023`,
`tempo_2023q4`, `ghsl_built_s_2025`, `wsf_tracker`. Grids: `10m`, `100m` (a product
is only evaluated at grids at or coarser than its native resolution).

| # | Column | Definition |
|---|---|---|
| 1 | `city` | City slug. |
| 2 | `dataset` | Raster product base name (e.g. `wsf_tracker`). Combine with `grid` for a unique key (`wsf_tracker@100m`). |
| 3 | `grid` | Evaluation grid label: `10m` \| `100m`. |
| 4 | `resolution_m` | Grid resolution in metres (10 or 100). |
| 5 | `n_ref_buildings` | Reference building count (from the vector reference). |
| 6 | `aoi_area_km2` | AOI area. |
| 7 | `buildings_per_km2` | Reference building density. |
| 8 | `avg_building_size_m2` | Mean reference building footprint area. |
| 9 | `ref_building_count` | Reference building count (= `n_ref_buildings`, as float). |
| 10 | `pred_building_count` | **Estimated** candidate building count = `pred_area_total_m2 / avg_building_size_m2`. Raster products have no discrete buildings, so this is a derived proxy, not a count. |
| 11 | `delta_building_count` | `pred_building_count − ref_building_count` (estimated). |
| 12 | `rel_delta_building_count` | `delta_building_count / ref_building_count`. |
| 13 | `n_tiles` | Tiles evaluated. |
| 14 | `valid_area_total_m2` | Total valid (non-nodata) evaluated area. |
| 15 | `ref_area_total_m2` | Total **reference** built-up area (m²). |
| 16 | `pred_area_total_m2` | Total **predicted** built-up area (m²). |
| 17 | `tp_total` | True-positive built-up **pixels**, summed over tiles. |
| 18 | `fp_total` | False-positive pixels (predicted built, reference not). |
| 19 | `fn_total` | False-negative pixels (reference built, predicted not). |
| 20 | `precision_area` | **Area-based** precision: `tp_area / (tp_area + fp_area)`, where area = pixels × pixel area. |
| 21 | `recall_area` | Area-based recall. |
| 22 | `f1_area` | Area-based F1 (**micro** — whole-city area-weighted). *Not* the headline metric. |
| 23 | `f1_tile_mean` | **Mean of per-tile F1** (each tile equal weight). **Macro F1 — the headline raster accuracy.** |
| 24 | `f1_tile_median` | Median of per-tile F1. |
| 25 | `rel_area_error_mean` | Unweighted mean of per-tile signed relative area error `(pred − ref)/ref`. Each tile counts equally, so **sparse (low-reference) tiles inflate it**. |
| 26 | `rel_area_error_median` | Median of per-tile signed relative area error. |
| 27 | `signed_area_bias` | `(Σ pred area − Σ ref area) / Σ ref area` from city totals — **area-weighted**. Same quantity as `rel_area_error_mean` but weighted by reference area (large tiles dominate). **The trustworthy "does this product over/under-represent total built-up area" number.** Positive = over-represents. Comparable to the vector `total_area_bias`. |
| 28 | `signed_area_bias_pct` | `signed_area_bias × 100`. |
| 29 | `quantity_disagreement_mean` | Mean per-tile **quantity disagreement** (Pontius): error from a differing *total amount* of built-up. |
| 30 | `allocation_disagreement_mean` | Mean per-tile **allocation disagreement** (Pontius): error from spatial *misplacement* (right amount, wrong place). |
| 31 | `precision_weighted_mean` | Per-tile precision averaged **weighted by each tile's valid area**. |
| 32 | `recall_weighted_mean` | Valid-area-weighted mean of per-tile recall. |
| 33 | `f1_weighted_mean` | Valid-area-weighted mean of per-tile F1 (between `f1_tile_mean` and `f1_area`). |
| 34 | `rel_area_error_weighted_mean` | Valid-area-weighted mean of per-tile relative area error. |
| 35 | `signed_area_bias_weighted_mean` | Valid-area-weighted mean of per-tile signed area bias (≈ `signed_area_bias`). |
| 36 | `signed_area_bias_pct_weighted_mean` | The above × 100. |
| 37 | `quantity_disagreement_weighted_mean` | Valid-area-weighted mean of per-tile quantity disagreement. |
| 38 | `allocation_disagreement_weighted_mean` | Valid-area-weighted mean of per-tile allocation disagreement. |

### Key raster distinctions

- **Three flavours of F1**, in increasing weight on large tiles:
  `f1_tile_mean` (macro, tiles equal — **use this**) → `f1_weighted_mean`
  (tiles weighted by valid area) → `f1_area` (micro, whole-city area pooled).
- **`signed_area_bias` vs `rel_area_error_mean`** measure the *same* signed
  relative area bias `(pred − ref)/ref`; they are identical per tile. They diverge
  at the city level only because `signed_area_bias` is **area-weighted** (from
  totals) while `rel_area_error_mean` is a **plain tile average** inflated by sparse
  tiles. For "how much does this product over/under-estimate built-up area?", use
  `signed_area_bias`.
- `tp/fp/fn_total` are **pixel counts**; the precision/recall/F1 in cols 20–22 are
  **area**-based (pixels × pixel area), so they hold up when pixel sizes differ.
- Built-up **area** products (WSF, GHSL) legitimately show large positive
  `signed_area_bias` (e.g. +2 to +4) — they map settlement *area*, not building
  footprints, so "area bias" is not comparable between area products and
  footprint-derived products.

---

*Generated as part of the pipeline audit. Definitions trace to `summarize_city()`
(vector) and the raster city-summary builder in `src/plots/output.py`.*

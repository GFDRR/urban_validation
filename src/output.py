import datetime
import pandas as pd
import matplotlib.pyplot as plt 
from pathlib import Path 

def fig_name(city, stem: str, ext: str = "png") -> str:
    # e.g. juba_tile_f1_boxplot_20260203_142530.png
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{city.lower()}_{stem}_{ts}.{ext}"

def save_figure(fig, figures_dir, filename: str, dpi: int = 200):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

def summarize_city(city, metrics_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, mds in metrics_df.groupby("dataset"):
        # Totals from tiles
        tp = int(mds["tp"].sum())
        fp = int(mds["fp"].sum())
        fn = int(mds["fn"].sum())
        n_ref = int(mds["n_ref"].sum())
        n_cand = int(mds["n_cand"].sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        # Match-based distribution stats (TP-only)
        dsmatches = matches_df[matches_df["dataset"] == ds] if not matches_df.empty else pd.DataFrame()
        if not dsmatches.empty:
            ious = dsmatches["iou"].astype(float)
            bf = dsmatches["boundary_f_pair"].astype(float) if "boundary_f_pair" in dsmatches.columns else pd.Series(dtype=float)
            rel_area = dsmatches["rel_area_error"].astype(float)

            iou_mean = float(ious.mean())
            iou_median = float(ious.median())
            iou_p25 = float(ious.quantile(0.25))
            iou_p75 = float(ious.quantile(0.75))

            bf_mean = float(bf.mean()) if len(bf) else 0.0
            rel_area_mean = float(rel_area.mean())
            rel_area_median = float(rel_area.median())

            area_ref_sum = float(dsmatches["area_ref"].sum())
            area_cand_sum = float(dsmatches["area_cand"].sum())
            signed_area_bias = ((area_cand_sum - area_ref_sum) / area_ref_sum) if area_ref_sum > 0 else float("nan")
        else:
            iou_mean = iou_median = iou_p25 = iou_p75 = 0.0
            bf_mean = 0.0
            rel_area_mean = rel_area_median = float("nan")
            signed_area_bias = float("nan")

        rows.append({
            "city": city,
            "dataset": ds,
            "n_tiles": int(mds["tile_id"].nunique()),
            "n_ref_total": n_ref,
            "n_cand_total": n_cand,
            "tp_total": tp,
            "fp_total": fp,
            "fn_total": fn,
            "precision_city": precision,
            "recall_city": recall,
            "f1_city": f1,
            "iou_mean_tp": iou_mean,
            "iou_median_tp": iou_median,
            "iou_p25_tp": iou_p25,
            "iou_p75_tp": iou_p75,
            "boundary_f_meanpair_tp": bf_mean,
            "rel_area_error_mean_tp": rel_area_mean,
            "rel_area_error_median_tp": rel_area_median,
            "signed_area_bias_tp": signed_area_bias,
        })

    return pd.DataFrame(rows)


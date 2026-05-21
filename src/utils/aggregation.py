"""Shared aggregation helpers for city and tile summaries."""
from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

import pandas as pd


def aggregate_weighted(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    weight_col: str,
    groupby_cols: Sequence[str],
) -> pd.DataFrame:
    """Return weighted means of ``metric_cols`` for each group.

    Rows with missing weights or missing metric values are ignored on a per-
    metric basis. Groups with zero usable weight return NaN for the metric.
    """
    if df is None or df.empty:
        cols = list(groupby_cols) + list(metric_cols)
        return pd.DataFrame(columns=cols)

    if weight_col not in df.columns:
        raise KeyError(f"weight_col {weight_col!r} is missing from DataFrame")

    missing_group_cols = [col for col in groupby_cols if col not in df.columns]
    if missing_group_cols:
        raise KeyError(f"groupby_cols missing from DataFrame: {missing_group_cols}")

    missing_metric_cols = [col for col in metric_cols if col not in df.columns]
    if missing_metric_cols:
        raise KeyError(f"metric_cols missing from DataFrame: {missing_metric_cols}")

    rows = []
    groupby_cols = list(groupby_cols)
    metric_cols = list(metric_cols)

    grouped = df.groupby(groupby_cols, dropna=False, sort=False) if groupby_cols else [((), df)]
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: key for col, key in zip(groupby_cols, keys)}
        weights = pd.to_numeric(g[weight_col], errors="coerce")

        for metric in metric_cols:
            values = pd.to_numeric(g[metric], errors="coerce")
            valid = values.notna() & weights.notna()
            if valid.any():
                valid = valid & values.map(lambda v: isfinite(v) if pd.notna(v) else False)
                valid = valid & weights.map(lambda v: isfinite(v) if pd.notna(v) else False)
            if not valid.any():
                row[metric] = float("nan")
                continue

            metric_weights = weights[valid].astype(float)
            total_weight = float(metric_weights.sum())
            if total_weight <= 0:
                row[metric] = float("nan")
                continue

            weighted_sum = float((values[valid].astype(float) * metric_weights).sum())
            row[metric] = weighted_sum / total_weight

        rows.append(row)

    return pd.DataFrame(rows)
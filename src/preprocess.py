from __future__ import annotations
from typing import Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd 

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
        # Shapely 2.x
        from shapely import make_valid as _make_valid
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

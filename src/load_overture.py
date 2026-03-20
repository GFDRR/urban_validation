import os
import json
from pathlib import Path

import duckdb
import geopandas as gpd
import shapely
from shapely.geometry import mapping
import yaml
import requests

from pathlib import Path
import geopandas as gpd
import fiona

def read_aoi_any(aoi_path: str, layer: str | None = None) -> gpd.GeoDataFrame:
    p = Path(aoi_path)
    suf = p.suffix.lower()

    if suf in {".parquet", ".geoparquet"}:
        return gpd.read_parquet(p)

    if suf in {".geojson", ".json"}:
        return gpd.read_file(p)  # no layer for GeoJSON

    # Multi-layer containers (gpkg, gdb, etc.)
    if layer:
        return gpd.read_file(p, layer=layer)

    try:
        layers = fiona.listlayers(p)
        if layers:
            return gpd.read_file(p, layer=layers[0])
    except Exception:
        pass

    return gpd.read_file(p)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_latest_overture_release() -> str:
    # Overture STAC root exposes ".latest" and is designed to keep scripts stable across releases :contentReference[oaicite:4]{index=4}
    j = requests.get("https://stac.overturemaps.org/catalog.json", timeout=30).json()
    return j["latest"]

def overture_base_path(cfg: dict) -> str:
    rel = cfg["overture"]["release"]
    if rel == "latest":
        rel = get_latest_overture_release()

    provider = cfg["overture"]["provider"].lower()
    if provider == "aws":
        # Official AWS pattern :contentReference[oaicite:5]{index=5}
        return f"s3://overturemaps-us-west-2/release/{rel}"
    elif provider == "azure":
        # Official Azure pattern :contentReference[oaicite:6]{index=6}
        return f"az://overturemapswestus2.blob.core.windows.net/release/{rel}"
    else:
        raise ValueError(f"Unknown provider: {provider}")

def duckdb_connect(cfg: dict):
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    provider = cfg["overture"]["provider"].lower()
    if provider == "aws":
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute(f"SET s3_region='{cfg['overture'].get('s3_region','us-west-2')}';")
        con.execute("SET s3_url_style='path';")
    elif provider == "azure":
        # IMPORTANT: use azure extension for az:// paths
        con.execute("INSTALL azure; LOAD azure;")
    return con

def aoi_to_4326(gdf: gpd.GeoDataFrame, cfg: dict) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    if cfg["aoi"].get("buffer_meters", 0) and gdf.crs and gdf.crs.is_projected:
        gdf["geometry"] = gdf.geometry.buffer(cfg["aoi"]["buffer_meters"])
    # ensure output CRS in lon/lat
    gdf = gdf.to_crs(cfg["aoi"]["crs_out"])
    return gdf

def extract_overture_for_aoi(con, base_path: str, theme: str, typ: str,
                            aoi_geom_4326, out_path: Path, overwrite: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return str(out_path)

    # AOI bbox (lon/lat)
    minx, miny, maxx, maxy = aoi_geom_4326.bounds

    # Pass AOI geometry as GeoJSON into DuckDB
    aoi_geojson = json.dumps(mapping(aoi_geom_4326))

    # Remote parquet glob
    # parquet_glob = f"{base_path}/theme={theme}/type={typ}/*"
    parquet_glob = f"{base_path}/theme={theme}/type={typ}/*.parquet"
    print(parquet_glob)

    sql = f"""
    COPY (
    WITH aoi AS (
        SELECT ST_GeomFromGeoJSON('{aoi_geojson}') AS geom
    )
    SELECT b.*
    FROM read_parquet('{parquet_glob}') AS b, aoi
    WHERE
        b.bbox.xmin <= {maxx} AND b.bbox.xmax >= {minx}
        AND b.bbox.ymin <= {maxy} AND b.bbox.ymax >= {miny}
        AND ST_Intersects(b.geometry, aoi.geom)
    )
    TO '{str(out_path)}'
    (FORMAT PARQUET);
    """
    con.execute(sql)
    return str(out_path)

def run_pipeline(config_path: str):
    cfg = load_config(config_path)

    # Load AOIs
    aoi_path = cfg["aoi"]["path"]
    layer = cfg["aoi"].get("layer", None)
    gdf = read_aoi_any(aoi_path, layer=layer)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")  # typical for GeoJSON
    gdf = gdf.to_crs(cfg["aoi"]["crs_out"])

    id_col = cfg["aoi"]["id_col"]
    gdf = aoi_to_4326(gdf, cfg)

    # DuckDB connection
    con = duckdb_connect(cfg)
    base_path = overture_base_path(cfg)

    out_root = Path(cfg["output"]["root_dir"])
    overwrite = bool(cfg["output"].get("overwrite", False))

    theme = cfg["overture"]["theme"]
    types = cfg["overture"]["types"]

    outputs = []
    for _, row in gdf.iterrows():
        aoi_id = str(row[id_col])
        geom = row.geometry

        for typ in types:
            out_path = out_root / f"aoi={aoi_id}" / f"{theme}_{typ}.parquet"
            out_file = extract_overture_for_aoi(
                con=con,
                base_path=base_path,
                theme=theme,
                typ=typ,
                aoi_geom_4326=geom,
                out_path=out_path,
                overwrite=overwrite
            )
            outputs.append(out_file)

    return outputs
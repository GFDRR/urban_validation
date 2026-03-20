"""
TODO: contain code for reporting output visualizations
"""


import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, GroupedLayerControl
from pathlib import Path

COUNTRY_CENTROIDS = {
    "ant": (12.10, -68.90),  "bgd": (23.70,  90.40),  "blz": (17.30, -88.50),
    "bra": (-12.90, -38.40), "col": ( 6.20, -75.60),  "cvg": (13.20, -61.20),
    "dom": (15.40, -61.40),  "gha": ( 7.90,  -1.00),  "grd": (12.10, -61.70),
    "jam": (18.00, -76.80),  "jpn": (35.70, 139.70),  "ken": ( 1.30,  38.00),
    "lbr": ( 6.30, -10.80),  "lby": (27.00,  17.00),  "lca": (13.90, -60.90),
    "maf": (18.10, -63.10),  "mmr": (21.90,  96.10),  "moz": (-18.70, 35.50),
    "mwi": (-13.30, 34.30),  "ner": (13.50,   2.10),  "nga": ( 7.40,   3.90),
    "phl": (11.80, 122.10),  "sle": ( 8.50, -13.20),  "ssd": ( 4.90,  31.60),
    "swz": (-26.50, 31.50),  "sxm": (18.00, -63.10),  "tjk": (39.00,  71.00),
    "ton": (-21.10,-175.20), "tto": (10.40, -61.30),  "uga": ( 1.40,  32.30),
    "ukr": (50.90,  28.10),
}

def _country_centroid(dataset_id: str):
    prefix = dataset_id[:3].lower()
    return COUNTRY_CENTROIDS.get(prefix, (None, None))

def build_inventory_map(
    csv_path: str,
    base_dir: str,
    out_html: str = "dataset_inventory_map.html",
) -> folium.Map:
    """
    Build an interactive Folium map from the dataset AOI reference inventory.

    Parameters
    ----------
    csv_path  : path to dataset_aoi_reference_list.csv
    base_dir  : root directory where AOI geojson files live (e.g. data/01_raw)
    out_html  : output path for the saved HTML map
    """
    base_dir = Path(base_dir)
    df = pd.read_csv(csv_path).drop_duplicates(subset=["Dataset code"])


    m = folium.Map(
        location=[15, 20],
        zoom_start=2,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    layer_full    = folium.FeatureGroup(name="✅ Full coverage (AOI + Reference)", show=True)
    layer_partial = folium.FeatureGroup(name="⚠️ Partial (missing file)", show=True)
    layer_aoi     = folium.FeatureGroup(name="🗺️ AOI polygons", show=False)

    # ── Process each dataset ──────────────────────────────────────────────────
    for _, row in df.iterrows():
        dataset_id = str(row["Dataset code"])
        has_aoi    = bool(row["has_aoi_file"])
        has_ref    = bool(row["has_reference_file"])
        aoi_count  = int(row["aoi_file_count"])
        ref_count  = int(row["reference_file_count"])
        score      = row["match_score"]
        raw_paths  = str(row.get("aoi_file_path", ""))

        # Resolve AOI file paths
        aoi_paths = [
            base_dir / p.strip()
            for p in raw_paths.split("|")
            if p.strip()
        ]
        existing = [p for p in aoi_paths if p.exists()]

        # Status and styling
        if has_aoi and has_ref:
            status      = "Full"
            color       = "#00e5ff"
            fill_color  = "#00e5ff"
            layer_target = layer_full
        elif has_aoi or has_ref:
            status      = "Partial"
            color       = "#fbbf24"
            fill_color  = "#fbbf24"
            layer_target = layer_partial
        else:
            status      = "Missing"
            color       = "#94a3b8"
            fill_color  = "#94a3b8"
            layer_target = layer_partial

        # Marker radius scaled by file count
        radius = min(6 + (max(aoi_count, ref_count, 1) ** 0.5) * 3.5, 28)

        centroid_lat, centroid_lon = None, None
        aoi_gdf = None

        if existing:
            try:
                parts = [gpd.read_file(p) for p in existing]
                aoi_gdf = gpd.GeoDataFrame(
                    pd.concat(parts, ignore_index=True), crs=parts[0].crs
                )
                if aoi_gdf.crs and not aoi_gdf.crs.is_geographic:
                    aoi_gdf = aoi_gdf.to_crs("EPSG:4326")
                centroid = aoi_gdf.geometry.union_all().centroid
                centroid_lat, centroid_lon = centroid.y, centroid.x
            except Exception as e:
                print(f"[WARN] {dataset_id}: could not load AOI — {e}")

        if centroid_lat is None:
            # Fallback: derive from country code prefix
            centroid_lat, centroid_lon = _country_centroid(dataset_id)

        if centroid_lat is None:
            print(f"[SKIP] {dataset_id}: no geometry and no known centroid")
            continue

        popup_html = f"""
        <div style="font-family:monospace; font-size:12px; min-width:200px;">
          <b style="font-size:14px; color:{color};">{dataset_id}</b><br><br>
          <table style="width:100%; border-collapse:collapse;">
            <tr><td style="color:#888;">Status</td>
                <td><b style="color:{'#a8ff3e' if status=='Full' else '#fbbf24'};">{status}</b></td></tr>
            <tr><td style="color:#888;">AOI files</td><td><b>{aoi_count}</b></td></tr>
            <tr><td style="color:#888;">Ref files</td><td><b>{ref_count}</b></td></tr>
            <tr><td style="color:#888;">Match score</td><td><b>{score}%</b></td></tr>
            <tr><td style="color:#888;">Has AOI</td>
                <td><b style="color:{'#a8ff3e' if has_aoi else '#fbbf24'};">{'✓' if has_aoi else '✗'}</b></td></tr>
            <tr><td style="color:#888;">Has Ref</td>
                <td><b style="color:{'#a8ff3e' if has_ref else '#fbbf24'};">{'✓' if has_ref else '✗'}</b></td></tr>
          </table>
        </div>
        """

        tooltip = folium.Tooltip(
            f"<b>{dataset_id}</b> — {status} | AOI: {aoi_count} · Ref: {ref_count}",
            sticky=True,
        )

        folium.CircleMarker(
            location=[centroid_lat, centroid_lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.3,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=tooltip,
        ).add_to(layer_target)

        if aoi_gdf is not None:
            try:
                dissolved = aoi_gdf.dissolve().reset_index(drop=True)
                folium.GeoJson(
                    dissolved.__geo_interface__,
                    name=dataset_id,
                    style_function=lambda _, c=color: {
                        "color": c,
                        "weight": 1.5,
                        "fillOpacity": 0.08,
                        "fillColor": c,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=[],
                        aliases=[],
                        sticky=True,
                    ) if False else folium.Tooltip(dataset_id),
                ).add_to(layer_aoi)
            except Exception as e:
                print(f"[WARN] {dataset_id}: polygon layer failed — {e}")

    layer_full.add_to(m)
    layer_partial.add_to(m)
    layer_aoi.add_to(m)
    # folium.LayerControl(collapsed=False).add_to(m)

    total    = len(df)
    n_full   = df.apply(lambda r: bool(r["has_aoi_file"]) and bool(r["has_reference_file"]), axis=1).sum()
    n_partial = total - n_full

    stats_html = f"""
    <div style="
        position: fixed; top: 10px; right: 10px; z-index: 1000;
        background: rgba(17,24,39,0.92); border: 1px solid #1e2d45;
        padding: 14px 18px; font-family: monospace; font-size: 12px;
        color: #e2e8f0; min-width: 180px; backdrop-filter: blur(4px);
    ">
      <div style="font-size:11px; color:#64748b; text-transform:uppercase;
                  letter-spacing:0.1em; margin-bottom:10px;">
        Dataset Inventory
      </div>
      <div style="margin-bottom:6px;">
        <span style="color:#64748b;">Total datasets</span><br>
        <b style="font-size:20px; color:#00e5ff;">{total}</b>
      </div>
      <div style="margin-bottom:6px;">
        <span style="color:#64748b;">Full coverage</span><br>
        <b style="font-size:18px; color:#a8ff3e;">{n_full}</b>
      </div>
      <div>
        <span style="color:#64748b;">Partial</span><br>
        <b style="font-size:18px; color:#fbbf24;">{n_partial}</b>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    m.save(out_html)
    print(f"Map saved → {out_html}")
    return m
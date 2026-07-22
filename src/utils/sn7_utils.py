#!/usr/bin/env python3
"""
Group SpaceNet 7 building labels by city and write merged reference files.

For each city (single-tile or multi-tile), reads all monthly GeoJSONs from
  {sn7_dir}/{tile}/labels_match/*.geojson
merges footprints across all months and tiles via spatial union, then writes:
  {output_root}/{city_id}/vector/{city_id}_SN7.geojson
  {output_root}/{city_id}/aoi/{city_id}_aoi.geojson   (union of tile extents)

Optionally updates aoi_tracker.csv (--tracker-path) with one row per city,
skipping any city that already has an SN7 entry.

Usage (Colab):
    python src/utils/sn7_utils.py \\
        --sn7-dir /path/to/local/data \\
        --output-root "/content/drive/MyDrive/<project>/data/01_raw" \\
        --tracker-path "/content/drive/MyDrive/<project>/data/02_interim/aoi_tracker.csv"

Usage (local with Google Drive Desktop):
    python src/utils/sn7_utils.py \\
        --sn7-dir data \\
        --output-root "/path/to/<project>/data/01_raw" \\
        --tracker-path "/path/to/<project>/data/02_interim/aoi_tracker.csv"

Optional flags:
    --cities city1 city2   Process only these city IDs (default: all)
    --overwrite            Overwrite existing output and AOI files
    --min-area-m2 N        Drop polygons smaller than N m² (default: 10)
    --min-year YYYY        Use snapshots >= this year; fallback to most-recent
                           per tile if none qualify (default: 2020)
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CITY_TILES: dict[str, list[str]] = {
    # Americas
    # Tile centroid is 6 km from Brentwood CA (eastern Contra Costa Co.), not Oakland (57 km away)
    "usa-brentwood":     ["L15-0331E-1257N_1327_3160_13"],
    # Two tiles 60 km apart: north tile is San Marcos/Escondido (N San Diego Co.),
    # south tile is Chula Vista / Otay Ranch near the US–Mexico border.
    # Split into usa-sandiego + mex-tijuana if you need separate national datasets.
    "usa-sandiego":      ["L15-0357E-1223N_1429_3296_13",
                          "L15-0358E-1220N_1433_3310_13"],
    "usa-boise":         ["L15-0361E-1300N_1446_2989_13"],
    "usa-lasvegas":      ["L15-0368E-1245N_1474_3210_13"],
    # Tile centroid is 33 km south of SLC downtown, in Saratoga Springs / Lehi (Utah County)
    "usa-saltlakecity":  ["L15-0387E-1276N_1549_3087_13"],
    "usa-permianbasin":  ["L15-0434E-1218N_1736_3318_13"],   # TODO: no mapped city; oil-field area
    "mex-mexicocity":    ["L15-0457E-1135N_1831_3648_13"],
    "usa-bentonville":   ["L15-0487E-1246N_1950_3207_13"],
    # Tile centroid is 85 km west of New Orleans, 4 km from Gonzales/Sorrento (Ascension Parish LA)
    "usa-gonzales":      ["L15-0506E-1204N_2027_3374_13"],
    "usa-atlanta":       ["L15-0544E-1228N_2176_3279_13"],
    # Tile centroid is 69 km north of West Palm Beach, 3 km from Tradition / 10 km from Port St. Lucie
    "usa-portstlucie":   ["L15-0566E-1185N_2265_3451_13"],
    "pan-panamacity":    ["L15-0571E-1075N_2287_3888_13"],
    "usa-raleigh":       ["L15-0577E-1243N_2309_3217_13"],   # Wake Forest area, Raleigh metro
    "jam-kingston":      ["L15-0586E-1127N_2345_3680_13"],
    "usa-allentown":     ["L15-0595E-1278N_2383_3079_13"],   # Easton / Palmer Township, Lehigh Valley
    "per-cusco":         ["L15-0614E-0946N_2459_4406_13"],
    "chl-calama":        ["L15-0632E-0892N_2528_4620_13"],
    "bra-manaus":        ["L15-0683E-1006N_2732_4164_13"],
    "bra-saopaulo":      ["L15-0760E-0887N_3041_4643_13"],   # NE metro (Guarulhos / Itaquaquecetuba)

    # Africa
    "sen-dakar":         ["L15-0924E-1108N_3699_3757_13"],
    "alg-tindouf":       ["L15-0977E-1187N_3911_3441_13"],   # 7 km from Tindouf city center
    "gha-kumasi":        ["L15-1015E-1062N_4061_3941_13"],
    "lby-benghazi":      ["L15-1138E-1216N_4553_3325_13"],
    "zmb-lusaka":        ["L15-1185E-0935N_4742_4450_13"],
    "zaf-durban":        ["L15-1200E-0847N_4802_4803_13"],
    # Three tiles in the greater Cairo development corridor (east of the city):
    #   tile 1 (31.62, 30.05): New Cairo area, 37 km from Tahrir Sq
    #   tile 2 (31.66, 29.97): SE Cairo fringe, 42 km
    #   tile 3 (31.79, 30.28): 7 km from 10th of Ramadan City, 60 km from Tahrir Sq
    # Kept together as egy-cairo (Greater Cairo master-plan zone). To separate tile 3,
    # rename it to egy-ramadancity.
    "egy-cairo":         ["L15-1203E-1203N_4815_3378_13",
                          "L15-1204E-1202N_4816_3380_13",
                          "L15-1204E-1204N_4819_3372_13"],
    "sdn-khartoum":      ["L15-1209E-1113N_4838_3737_13"],
    "uga-kampala":       ["L15-1210E-1025N_4840_4088_13"],

    # Europe
    "gbr-birmingham":    ["L15-1014E-1375N_4056_2688_13"],   # Birmingham Airport / NEC area
    "gbr-london":        ["L15-1025E-1366N_4102_2726_13"],   # Dartford / Gravesend, 30 km E of London
    "nld-rotterdam":     ["L15-1049E-1370N_4196_2710_13"],
    "rou-bucharest":     ["L15-1172E-1306N_4688_2967_13"],

    # Middle East / Central Asia
    "yem-dhamar":        ["L15-1276E-1107N_5105_3761_13"],   # 4 km from Dhamar (not Sana'a)
    "sau-riyadh":        ["L15-1289E-1169N_5156_3514_13"],
    "kwt-kuwaitcity":    ["L15-1296E-1198N_5184_3399_13"],   # SW suburbs (Ahmadi / Fahaheel area)
    "rus-astrakhan":     ["L15-1298E-1322N_5193_2903_13"],
    "are-abudhabi":      ["L15-1335E-1166N_5342_3524_13"],   # 44 km E of Abu Dhabi city, eastern emirate
    "uzb-zarafshan":     ["L15-1389E-1284N_5557_3054_13"],

    # South / Southeast Asia
    "ind-mumbai":        ["L15-1438E-1134N_5753_3655_13",
                          "L15-1439E-1134N_5759_3655_13"],   # tile 2 covers Navi Mumbai / Panvel
    # Tile centroid is 56 km NW of Chennai, 10 km from Tada (SPSR Nellore district, Andhra Pradesh)
    "ind-tada":          ["L15-1479E-1101N_5916_3785_13"],
    "ind-vijayawada":    ["L15-1481E-1119N_5927_3715_13"],
    "bgd-dhaka":         ["L15-1538E-1163N_6154_3539_13"],

    # East Asia
    # SW tile (103.91, 30.35) is 37 km from Chengdu; other two are within the metro (20-23 km).
    "chn-chengdu":       ["L15-1615E-1205N_6460_3370_13",
                          "L15-1615E-1206N_6460_3366_13",
                          "L15-1617E-1207N_6468_3360_13"],
    "chn-zhuhai":        ["L15-1669E-1153N_6678_3579_13"],   # Hengqin area, 18 km from Zhuhai center
    "chn-guangzhou":     ["L15-1669E-1160N_6678_3548_13",
                          "L15-1669E-1160N_6679_3549_13"],   # NE metro, 36 km from Tianhe
    "chn-wuhan":         ["L15-1672E-1207N_6691_3363_13"],
    # Tiles are 4 km from Lujiang county seat (Lujiang is administered by Hefei City but
    # is 67 km south of the Hefei urban area). Label reflects administrative jurisdiction.
    "chn-lujiang":       ["L15-1690E-1211N_6763_3346_13",
                          "L15-1691E-1211N_6764_3347_13"],
    "chn-yangzhou":      ["L15-1703E-1219N_6813_3313_13"],
    "phl-angelescity":   ["L15-1709E-1112N_6838_3742_13"],   # 23 km N of Angeles City (Capas/Tarlac area)
    "chn-shanghai":      ["L15-1716E-1211N_6864_3345_13"],   # Pudong / eastern Shanghai
    "kor-sejong":        ["L15-1748E-1247N_6993_3202_13"],

    # Oceania
    "aus-melbourne":     ["L15-1848E-0793N_7394_5018_13"],   # N suburbs (Craigieburn area)
}

SRC_CRS = "EPSG:3857"
OUT_CRS = "EPSG:4326"

_DATE_RE = re.compile(r"global_monthly_(\d{4})_(\d{2})_mosaic")
_TILE_RE = re.compile(r"_(\d+)_(\d+)_13$")

# Columns in aoi_tracker.csv (matches the live file; BOM handled via utf-8-sig).
_TRACKER_FIELDNAMES = [
    "Dataset code",
    "Suitable (yes/N)",
    "is_high_quality",
    "dataset_folder_name",
    "aoi_file_name",
    "reference_file_name",
    "aoi_file_count",
    "reference_file_count",
    "has_aoi_file",
    "has_reference_file",
]


# ---------------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------------

def _select_files(labels_dir: Path, min_year: int = 2020) -> tuple[list[Path], str]:
    """Return the files to load for one tile and a human-readable note.

    Strategy:
      1. Parse YYYY_MM from each filename.
      2. If any file has year >= min_year, return all such files.
      3. Otherwise fall back to the single most-recent file (closest to
         min_year without going over), so no data is silently excluded.
    """
    candidates: list[tuple[int, int, Path]] = []
    for f in sorted(labels_dir.glob("*.geojson")):
        m = _DATE_RE.search(f.name)
        if m:
            candidates.append((int(m.group(1)), int(m.group(2)), f))

    if not candidates:
        return [], "no dated files found"

    preferred = [(y, mo, p) for y, mo, p in candidates if y >= min_year]
    if preferred:
        note = f"{len(preferred)} file(s) ≥ {min_year}"
        return [p for _, _, p in preferred], note

    latest = max(candidates, key=lambda t: (t[0], t[1]))
    note = f"fallback → {latest[0]}-{latest[1]:02d} (no {min_year}+ data)"
    return [latest[2]], note


# ---------------------------------------------------------------------------
# Tile loading and city reference building
# ---------------------------------------------------------------------------

def _load_tile_labels(tile_dir: Path, min_year: int = 2020) -> gpd.GeoDataFrame | None:
    """Read and concatenate the selected monthly GeoJSONs for one tile."""
    labels_dir = tile_dir / "labels_match"
    if not labels_dir.is_dir():
        log.warning("  labels_match not found in %s — skipping tile", tile_dir.name)
        return None

    files, note = _select_files(labels_dir, min_year)
    if not files:
        log.warning("  No GeoJSON files in %s — skipping tile", labels_dir)
        return None

    parts: list[gpd.GeoDataFrame] = []
    for f in files:
        try:
            gdf = gpd.read_file(f)
            if gdf.empty:
                continue
            if gdf.crs is None:
                gdf = gdf.set_crs(SRC_CRS)
            parts.append(gdf[["geometry"]])
        except Exception as exc:
            log.warning("  Could not read %s: %s", f.name, exc)

    if not parts:
        return None

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=SRC_CRS)
    log.info("  tile %-45s  %d raw polygons  [%s]", tile_dir.name, len(combined), note)
    return combined


def _build_city_reference(
    city_id: str,
    tile_dirs: list[Path],
    min_area_m2: float,
    min_year: int = 2020,
) -> gpd.GeoDataFrame | None:
    """Merge selected tile/month footprints for a city into a single clean layer."""
    all_parts: list[gpd.GeoDataFrame] = []
    for td in tile_dirs:
        gdf = _load_tile_labels(td, min_year)
        if gdf is not None:
            all_parts.append(gdf)

    if not all_parts:
        log.error("  %s: no data loaded — skipping", city_id)
        return None

    raw = gpd.GeoDataFrame(pd.concat(all_parts, ignore_index=True), crs=SRC_CRS)
    raw["geometry"] = raw.geometry.buffer(0)
    raw = raw[raw.geometry.is_valid & ~raw.geometry.is_empty]

    log.info("  %s: unioning %d raw polygons …", city_id, len(raw))
    t0 = time.time()
    merged_geom = raw.geometry.union_all()
    log.info("  %s: union done in %.1f s", city_id, time.time() - t0)

    merged = (
        gpd.GeoDataFrame(geometry=[merged_geom], crs=SRC_CRS)
        .explode(index_parts=False)
        .reset_index(drop=True)
    )

    # Derive UTM zone from Web Mercator centroid to avoid geographic-CRS centroid warning.
    merc_cx = merged.geometry.centroid.x.mean()
    merc_cy = merged.geometry.centroid.y.mean()
    center_lon = merc_cx / 20037508.34 * 180
    center_lat = math.degrees(2 * math.atan(math.exp(merc_cy / 20037508.34 * math.pi)) - math.pi / 2)
    utm_zone = int((center_lon + 180) / 6) + 1
    hemisphere = "north" if center_lat >= 0 else "south"
    utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84 +units=m +no_defs"

    merged_utm = merged.to_crs(utm_crs)
    merged["area_m2"] = merged_utm.geometry.area.values
    merged = merged.to_crs(OUT_CRS)

    before = len(merged)
    merged = merged[merged["area_m2"] >= min_area_m2].reset_index(drop=True)
    dropped = before - len(merged)
    if dropped:
        log.info("  %s: dropped %d polygon(s) < %.0f m²", city_id, dropped, min_area_m2)

    log.info("  %s: final reference — %d polygons, %.1f km² total area",
             city_id, len(merged), merged["area_m2"].sum() / 1e6)
    return merged


# ---------------------------------------------------------------------------
# AOI generation from tile bounding boxes
# ---------------------------------------------------------------------------

def _tile_bbox_wgs84(tile_name: str):
    """Return a WGS84 bounding-box polygon for a zoom-13 SN7 tile name."""
    m = _TILE_RE.search(tile_name)
    if not m:
        return None
    x, y, n = int(m.group(1)), int(m.group(2)), 2 ** 13
    west  = x / n * 360 - 180
    east  = (x + 1) / n * 360 - 180
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return shapely_box(west, south, east, north)


def _save_city_aoi(
    city_id: str, tiles: list[str], output_root: Path, overwrite: bool
) -> str | None:
    """Union tile bounding boxes and write {city_id}_aoi.geojson; return filename or None."""
    aoi_dir = output_root / city_id / "aoi"
    aoi_path = aoi_dir / f"{city_id}_aoi.geojson"

    if aoi_path.exists() and not overwrite:
        log.info("  %s: AOI already exists — skipping", city_id)
        return aoi_path.name

    bboxes = [b for t in tiles if (b := _tile_bbox_wgs84(t)) is not None]
    if not bboxes:
        log.warning("  %s: could not derive tile bounding boxes for AOI", city_id)
        return None

    aoi_geom = unary_union(bboxes)
    aoi_dir.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[aoi_geom], crs="EPSG:4326").to_file(aoi_path, driver="GeoJSON")
    log.info("  %s: AOI saved → %s", city_id, aoi_path)
    return aoi_path.name


# ---------------------------------------------------------------------------
# aoi_tracker.csv update
# ---------------------------------------------------------------------------

def _update_tracker(tracker_path: Path, new_rows: list[dict]) -> None:
    """Append new SN7 rows to aoi_tracker.csv, skipping cities already present."""
    if not new_rows:
        return

    existing_rows: list[dict] = []
    fieldnames: list[str] = _TRACKER_FIELDNAMES

    if tracker_path.exists():
        with open(tracker_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            existing_rows = list(reader)

    ref_col = next(
        (c for c in fieldnames if "reference" in c.lower() and "file" in c.lower()),
        "reference_file_name",
    )
    already_sn7 = {
        r["dataset_folder_name"]
        for r in existing_rows
        if "SN7" in (r.get(ref_col) or "")
    }

    to_add = [r for r in new_rows if r["dataset_folder_name"] not in already_sn7]
    if not to_add:
        log.info(
            "Tracker: all %d processed cities already have SN7 entries — nothing added",
            len(new_rows),
        )
        return

    with open(tracker_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows + to_add)

    log.info("Tracker: added %d new row(s) → %s", len(to_add), tracker_path)
    for r in to_add:
        log.info("  + %s", r["dataset_folder_name"])


# ---------------------------------------------------------------------------
# Per-city orchestration
# ---------------------------------------------------------------------------

def process_city(
    city_id: str,
    tiles: list[str],
    sn7_dir: Path,
    output_root: Path,
    min_area_m2: float,
    min_year: int,
    overwrite: bool,
) -> dict | None:
    """Process one city; return a tracker-row dict on success, None on failure."""
    out_dir = output_root / city_id / "vector"
    out_path = out_dir / f"{city_id}_SN7.geojson"

    if out_path.exists() and not overwrite:
        log.info("[SKIP] %s — reference already exists (use --overwrite to replace)", city_id)
    else:
        log.info("[START] %s (%d tile(s))", city_id, len(tiles))

        tile_dirs = [sn7_dir / t for t in tiles]
        missing = [td for td in tile_dirs if not td.is_dir()]
        if missing:
            log.warning("  Missing tile dirs: %s", [td.name for td in missing])

        existing_dirs = [td for td in tile_dirs if td.is_dir()]
        if not existing_dirs:
            log.error("  %s: no tile directories found — skipping", city_id)
            return None

        gdf = _build_city_reference(city_id, existing_dirs, min_area_m2, min_year)
        if gdf is None or gdf.empty:
            log.error("  %s: empty result — nothing saved", city_id)
            return None

        out_dir.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_path, driver="GeoJSON")
        log.info("[DONE]  %s → %s", city_id, out_path)

    # AOI is cheap (tile bbox only) — always attempt even on reference skip.
    aoi_fname = _save_city_aoi(city_id, tiles, output_root, overwrite)

    return {
        "Dataset code":         city_id,
        "Suitable (yes/N)":     "yes",
        "is_high_quality":      "TRUE",
        "dataset_folder_name":  city_id,
        "aoi_file_name":        aoi_fname or "",
        "reference_file_name":  out_path.name,
        "aoi_file_count":       "1",
        "reference_file_count": "1",
        "has_aoi_file":         "TRUE" if aoi_fname else "FALSE",
        "has_reference_file":   "TRUE",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments and build merged SN7 city reference and AOI files."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sn7-dir",
        default="data",
        help="Root directory containing the SN7 tile folders (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help=(
            "Root of data/01_raw, e.g. "
            "'/content/drive/MyDrive/<project>/data/01_raw'"
        ),
    )
    parser.add_argument(
        "--tracker-path",
        default=None,
        help=(
            "Path to aoi_tracker.csv on Google Drive. "
            "If provided, successfully processed cities are appended (skipping duplicates). "
            "e.g. '.../data/02_interim/aoi_tracker.csv'"
        ),
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        metavar="CITY_ID",
        help="Process only these city IDs (default: all cities)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing reference and AOI output files",
    )
    parser.add_argument(
        "--min-area-m2",
        type=float,
        default=10.0,
        help="Drop merged polygons smaller than this area in m² (default: %(default)s)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2020,
        help=(
            "Use only snapshots from this year onward. "
            "For tiles with no data in that range, falls back to the single "
            "most-recent snapshot available (default: %(default)s)"
        ),
    )
    args = parser.parse_args()

    sn7_dir = Path(args.sn7_dir)
    output_root = Path(args.output_root)

    if not sn7_dir.is_dir():
        log.error("--sn7-dir %s does not exist", sn7_dir)
        sys.exit(1)

    city_map = CITY_TILES
    if args.cities:
        unknown = set(args.cities) - set(city_map)
        if unknown:
            log.error("Unknown city IDs: %s\nValid IDs: %s",
                      sorted(unknown), sorted(city_map))
            sys.exit(1)
        city_map = {k: v for k, v in city_map.items() if k in args.cities}

    log.info("Processing %d city/cities from %s", len(city_map), sn7_dir)
    log.info("Output root: %s", output_root)
    log.info("Year filter: >= %d (fallback to most-recent snapshot per tile)", args.min_year)
    if args.tracker_path:
        log.info("Tracker: %s", args.tracker_path)

    t_total = time.time()
    tracker_rows: list[dict] = []
    failed: list[str] = []

    for city_id, tiles in city_map.items():
        row = process_city(
            city_id, tiles, sn7_dir, output_root,
            args.min_area_m2, args.min_year, args.overwrite,
        )
        if row is not None:
            tracker_rows.append(row)
        else:
            failed.append(city_id)

    log.info("─" * 60)
    log.info(
        "Done in %.1f s — %d/%d succeeded",
        time.time() - t_total, len(tracker_rows), len(city_map),
    )

    if args.tracker_path:
        _update_tracker(Path(args.tracker_path), tracker_rows)

    if failed:
        log.warning("Failed: %s", failed)
        sys.exit(1)


if __name__ == "__main__":
    main()

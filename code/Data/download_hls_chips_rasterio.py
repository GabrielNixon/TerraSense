import os
import math
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import earthaccess
from pystac_client import Client
from pyproj import CRS, Transformer

import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "Fmask"]


@dataclass
class ChipSpec:
    chip_meters: float = 7680.0
    resolution: float = 30.0
    max_cloud_frac: float = 0.20
    max_items_to_try: int = 30
    year_windows_north: Tuple[str, str] = ("06-01", "09-15")
    year_windows_south: Tuple[str, str] = ("12-01", "03-15")


def bbox_wgs84_for_search(lat: float, lon: float, half_size_m: float) -> List[float]:
    lat_deg_per_m = 1.0 / 111_320.0
    lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat)) + 1e-9)
    dlat = half_size_m * lat_deg_per_m
    dlon = half_size_m * lon_deg_per_m
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def pick_date_range(year: int, lat: float, spec: ChipSpec) -> str:
    if lat >= 0:
        start = f"{year}-{spec.year_windows_north[0]}"
        end = f"{year}-{spec.year_windows_north[1]}"
    else:
        start = f"{year}-{spec.year_windows_south[0]}"
        end = f"{year+1}-{spec.year_windows_south[1]}"
    return f"{start}/{end}"


def stac_search_items(catalog: Client, collections: List[str], bbox_wgs84: List[float], datetime_range: str, limit: int = 200):
    search = catalog.search(
        collections=collections,
        bbox=bbox_wgs84,
        datetime=datetime_range,
        max_items=limit,
    )
    return list(search.items())


def get_cloud_cover(item) -> float:
    cc = item.properties.get("eo:cloud_cover")
    if cc is None:
        cc = item.properties.get("cloud_cover")
    if cc is None:
        return 9999.0
    try:
        return float(cc)
    except Exception:
        return 9999.0


def compute_cloud_fraction_from_fmask(fmask: np.ndarray) -> float:
    f = np.nan_to_num(fmask.astype(np.float32), nan=255.0).astype(np.uint16)
    valid = f != 255
    if valid.sum() == 0:
        return 1.0
    bad = (f == 4) | (f == 2) | (f == 3)
    return float((bad & valid).sum() / valid.sum())


def save_npz(out_path: str, arr: np.ndarray, meta: Dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, x=arr.astype(np.float32), bands=np.array(BANDS), meta=json.dumps(meta))


def ensure_earthaccess_login():
    earthaccess.login(persist=True)


def download_item_assets_locally(item, cache_dir: str) -> Optional[Dict[str, str]]:
    os.makedirs(cache_dir, exist_ok=True)
    hrefs = []
    for b in BANDS:
        a = item.assets.get(b)
        if a is None:
            return None
        hrefs.append(a.href)

    local_paths = earthaccess.download(hrefs, local_path=cache_dir)
    if not local_paths:
        return None

    band_to_local = {}
    for b, lp in zip(BANDS, local_paths):
        if not lp or not os.path.exists(lp):
            return None
        band_to_local[b] = str(Path(lp).resolve())
    return band_to_local


def read_chip_from_cached_bands(band_to_local: Dict[str, str], lat: float, lon: float, spec: ChipSpec) -> Optional[Dict]:
    b0 = band_to_local["B02"]
    with rasterio.open(b0) as src0:
        dst_crs = src0.crs
        if dst_crs is None:
            return None

        tf = Transformer.from_crs(CRS.from_epsg(4326), dst_crs, always_xy=True)
        x, y = tf.transform(lon, lat)
        if not np.isfinite(x) or not np.isfinite(y):
            return None

        px = abs(src0.transform.a)
        py = abs(src0.transform.e)

        target_px = float(spec.resolution)
        out_size = int(round(spec.chip_meters / target_px))
        if out_size <= 0:
            return None

        col, row = src0.index(x, y)

        half = out_size / 2.0
        win = Window(col - half, row - half, out_size, out_size)

        out = np.zeros((out_size, out_size, len(BANDS)), dtype=np.float32)

        for i, b in enumerate(BANDS):
            path = band_to_local[b]
            with rasterio.open(path) as src:
                resamp = Resampling.nearest if b == "Fmask" else Resampling.bilinear
                data = src.read(
                    1,
                    window=win,
                    out_shape=(out_size, out_size),
                    boundless=True,
                    fill_value=np.nan,
                    resampling=resamp,
                ).astype(np.float32)

                if b != "Fmask":
                    data = data / 10000.0

                out[..., i] = data

        fmask = out[..., BANDS.index("Fmask")]
        cloud_frac = compute_cloud_fraction_from_fmask(fmask)

        return {
            "stack": out,
            "cloud_frac": float(cloud_frac),
            "tile_crs": str(dst_crs),
            "tile_px": float(px),
            "tile_py": float(py),
            "center_xy": [float(x), float(y)],
            "out_size": int(out_size),
        }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--locations_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--years", default="2019,2021,2023")
    p.add_argument("--collections", default="HLSS30_2.0")
    p.add_argument("--max_cloud_frac", type=float, default=0.20)
    p.add_argument("--chip_meters", type=float, default=7680.0)
    p.add_argument("--resolution", type=float, default=30.0)
    p.add_argument("--max_locations", type=int, default=0)
    args = p.parse_args()

    ensure_earthaccess_login()

    spec = ChipSpec(
        chip_meters=args.chip_meters,
        resolution=args.resolution,
        max_cloud_frac=args.max_cloud_frac,
    )

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    collections = [c.strip() for c in args.collections.split(",") if c.strip()]

    df = pd.read_csv(args.locations_csv)
    for col in ["id", "name", "lat", "lon"]:
        if col not in df.columns:
            raise RuntimeError(f"locations_csv missing column: {col}")

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    if args.max_locations and args.max_locations > 0:
        df = df.head(args.max_locations)

    catalog = Client.open(STAC_URL)

    saved = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Locations"):
        loc_id = str(row["id"])
        name = str(row["name"])
        lat = float(row["lat"])
        lon = float(row["lon"])

        half = spec.chip_meters / 2.0
        bbox_search = bbox_wgs84_for_search(lat, lon, half)

        for year in years:
            out_path = os.path.join(args.out_dir, loc_id, f"{year}.npz")
            if os.path.exists(out_path):
                continue

            date_range = pick_date_range(year, lat, spec)
            items = stac_search_items(catalog, collections, bbox_search, date_range, limit=200)
            if not items:
                continue

            items_sorted = sorted(items, key=get_cloud_cover)

            best = None
            best_cf = 1e9
            best_meta = None

            for item in items_sorted[:spec.max_items_to_try]:
                item_cache = os.path.join(args.cache_dir, item.id)
                band_to_local = download_item_assets_locally(item, item_cache)
                if band_to_local is None:
                    continue

                res = read_chip_from_cached_bands(band_to_local, lat, lon, spec)
                if res is None:
                    continue

                cf = float(res["cloud_frac"])
                meta = {
                    "location_id": loc_id,
                    "location_name": name,
                    "lat": lat,
                    "lon": lon,
                    "year": year,
                    "datetime_range": date_range,
                    "collections": collections,
                    "item_id": item.id,
                    "item_datetime": str(item.datetime),
                    "eo_cloud_cover": item.properties.get("eo:cloud_cover"),
                    "cloud_frac_fmask_valid": cf,
                    "tile_crs": res["tile_crs"],
                    "tile_px": res["tile_px"],
                    "tile_py": res["tile_py"],
                    "center_xy": res["center_xy"],
                    "out_size": res["out_size"],
                }

                if cf < best_cf:
                    best_cf = cf
                    best = res["stack"]
                    best_meta = dict(meta)

                if cf <= spec.max_cloud_frac:
                    save_npz(out_path, res["stack"], meta)
                    saved += 1
                    best = None
                    best_meta = None
                    break

            if (not os.path.exists(out_path)) and (best is not None):
                best_meta["note"] = "No item met max_cloud_frac; best available selected."
                save_npz(out_path, best, best_meta)
                saved += 1

    print("saved_npz_files:", saved)


if __name__ == "__main__":
    main()

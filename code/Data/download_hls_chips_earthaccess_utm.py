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
from odc.stac import stac_load
from pyproj import CRS, Transformer

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


def utm_epsg_from_latlon(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def bbox_wgs84_for_search(lat: float, lon: float, half_size_m: float) -> List[float]:
    lat_deg_per_m = 1.0 / 111_320.0
    lon_deg_per_m = 1.0 / (111_320.0 * math.cos(math.radians(lat)) + 1e-9)
    dlat = half_size_m * lat_deg_per_m
    dlon = half_size_m * lon_deg_per_m
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def bbox_utm_from_latlon(lat: float, lon: float, half_size_m: float, epsg: int) -> List[float]:
    tf = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    x, y = tf.transform(lon, lat)
    return [x - half_size_m, y - half_size_m, x + half_size_m, y + half_size_m]


def pick_date_range(year: int, lat: float, spec: ChipSpec) -> str:
    if lat >= 0:
        start = f"{year}-{spec.year_windows_north[0]}"
        end = f"{year}-{spec.year_windows_north[1]}"
    else:
        start = f"{year}-{spec.year_windows_south[0]}"
        end = f"{year+1}-{spec.year_windows_south[1]}"
    return f"{start}/{end}"


def stac_search_items(
    catalog: Client,
    collections: List[str],
    bbox_wgs84: List[float],
    datetime_range: str,
    limit: int = 200,
):
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


def stac_item_with_local_assets(item, band_to_local: Dict[str, str]):
    for b in BANDS:
        item.assets[b].href = str(Path(band_to_local[b]).resolve())  # plain path (not file://)
    return item


def load_chip_from_local_item(item, bbox_utm, epsg: int, spec: ChipSpec) -> Optional[Dict]:
    try:
        ds = stac_load(
            [item],
            bands=BANDS,
            bbox=bbox_utm,
            crs=f"EPSG:{epsg}",
            resolution=spec.resolution,
            groupby="solar_day",
            chunks={},
        )
        if ds is None or len(ds.data_vars) == 0:
            return None

        if "time" in ds.dims and ds.sizes.get("time", 0) >= 1:
            ds = ds.isel(time=0)

        stack = np.stack([ds[b].values for b in BANDS], axis=-1)

        for i, b in enumerate(BANDS):
            if b != "Fmask":
                stack[..., i] = stack[..., i] / 10000.0

        cloud_frac = compute_cloud_fraction_from_fmask(stack[..., BANDS.index("Fmask")])
        return {"stack": stack, "cloud_frac": cloud_frac}

    except Exception as e:
        print("stac_load failed:", type(e).__name__, str(e)[:400])
        return None


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

    if args.max_locations and args.max_locations > 0:
        df = df.head(args.max_locations)

    catalog = Client.open(STAC_URL)

    saved = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Locations"):
        loc_id = str(row["id"])
        name = str(row["name"])
        lat = float(row["lat"])
        lon = float(row["lon"])

        epsg = utm_epsg_from_latlon(lat, lon)
        half = spec.chip_meters / 2.0
        bbox_search = bbox_wgs84_for_search(lat, lon, half)
        bbox_utm = bbox_utm_from_latlon(lat, lon, half, epsg)

        for year in years:
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

                local_item = stac_item_with_local_assets(item, band_to_local)
                res = load_chip_from_local_item(local_item, bbox_utm, epsg, spec)
                if res is None:
                    continue

                cf = float(res["cloud_frac"])
                meta = {
                    "location_id": loc_id,
                    "location_name": name,
                    "lat": lat,
                    "lon": lon,
                    "epsg": epsg,
                    "bbox_wgs84_search": bbox_search,
                    "bbox_utm": bbox_utm,
                    "year": year,
                    "datetime_range": date_range,
                    "collections": collections,
                    "item_id": item.id,
                    "item_datetime": str(item.datetime),
                    "eo_cloud_cover": item.properties.get("eo:cloud_cover"),
                    "cloud_frac_fmask_valid": cf,
                }

                if cf < best_cf:
                    best_cf = cf
                    best = res["stack"]
                    best_meta = dict(meta)

                if cf <= spec.max_cloud_frac:
                    out_path = os.path.join(args.out_dir, loc_id, f"{year}.npz")
                    save_npz(out_path, res["stack"], meta)
                    saved += 1
                    break

            out_path = os.path.join(args.out_dir, loc_id, f"{year}.npz")
            if (not os.path.exists(out_path)) and (best is not None):
                best_meta["note"] = "No item met max_cloud_frac; best available selected."
                save_npz(out_path, best, best_meta)
                saved += 1

    print("saved_npz_files:", saved)


if __name__ == "__main__":
    main()

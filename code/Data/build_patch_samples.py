import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def list_location_dirs(chips_root: Path):
    return sorted([p for p in chips_root.iterdir() if p.is_dir()])


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    x = d["x"]
    bands = list(d["bands"])
    meta = json.loads(str(d["meta"]))
    return x, bands, meta


def patchify(arr: np.ndarray, patch: int):
    h, w, c = arr.shape
    h2 = (h // patch) * patch
    w2 = (w // patch) * patch
    arr = arr[:h2, :w2, :]
    ph = h2 // patch
    pw = w2 // patch
    patches = arr.reshape(ph, patch, pw, patch, c).transpose(0, 2, 1, 3, 4).reshape(ph * pw, patch, patch, c)
    return patches, (ph, pw)


def cloud_and_valid_from_fmask(fmask_patch: np.ndarray):
    f = fmask_patch.astype(np.float32)
    f = np.nan_to_num(f, nan=255.0).astype(np.uint16)

    valid = f != 255
    valid_frac = float(valid.mean())

    if valid_frac == 0.0:
        return 1.0, 0.0

    bad = (f == 4) | (f == 2) | (f == 3)
    cloud_frac_valid = float((bad & valid).sum() / valid.sum())
    return cloud_frac_valid, valid_frac


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chips_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--years", default="2019,2021,2023")
    p.add_argument("--patch", type=int, default=64)
    p.add_argument("--max_cloud", type=float, default=0.10)
    p.add_argument("--min_valid", type=float, default=0.70)
    p.add_argument("--require_years", default="2019,2023")
    p.add_argument("--mode", choices=["pair", "triplet"], default="pair")
    p.add_argument("--max_locations", type=int, default=0)
    args = p.parse_args()

    chips_root = Path(args.chips_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    samples_dir = out_root / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    req_years = [int(y.strip()) for y in args.require_years.split(",") if y.strip()]

    if args.mode == "triplet":
        if len(years) != 3:
            raise RuntimeError("triplet mode expects --years like 2019,2021,2023")
        req_years = years

    loc_dirs = list_location_dirs(chips_root)
    if args.max_locations and args.max_locations > 0:
        loc_dirs = loc_dirs[: args.max_locations]

    rows = []
    sample_idx = 0

    cnt_locations = 0
    cnt_missing_req = 0
    cnt_band_mismatch = 0
    cnt_no_fmask = 0
    cnt_grid_mismatch = 0
    cnt_total_patches = 0
    cnt_drop_low_valid = 0
    cnt_drop_cloud = 0
    cnt_saved = 0

    for loc_dir in tqdm(loc_dirs, desc="Locations"):
        cnt_locations += 1
        loc_id = loc_dir.name

        year_to_path = {y: (loc_dir / f"{y}.npz") for y in years}
        if any(not year_to_path[y].exists() for y in req_years):
            cnt_missing_req += 1
            continue

        loaded = {}
        bands_ref = None
        meta_ref = None

        ok = True
        for y in req_years:
            try:
                x, bands, meta = load_npz(year_to_path[y])
            except Exception:
                ok = False
                break
            if bands_ref is None:
                bands_ref = bands
                meta_ref = meta
            else:
                if bands != bands_ref:
                    ok = False
                    break
            loaded[y] = x

        if not ok:
            cnt_band_mismatch += 1
            continue

        if "Fmask" not in bands_ref:
            cnt_no_fmask += 1
            continue

        fmask_idx = bands_ref.index("Fmask")

        patches_by_year = {}
        grid_shape = None
        for y in req_years:
            patches, gs = patchify(loaded[y], args.patch)
            if grid_shape is None:
                grid_shape = gs
            else:
                if gs != grid_shape:
                    ok = False
                    break
            patches_by_year[y] = patches

        if not ok:
            cnt_grid_mismatch += 1
            continue

        n_patches = patches_by_year[req_years[0]].shape[0]
        cnt_total_patches += n_patches

        for pid in range(n_patches):
            clouds = {}
            valids = {}

            for y in req_years:
                fmask_patch = patches_by_year[y][pid, :, :, fmask_idx]
                cf, vf = cloud_and_valid_from_fmask(fmask_patch)
                clouds[y] = cf
                valids[y] = vf

            if any(valids[y] < args.min_valid for y in req_years):
                cnt_drop_low_valid += 1
                continue

            if any(clouds[y] > args.max_cloud for y in req_years):
                cnt_drop_cloud += 1
                continue

            if args.mode == "pair":
                y0, y1 = req_years[0], req_years[1]
                x0 = patches_by_year[y0][pid].astype(np.float32)
                x1 = patches_by_year[y1][pid].astype(np.float32)

                out_path = samples_dir / f"{sample_idx:07d}.npz"
                np.savez_compressed(
                    out_path,
                    x_a=x0,
                    x_b=x1,
                    bands=np.array(bands_ref),
                    meta=json.dumps(
                        {
                            "location_id": loc_id,
                            "patch_id": pid,
                            "year_a": y0,
                            "year_b": y1,
                            "cloud_a": float(clouds[y0]),
                            "cloud_b": float(clouds[y1]),
                            "valid_a": float(valids[y0]),
                            "valid_b": float(valids[y1]),
                            "source_meta": meta_ref,
                            "grid_ph": grid_shape[0],
                            "grid_pw": grid_shape[1],
                            "patch": args.patch,
                        }
                    ),
                )

                rows.append(
                    {
                        "sample_id": sample_idx,
                        "location_id": loc_id,
                        "patch_id": pid,
                        "year_a": y0,
                        "year_b": y1,
                        "cloud_a": float(clouds[y0]),
                        "cloud_b": float(clouds[y1]),
                        "valid_a": float(valids[y0]),
                        "valid_b": float(valids[y1]),
                        "grid_ph": grid_shape[0],
                        "grid_pw": grid_shape[1],
                        "patch": args.patch,
                        "path": str(out_path),
                    }
                )

                sample_idx += 1
                cnt_saved += 1

            else:
                y0, y1, y2 = years
                x0 = patches_by_year[y0][pid].astype(np.float32)
                x1 = patches_by_year[y1][pid].astype(np.float32)
                x2 = patches_by_year[y2][pid].astype(np.float32)

                out_path = samples_dir / f"{sample_idx:07d}.npz"
                np.savez_compressed(
                    out_path,
                    x_0=x0,
                    x_1=x1,
                    x_2=x2,
                    bands=np.array(bands_ref),
                    meta=json.dumps(
                        {
                            "location_id": loc_id,
                            "patch_id": pid,
                            "years": [y0, y1, y2],
                            "clouds": [float(clouds[y0]), float(clouds[y1]), float(clouds[y2])],
                            "valids": [float(valids[y0]), float(valids[y1]), float(valids[y2])],
                            "source_meta": meta_ref,
                            "grid_ph": grid_shape[0],
                            "grid_pw": grid_shape[1],
                            "patch": args.patch,
                        }
                    ),
                )

                rows.append(
                    {
                        "sample_id": sample_idx,
                        "location_id": loc_id,
                        "patch_id": pid,
                        "year_0": y0,
                        "year_1": y1,
                        "year_2": y2,
                        "cloud_0": float(clouds[y0]),
                        "cloud_1": float(clouds[y1]),
                        "cloud_2": float(clouds[y2]),
                        "valid_0": float(valids[y0]),
                        "valid_1": float(valids[y1]),
                        "valid_2": float(valids[y2]),
                        "grid_ph": grid_shape[0],
                        "grid_pw": grid_shape[1],
                        "patch": args.patch,
                        "path": str(out_path),
                    }
                )

                sample_idx += 1
                cnt_saved += 1

    df = pd.DataFrame(rows)
    manifest_path = out_root / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    print("chips_root:", str(chips_root))
    print("out_root:", str(out_root))
    print("samples_dir:", str(samples_dir))
    print("manifest:", str(manifest_path))
    print("n_samples:", int(cnt_saved))
    print("locations_seen:", int(cnt_locations))
    print("locations_missing_required_years:", int(cnt_missing_req))
    print("locations_band_mismatch_or_load_fail:", int(cnt_band_mismatch))
    print("locations_no_fmask:", int(cnt_no_fmask))
    print("locations_grid_mismatch:", int(cnt_grid_mismatch))
    print("total_patches_considered:", int(cnt_total_patches))
    print("dropped_low_valid:", int(cnt_drop_low_valid))
    print("dropped_cloud:", int(cnt_drop_cloud))


if __name__ == "__main__":
    main()
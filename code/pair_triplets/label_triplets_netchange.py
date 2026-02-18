import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def safe_ndvi(nir: np.ndarray, red: np.ndarray):
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red
    return (nir - red) / np.where(np.abs(denom) < 1e-6, np.nan, denom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--delta_thr", type=float, default=0.05)
    p.add_argument("--max_samples", type=int, default=0)
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    if args.max_samples and args.max_samples > 0:
        df = df.head(args.max_samples).copy()

    thr = float(args.delta_thr)
    rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Labeling triplets (net change)"):
        d = np.load(r["path"], allow_pickle=True)
        bands = list(d["bands"])
        if ("B08" not in bands) or ("B04" not in bands):
            continue

        b_nir = bands.index("B08")
        b_red = bands.index("B04")

        x0 = d["x_0"].astype(np.float32)  # 2019
        x1 = d["x_1"].astype(np.float32)  # 2021
        x2 = d["x_2"].astype(np.float32)  # 2023

        m0 = float(np.nanmean(safe_ndvi(x0[..., b_nir], x0[..., b_red])))
        m1 = float(np.nanmean(safe_ndvi(x1[..., b_nir], x1[..., b_red])))
        m2 = float(np.nanmean(safe_ndvi(x2[..., b_nir], x2[..., b_red])))

        d02 = m2 - m0
        if d02 > thr:
            label = 2
        elif d02 < -thr:
            label = 0
        else:
            label = 1

        rows.append({
            "sample_id": int(r["sample_id"]),
            "location_id": r["location_id"],
            "patch_id": int(r["patch_id"]),
            "ndvi_2019": m0,
            "ndvi_2021": m1,
            "ndvi_2023": m2,
            "delta_2019_2023": float(d02),
            "label": int(label),
            "path": r["path"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    print("wrote:", args.out_csv)
    print("n:", len(out))
    print("label_counts:", out["label"].value_counts().sort_index().to_dict())
    print("delta_thr:", thr)


if __name__ == "__main__":
    main()

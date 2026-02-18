import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def safe_ndvi(nir: np.ndarray, red: np.ndarray):
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red
    ndvi = (nir - red) / np.where(np.abs(denom) < 1e-6, np.nan, denom)
    return ndvi


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

    rows = []
    thr = float(args.delta_thr)

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Labeling"):
        path = Path(r["path"])
        d = np.load(path, allow_pickle=True)
        bands = list(d["bands"])

        if ("B08" not in bands) or ("B04" not in bands):
            continue

        b_nir = bands.index("B08")
        b_red = bands.index("B04")

        x_a = d["x_a"]  # (patch,patch,C)
        x_b = d["x_b"]

        ndvi_a = safe_ndvi(x_a[..., b_nir], x_a[..., b_red])
        ndvi_b = safe_ndvi(x_b[..., b_nir], x_b[..., b_red])

        m_a = float(np.nanmean(ndvi_a))
        m_b = float(np.nanmean(ndvi_b))
        delta = m_b - m_a

        if delta > thr:
            label = 2  # up
        elif delta < -thr:
            label = 0  # down
        else:
            label = 1  # flat

        rows.append(
            {
                "sample_id": int(r["sample_id"]),
                "location_id": r["location_id"],
                "patch_id": int(r["patch_id"]),
                "year_a": int(r["year_a"]),
                "year_b": int(r["year_b"]),
                "ndvi_a": m_a,
                "ndvi_b": m_b,
                "delta_ndvi": float(delta),
                "label": int(label),
                "path": str(path),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    counts = out["label"].value_counts().to_dict()
    print("wrote:", args.out_csv)
    print("n:", len(out))
    print("label_counts:", counts)
    print("delta_thr:", thr)


if __name__ == "__main__":
    main()

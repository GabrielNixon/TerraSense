import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LABEL_NAME = {0: "DOWN", 1: "STABLE", 2: "UP"}


def safe_ndvi(nir, red):
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red
    return (nir - red) / np.where(np.abs(denom) < 1e-6, np.nan, denom)


def norm01(img, lo=2, hi=98):
    x = img.astype(np.float32)
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    if not np.isfinite(a) or not np.isfinite(b) or abs(b - a) < 1e-6:
        return np.clip(x, 0, 1)
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1)


def make_rgb(x, bands):
    # H,W,C
    b2 = bands.index("B02")  # blue
    b3 = bands.index("B03")  # green
    b4 = bands.index("B04")  # red
    rgb = np.stack([x[..., b4], x[..., b3], x[..., b2]], axis=-1)
    return norm01(rgb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", required=True, help="e.g., .../labels_triplet_netchange_split.csv")
    p.add_argument("--out_png", default="figures/triplet_example.png")
    p.add_argument("--pick", choices=["first", "by_id", "by_location", "by_label"], default="by_label")
    p.add_argument("--sample_id", type=int, default=-1)
    p.add_argument("--location_id", default="")
    p.add_argument("--label", type=int, default=2, help="0=DOWN, 1=STABLE, 2=UP (used for pick=by_label)")
    args = p.parse_args()

    df = pd.read_csv(args.labels_csv)

    if args.pick == "by_id":
        if args.sample_id < 0:
            raise RuntimeError("--sample_id required for pick=by_id")
        row = df[df["sample_id"] == args.sample_id].iloc[0]
    elif args.pick == "by_location":
        if not args.location_id:
            raise RuntimeError("--location_id required for pick=by_location")
        row = df[df["location_id"] == args.location_id].iloc[0]
    elif args.pick == "by_label":
        sub = df[df["label"] == args.label]
        if len(sub) == 0:
            raise RuntimeError(f"No samples found with label={args.label}")
        row = sub.iloc[0]
    else:
        row = df.iloc[0]

    path = row["path"]
    d = np.load(path, allow_pickle=True)
    bands = list(d["bands"])

    x0 = d["x_0"].astype(np.float32)  # 2019
    x1 = d["x_1"].astype(np.float32)  # 2021
    x2 = d["x_2"].astype(np.float32)  # 2023

    # NDVI
    bnir = bands.index("B08")
    bred = bands.index("B04")
    nd0 = safe_ndvi(x0[..., bnir], x0[..., bred])
    nd1 = safe_ndvi(x1[..., bnir], x1[..., bred])
    nd2 = safe_ndvi(x2[..., bnir], x2[..., bred])

    m0 = float(np.nanmean(nd0))
    m1 = float(np.nanmean(nd1))
    m2 = float(np.nanmean(nd2))
    d02 = m2 - m0

    # RGB
    rgb0 = make_rgb(x0, bands)
    rgb1 = make_rgb(x1, bands)
    rgb2 = make_rgb(x2, bands)

    # delta maps
    dnd = nd2 - nd0
    dm = np.linalg.norm((x2 - x0)[..., :6], axis=-1)  # magnitude using 6 reflectance bands (ignore Fmask)

    # Title/meta
    loc_id = row["location_id"]
    patch_id = int(row["patch_id"])
    sample_id = int(row["sample_id"])
    label = int(row["label"])
    label_name = LABEL_NAME.get(label, str(label))

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Plot grid: 3 RGB + 3 NDVI + delta NDVI + diff magnitude
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

    axes[0, 0].imshow(rgb0); axes[0, 0].set_title("RGB 2019"); axes[0, 0].axis("off")
    axes[0, 1].imshow(rgb1); axes[0, 1].set_title("RGB 2021"); axes[0, 1].axis("off")
    axes[0, 2].imshow(rgb2); axes[0, 2].set_title("RGB 2023"); axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.0, 0.9,
        f"sample_id: {sample_id}\nlocation_id: {loc_id}\npatch_id: {patch_id}\nlabel: {label} ({label_name})\n"
        f"mean NDVI: 2019={m0:.3f}, 2021={m1:.3f}, 2023={m2:.3f}\n"
        f"ΔNDVI (2023-2019)={d02:.3f}\n"
        f"path:\n{path}",
        fontsize=10, va="top"
    )

    im = axes[1, 0].imshow(nd0, vmin=-1, vmax=1); axes[1, 0].set_title("NDVI 2019"); axes[1, 0].axis("off")
    axes[1, 1].imshow(nd1, vmin=-1, vmax=1); axes[1, 1].set_title("NDVI 2021"); axes[1, 1].axis("off")
    axes[1, 2].imshow(nd2, vmin=-1, vmax=1); axes[1, 2].set_title("NDVI 2023"); axes[1, 2].axis("off")

    axes[1, 3].imshow(dnd, vmin=-1, vmax=1)
    axes[1, 3].set_title("ΔNDVI (2023-2019)")
    axes[1, 3].axis("off")

    # Also save a second figure for diff magnitude (optional but nice)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    ax2.imshow(dm)
    ax2.set_title("Change magnitude ‖x2023 − x2019‖")
    ax2.axis("off")

    fig.suptitle("Triplet explanation: same location across time", fontsize=14)
    fig.savefig(out_png, dpi=200)
    fig2.savefig(out_png.with_name(out_png.stem + "_diffmag.png"), dpi=200)

    print("Saved:", str(out_png))
    print("Saved:", str(out_png.with_name(out_png.stem + "_diffmag.png")))
    print(f"sample_id={sample_id} location_id={loc_id} patch_id={patch_id} label={label}({label_name}) ΔNDVI={d02:.3f}")


if __name__ == "__main__":
    main()

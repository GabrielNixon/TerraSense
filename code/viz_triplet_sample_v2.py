import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABEL_NAME = {0: "DOWN", 1: "STABLE", 2: "UP"}

BAD_FMASK = {2, 3, 4, 255}  # shadow, snow, cloud, fill


def safe_ndvi(nir, red):
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red
    return (nir - red) / np.where(np.abs(denom) < 1e-6, np.nan, denom)


def mask_from_fmask(x, bands, mask_cloud=True):
    if "Fmask" not in bands:
        return np.ones(x.shape[:2], dtype=bool)

    f = x[..., bands.index("Fmask")]
    f = np.nan_to_num(f, nan=255.0).astype(np.uint16)

    valid = f != 255
    if not mask_cloud:
        return valid

    bad = np.zeros_like(valid, dtype=bool)
    for v in BAD_FMASK:
        bad |= (f == v)
    return valid & (~bad)


def robust_stretch_shared(rgbs, lo=2, hi=98, gamma=1.25):
    stack = np.concatenate([r.reshape(-1, 3) for r in rgbs], axis=0).astype(np.float32)
    a = np.nanpercentile(stack, lo, axis=0)
    b = np.nanpercentile(stack, hi, axis=0)
    b = np.where(np.abs(b - a) < 1e-6, a + 1e-6, b)

    out = []
    for rgb in rgbs:
        x = (rgb - a) / (b - a)
        x = np.clip(x, 0, 1)
        if gamma and gamma != 1.0:
            x = x ** (1.0 / gamma)
        out.append(x)
    return out


def make_rgb(x, bands):
    b2 = bands.index("B02")
    b3 = bands.index("B03")
    b4 = bands.index("B04")
    rgb = np.stack([x[..., b4], x[..., b3], x[..., b2]], axis=-1).astype(np.float32)
    return rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--out_png", default="figures/triplet_example_v2.png")
    p.add_argument("--pick", choices=["by_id", "by_location", "by_label"], default="by_label")
    p.add_argument("--sample_id", type=int, default=-1)
    p.add_argument("--location_id", default="")
    p.add_argument("--label", type=int, default=2)
    p.add_argument("--mask_cloud", action="store_true")
    p.add_argument("--dndvi_clip", type=float, default=0.30)
    p.add_argument("--diff_clip_pct", type=float, default=99.0)
    args = p.parse_args()

    df = pd.read_csv(args.labels_csv)

    if args.pick == "by_id":
        row = df[df["sample_id"] == args.sample_id].iloc[0]
    elif args.pick == "by_location":
        row = df[df["location_id"] == args.location_id].iloc[0]
    else:
        row = df[df["label"] == args.label].iloc[0]

    d = np.load(row["path"], allow_pickle=True)
    bands = list(d["bands"])

    x0 = d["x_0"].astype(np.float32)
    x1 = d["x_1"].astype(np.float32)
    x2 = d["x_2"].astype(np.float32)

    m0 = mask_from_fmask(x0, bands, mask_cloud=args.mask_cloud)
    m1 = mask_from_fmask(x1, bands, mask_cloud=args.mask_cloud)
    m2 = mask_from_fmask(x2, bands, mask_cloud=args.mask_cloud)
    m_all = m0 & m1 & m2  # only show pixels valid in all years

    # RGB (shared stretch)
    rgb0 = make_rgb(x0, bands)
    rgb1 = make_rgb(x1, bands)
    rgb2 = make_rgb(x2, bands)
    rgbs = robust_stretch_shared([rgb0, rgb1, rgb2], lo=2, hi=98, gamma=1.25)
    rgb0s, rgb1s, rgb2s = rgbs

    # apply mask (set invalid to white so it’s obvious)
    for rgb in (rgb0s, rgb1s, rgb2s):
        rgb[~m_all] = 1.0

    # NDVI maps (masked)
    bnir = bands.index("B08")
    bred = bands.index("B04")
    nd0 = safe_ndvi(x0[..., bnir], x0[..., bred])
    nd1 = safe_ndvi(x1[..., bnir], x1[..., bred])
    nd2 = safe_ndvi(x2[..., bnir], x2[..., bred])

    nd0[~m_all] = np.nan
    nd1[~m_all] = np.nan
    nd2[~m_all] = np.nan

    dnd = nd2 - nd0
    dnd[~m_all] = np.nan

    mean0 = float(np.nanmean(nd0))
    mean1 = float(np.nanmean(nd1))
    mean2 = float(np.nanmean(nd2))
    d02 = mean2 - mean0

    # change magnitude: mean abs change over 6 reflectance bands (ignore Fmask)
    diff = (x2 - x0)[..., :6]
    mag = np.nanmean(np.abs(diff), axis=-1)
    mag[~m_all] = np.nan
    vmax_mag = float(np.nanpercentile(mag[np.isfinite(mag)], args.diff_clip_pct)) if np.isfinite(mag).any() else 1.0

    # Plot
    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 6)

    ax_rgb0 = fig.add_subplot(gs[0, 0:2])
    ax_rgb1 = fig.add_subplot(gs[0, 2:4])
    ax_rgb2 = fig.add_subplot(gs[0, 4:6])

    ax_nd0 = fig.add_subplot(gs[1, 0])
    ax_nd1 = fig.add_subplot(gs[1, 1])
    ax_nd2 = fig.add_subplot(gs[1, 2])
    ax_dnd = fig.add_subplot(gs[1, 3])
    ax_mag = fig.add_subplot(gs[1, 4])
    ax_txt = fig.add_subplot(gs[1, 5])
    ax_txt.axis("off")

    ax_rgb0.imshow(rgb0s); ax_rgb0.set_title("RGB 2019"); ax_rgb0.axis("off")
    ax_rgb1.imshow(rgb1s); ax_rgb1.set_title("RGB 2021"); ax_rgb1.axis("off")
    ax_rgb2.imshow(rgb2s); ax_rgb2.set_title("RGB 2023"); ax_rgb2.axis("off")

    im0 = ax_nd0.imshow(nd0, vmin=-0.2, vmax=0.9, cmap="viridis"); ax_nd0.set_title("NDVI 2019"); ax_nd0.axis("off")
    im1 = ax_nd1.imshow(nd1, vmin=-0.2, vmax=0.9, cmap="viridis"); ax_nd1.set_title("NDVI 2021"); ax_nd1.axis("off")
    im2 = ax_nd2.imshow(nd2, vmin=-0.2, vmax=0.9, cmap="viridis"); ax_nd2.set_title("NDVI 2023"); ax_nd2.axis("off")

    clip = float(args.dndvi_clip)
    im3 = ax_dnd.imshow(dnd, vmin=-clip, vmax=clip, cmap="RdBu_r")
    ax_dnd.set_title("ΔNDVI (2023−2019)")
    ax_dnd.axis("off")

    im4 = ax_mag.imshow(mag, vmin=0.0, vmax=vmax_mag, cmap="magma")
    ax_mag.set_title("|Δreflectance| (mean abs)")
    ax_mag.axis("off")

    # Colorbars (small but useful)
    cbar1 = fig.colorbar(im3, ax=ax_dnd, fraction=0.046, pad=0.04)
    cbar1.set_label("ΔNDVI")
    cbar2 = fig.colorbar(im4, ax=ax_mag, fraction=0.046, pad=0.04)
    cbar2.set_label("mean abs Δ")

    loc_id = row["location_id"]
    patch_id = int(row["patch_id"])
    sample_id = int(row["sample_id"])
    label = int(row["label"])
    label_name = LABEL_NAME.get(label, str(label))

    ax_txt.text(
        0.0, 1.0,
        f"sample_id: {sample_id}\n"
        f"location_id: {loc_id}\n"
        f"patch_id: {patch_id}\n"
        f"label: {label} ({label_name})\n\n"
        f"mean NDVI:\n"
        f"  2019 = {mean0:.3f}\n"
        f"  2021 = {mean1:.3f}\n"
        f"  2023 = {mean2:.3f}\n"
        f"ΔNDVI (2023−2019) = {d02:.3f}\n\n"
        f"mask_cloud = {args.mask_cloud}\n"
        f"ΔNDVI clip = ±{clip}\n"
        f"mag clip pct = {args.diff_clip_pct}",
        va="top", fontsize=11
    )

    fig.suptitle("Triplet example: same location across time (shared stretch + masking)", fontsize=14)
    fig.savefig(out_png, dpi=220)
    print("Saved:", str(out_png))


if __name__ == "__main__":
    main()

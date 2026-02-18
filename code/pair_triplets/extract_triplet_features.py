import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


BANDS6 = ["B02","B03","B04","B08","B11","B12"]


def safe_ndvi(nir, red):
    nir = nir.astype(np.float32)
    red = red.astype(np.float32)
    denom = nir + red
    return (nir - red) / np.where(np.abs(denom) < 1e-6, np.nan, denom)


def fmask_valid(x, bands):
    if "Fmask" not in bands:
        return np.ones(x.shape[:2], dtype=bool)
    f = x[..., bands.index("Fmask")]
    f = np.nan_to_num(f, nan=255.0).astype(np.uint16)
    valid = f != 255
    bad = (f == 2) | (f == 3) | (f == 4)  # shadow/snow/cloud
    return valid & (~bad)


def nanmean(a):
    return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan


def nanstd(a):
    return float(np.nanstd(a)) if np.isfinite(a).any() else np.nan


def gradient_mag(img):
    img = img.astype(np.float32)
    gy, gx = np.gradient(img)
    return np.sqrt(gx*gx + gy*gy)


def box_filter3(img):
    # 3x3 mean filter without scipy
    x = img.astype(np.float32)
    p = np.pad(x, ((1,1),(1,1)), mode="edge")
    s = (
        p[0:-2,0:-2] + p[0:-2,1:-1] + p[0:-2,2:] +
        p[1:-1,0:-2] + p[1:-1,1:-1] + p[1:-1,2:] +
        p[2:,0:-2]   + p[2:,1:-1]   + p[2:,2:]
    )
    return s / 9.0


def local_variance(img):
    m = box_filter3(img)
    m2 = box_filter3(img*img)
    v = m2 - m*m
    return v


def hist_entropy(img, bins=16, vmin=-0.2, vmax=0.9):
    x = img[np.isfinite(img)]
    if x.size == 0:
        return np.nan
    x = np.clip(x, vmin, vmax)
    h, _ = np.histogram(x, bins=bins, range=(vmin, vmax))
    p = h.astype(np.float32)
    p = p / (p.sum() + 1e-12)
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(ent)


def spatial_texture_features(nd):
    # nd: 2D NDVI with NaNs already
    feats = {}
    feats["nd_mean"] = nanmean(nd)
    feats["nd_std"] = nanstd(nd)

    gm = gradient_mag(np.nan_to_num(nd, nan=0.0))
    feats["grad_mean"] = nanmean(gm)
    feats["grad_std"] = nanstd(gm)

    lv = local_variance(np.nan_to_num(nd, nan=0.0))
    feats["lvar_mean"] = nanmean(lv)
    feats["lvar_std"] = nanstd(lv)

    feats["entropy16"] = hist_entropy(nd, bins=16)
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--mask_cloud", action="store_true")
    args = p.parse_args()

    df = pd.read_csv(args.labels_csv)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        d = np.load(r["path"], allow_pickle=True)
        bands = list(d["bands"])

        # years in your triplet files are x_0=2019, x_1=2021, x_2=2023 by construction
        x0 = d["x_0"].astype(np.float32)
        x1 = d["x_1"].astype(np.float32)
        x2 = d["x_2"].astype(np.float32)

        if ("B08" not in bands) or ("B04" not in bands):
            continue

        m0 = fmask_valid(x0, bands) if args.mask_cloud else np.ones(x0.shape[:2], dtype=bool)
        m1 = fmask_valid(x1, bands) if args.mask_cloud else np.ones(x1.shape[:2], dtype=bool)
        m2 = fmask_valid(x2, bands) if args.mask_cloud else np.ones(x2.shape[:2], dtype=bool)
        mall = m0 & m1 & m2

        # NDVI per year
        bnir = bands.index("B08")
        bred = bands.index("B04")
        nd0 = safe_ndvi(x0[..., bnir], x0[..., bred]); nd0[~mall] = np.nan
        nd1 = safe_ndvi(x1[..., bnir], x1[..., bred]); nd1[~mall] = np.nan
        nd2 = safe_ndvi(x2[..., bnir], x2[..., bred]); nd2[~mall] = np.nan

        # Temporal NDVI stats
        nd_mean0, nd_mean1, nd_mean2 = nanmean(nd0), nanmean(nd1), nanmean(nd2)
        d01 = nd_mean1 - nd_mean0 if np.isfinite(nd_mean0) and np.isfinite(nd_mean1) else np.nan
        d12 = nd_mean2 - nd_mean1 if np.isfinite(nd_mean1) and np.isfinite(nd_mean2) else np.nan
        d02 = nd_mean2 - nd_mean0 if np.isfinite(nd_mean0) and np.isfinite(nd_mean2) else np.nan
        curv = d12 - d01 if np.isfinite(d01) and np.isfinite(d12) else np.nan
        nd_std_t = float(np.nanstd([nd_mean0, nd_mean1, nd_mean2])) if np.isfinite([nd_mean0, nd_mean1, nd_mean2]).all() else np.nan

        # Spatial texture features (NDVI)
        tex0 = spatial_texture_features(nd0)
        tex1 = spatial_texture_features(nd1)
        tex2 = spatial_texture_features(nd2)

        # Reflectance temporal features (band means)
        band_feats = {}
        for b in BANDS6:
            if b not in bands:
                band_feats[f"{b}_m0"] = np.nan
                band_feats[f"{b}_m1"] = np.nan
                band_feats[f"{b}_m2"] = np.nan
                band_feats[f"{b}_d01"] = np.nan
                band_feats[f"{b}_d12"] = np.nan
                band_feats[f"{b}_d02"] = np.nan
                band_feats[f"{b}_curv"] = np.nan
                band_feats[f"{b}_tstd"] = np.nan
                continue
            bi = bands.index(b)
            v0 = x0[..., bi].copy(); v0[~mall] = np.nan
            v1 = x1[..., bi].copy(); v1[~mall] = np.nan
            v2 = x2[..., bi].copy(); v2[~mall] = np.nan
            m_0, m_1, m_2 = nanmean(v0), nanmean(v1), nanmean(v2)
            _d01 = m_1 - m_0 if np.isfinite(m_0) and np.isfinite(m_1) else np.nan
            _d12 = m_2 - m_1 if np.isfinite(m_1) and np.isfinite(m_2) else np.nan
            _d02 = m_2 - m_0 if np.isfinite(m_0) and np.isfinite(m_2) else np.nan
            _curv = _d12 - _d01 if np.isfinite(_d01) and np.isfinite(_d12) else np.nan
            _tstd = float(np.nanstd([m_0, m_1, m_2])) if np.isfinite([m_0, m_1, m_2]).all() else np.nan

            band_feats[f"{b}_m0"] = m_0
            band_feats[f"{b}_m1"] = m_1
            band_feats[f"{b}_m2"] = m_2
            band_feats[f"{b}_d01"] = _d01
            band_feats[f"{b}_d12"] = _d12
            band_feats[f"{b}_d02"] = _d02
            band_feats[f"{b}_curv"] = _curv
            band_feats[f"{b}_tstd"] = _tstd

        row_out = {
            "sample_id": int(r["sample_id"]),
            "location_id": r["location_id"],
            "patch_id": int(r["patch_id"]),
            "label": int(r["label"]),

            "nd_m0": nd_mean0,
            "nd_m1": nd_mean1,
            "nd_m2": nd_mean2,
            "nd_d01": d01,
            "nd_d12": d12,
            "nd_d02": d02,
            "nd_curv": curv,
            "nd_tstd": nd_std_t,
        }

        # texture per year + deltas
        for k, v in tex0.items():
            row_out[f"t0_{k}"] = v
        for k, v in tex1.items():
            row_out[f"t1_{k}"] = v
        for k, v in tex2.items():
            row_out[f"t2_{k}"] = v

        for k in tex0.keys():
            a, b_, c = row_out[f"t0_{k}"], row_out[f"t1_{k}"], row_out[f"t2_{k}"]
            row_out[f"td01_{k}"] = b_ - a if np.isfinite(a) and np.isfinite(b_) else np.nan
            row_out[f"td12_{k}"] = c - b_ if np.isfinite(b_) and np.isfinite(c) else np.nan
            row_out[f"td02_{k}"] = c - a if np.isfinite(a) and np.isfinite(c) else np.nan
            row_out[f"tcurv_{k}"] = (row_out[f"td12_{k}"] - row_out[f"td01_{k}"]) if np.isfinite(row_out[f"td01_{k}"]) and np.isfinite(row_out[f"td12_{k}"]) else np.nan

        row_out.update(band_feats)
        rows.append(row_out)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print("wrote:", args.out_csv)
    print("n:", len(out))
    print("label_counts:", out["label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()

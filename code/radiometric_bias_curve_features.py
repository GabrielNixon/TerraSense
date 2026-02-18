import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def cosine_distance(a, b, eps=1e-12):
    an = np.linalg.norm(a, axis=1) + eps
    bn = np.linalg.norm(b, axis=1) + eps
    sim = (a * b).sum(axis=1) / (an * bn)
    return 1.0 - sim

def pick_feature_cols(df, id_cols, label_col, split_col, feature_prefix, drop_cols):
    if feature_prefix:
        cols = [c for c in df.columns if c.startswith(feature_prefix)]
        if not cols:
            raise ValueError(f"No columns start with prefix '{feature_prefix}'")
        return cols

    blacklist = set(id_cols + [label_col, split_col, "path", "sample_id", "delta_2019_2023"])
    blacklist |= set(drop_cols)
    cols = [c for c in df.columns if c not in blacklist and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("No numeric feature columns found (try --feature_prefix)")
    return cols

def match_band_columns(cols, bands, strict=False):
    bands = [b.strip() for b in bands.split(",") if b.strip()]
    hits = []
    for c in cols:
        for b in bands:
            if strict:
                if c.startswith(b + "_") or ("_" + b + "_") in c:
                    hits.append(c)
            else:
                if b in c:
                    hits.append(c)
                    break
    return sorted(list(set(hits)))

def match_band_mean_columns(cols, bands):
    # target likely band mean-like columns (keep broad but reasonable)
    # e.g., B08_2019_mean, B08_mean_2019, mean_B08_2019, etc.
    bands_list = [b.strip() for b in bands.split(",") if b.strip()]
    hits = []
    for c in cols:
        low = c.lower()
        if "mean" not in low:
            continue
        for b in bands_list:
            if b.lower() in low:
                hits.append(c)
                break
    return sorted(list(set(hits)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--split_mode", choices=["labels", "file"], default="labels")
    ap.add_argument("--split_file", default=None)  # used if split_mode=file
    ap.add_argument("--id_cols", default="location_id,patch_id")
    ap.add_argument("--feature_prefix", default=None)  # e.g., e_
    ap.add_argument("--drop_cols", default="")
    ap.add_argument("--bands_offset", default="B08", help="Bands to apply additive offset to (comma-separated)")
    ap.add_argument("--bands_brightness", default="B02,B03,B04,B08,B11,B12",
                    help="Bands to apply multiplicative brightness scaling to band-mean columns")
    ap.add_argument("--offset_grid", default="0,0.005,0.01,0.02,0.03,0.05")
    ap.add_argument("--bright_grid", default="0,0.01,0.02,0.05,0.1")
    ap.add_argument("--mode", choices=["offset", "brightness", "both"], default="offset")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    id_cols = [c.strip() for c in args.id_cols.split(",")]
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    Xdf = pd.read_csv(args.features_csv)
    L = pd.read_csv(args.labels_csv)

    for c in id_cols:
        if c not in Xdf.columns:
            raise ValueError(f"Missing id col '{c}' in features_csv")
        if c not in L.columns:
            raise ValueError(f"Missing id col '{c}' in labels_csv")

    if args.label_col not in L.columns:
        raise ValueError(f"label_col '{args.label_col}' not in labels_csv. Found: {L.columns.tolist()}")

    # Always trust labels_csv for label + split
    for c in [args.label_col, "split"]:
        if c in Xdf.columns and c in L.columns:
            Xdf = Xdf.drop(columns=[c])

    if args.split_mode == "labels":
        if "split" not in L.columns:
            raise ValueError("split_mode=labels requires 'split' in labels_csv")
        lab = L[id_cols + [args.label_col, "split"]].drop_duplicates()
    else:
        if not args.split_file:
            raise ValueError("--split_file required when split_mode=file")
        S = pd.read_csv(args.split_file)
        if "location_id" not in S.columns or "split" not in S.columns:
            raise ValueError("split_file must have columns: location_id, split")
        lab = (
            L[id_cols + [args.label_col]]
            .drop_duplicates()
            .merge(S[["location_id", "split"]].drop_duplicates(), on="location_id", how="inner")
        )

    df = Xdf.merge(lab, on=id_cols, how="inner")

    # Filter invalid labels
    y = df[args.label_col].to_numpy()
    valid = ~pd.isna(y)
    if np.issubdtype(df[args.label_col].dtype, np.number):
        valid = valid & (df[args.label_col].to_numpy() >= 0)
    df = df.loc[valid].copy()

    tr = df["split"].eq("train").to_numpy()
    te = df["split"].eq("test").to_numpy()
    if tr.sum() == 0 or te.sum() == 0:
        raise ValueError(f"Empty train/test after merge. train={int(tr.sum())} test={int(te.sum())}")

    feat_cols = pick_feature_cols(
        df=df,
        id_cols=id_cols,
        label_col=args.label_col,
        split_col="split",
        feature_prefix=args.feature_prefix,
        drop_cols=drop_cols
    )

    # Choose which columns are affected
    offset_cols = match_band_columns(feat_cols, args.bands_offset, strict=False)
    # brightness scaling should hit per-year means too (e.g., B08_m0/B08_m1/B08_m2)
    bright_cols = match_band_columns(feat_cols, args.bands_brightness, strict=False)

    if args.mode in ["offset", "both"] and len(offset_cols) == 0:
        raise ValueError(f"No columns matched for additive offset bands_offset={args.bands_offset}")
    if args.mode in ["brightness", "both"] and len(bright_cols) == 0:
        raise ValueError(f"No columns matched for brightness bands_brightness={args.bands_brightness}")

    X0 = df[feat_cols].to_numpy(dtype=float)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=args.seed
        ))
    ])

    # Train once on clean train
    pipe.fit(X0[tr], df.loc[tr, args.label_col].to_numpy())

    offset_grid = [float(x.strip()) for x in args.offset_grid.split(",") if x.strip() != ""]
    bright_grid = [float(x.strip()) for x in args.bright_grid.split(",") if x.strip() != ""]

    rows = []

    def eval_X(Xb, tag_a, tag_b):
        yp = pipe.predict(Xb[te])
        yte = df.loc[te, args.label_col].to_numpy()
        acc = accuracy_score(yte, yp)
        mf1 = f1_score(yte, yp, average="macro")

        # drift on TEST set only (clean vs biased)
        d = cosine_distance(X0[te], Xb[te])
        return acc, mf1, float(d.mean()), float(d.std()), tag_a, tag_b

    if args.mode == "offset":
        for off in offset_grid:
            Xb = X0.copy()
            for c in offset_cols:
                j = feat_cols.index(c)
                Xb[:, j] = Xb[:, j] + off
            acc, mf1, dmu, dsd, ta, tb = eval_X(Xb, "offset", str(off))
            rows.append({
                "mode": "offset",
                "offset": off,
                "brightness": 0.0,
                "acc": acc,
                "macro_f1": mf1,
                "drift_cos_mean": dmu,
                "drift_cos_std": dsd,
                "n_test": int(te.sum()),
                "n_features": len(feat_cols),
                "n_offset_cols": len(offset_cols),
                "n_bright_cols": len(bright_cols),
            })

    elif args.mode == "brightness":
        for s in bright_grid:
            Xb = X0.copy()
            scale = 1.0 + s
            for c in bright_cols:
                j = feat_cols.index(c)
                Xb[:, j] = Xb[:, j] * scale
            acc, mf1, dmu, dsd, ta, tb = eval_X(Xb, "brightness", str(s))
            rows.append({
                "mode": "brightness",
                "offset": 0.0,
                "brightness": s,
                "acc": acc,
                "macro_f1": mf1,
                "drift_cos_mean": dmu,
                "drift_cos_std": dsd,
                "n_test": int(te.sum()),
                "n_features": len(feat_cols),
                "n_offset_cols": len(offset_cols),
                "n_bright_cols": len(bright_cols),
            })

    else:  # both
        for off in offset_grid:
            for s in bright_grid:
                Xb = X0.copy()
                # brightness first
                scale = 1.0 + s
                for c in bright_cols:
                    j = feat_cols.index(c)
                    Xb[:, j] = Xb[:, j] * scale
                # then additive offset
                for c in offset_cols:
                    j = feat_cols.index(c)
                    Xb[:, j] = Xb[:, j] + off

                acc, mf1, dmu, dsd, ta, tb = eval_X(Xb, f"both_off={off}", f"bright={s}")
                rows.append({
                    "mode": "both",
                    "offset": off,
                    "brightness": s,
                    "acc": acc,
                    "macro_f1": mf1,
                    "drift_cos_mean": dmu,
                    "drift_cos_std": dsd,
                    "n_test": int(te.sum()),
                    "n_features": len(feat_cols),
                    "n_offset_cols": len(offset_cols),
                    "n_bright_cols": len(bright_cols),
                })

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

    print(f"Saved: {args.out_csv}")
    print(f"Matched offset cols: {len(offset_cols)} (example: {offset_cols[:8]})")
    print(f"Matched brightness mean cols: {len(bright_cols)} (example: {bright_cols[:8]})")

if __name__ == "__main__":
    main()
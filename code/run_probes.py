import argparse
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True, help="CSV with features/embeddings and id columns")
    ap.add_argument("--labels_csv", required=True, help="CSV with labels + split OR labels only (if split_mode=file)")
    ap.add_argument("--label_col", required=True, help="Label column name in labels_csv (e.g., label, y_curv4)")
    ap.add_argument("--split_mode", choices=["labels", "file"], default="labels")
    ap.add_argument("--split_file", default=None, help="If split_mode=file: CSV with columns location_id,split")
    ap.add_argument("--id_cols", default="location_id,patch_id")
    ap.add_argument("--feature_prefix", default=None, help="If set: use columns starting with this prefix (e.g., e_)")
    ap.add_argument("--drop_cols", default="", help="Comma-separated additional columns to drop from features")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    id_cols = [c.strip() for c in args.id_cols.split(",")]
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    Xdf = pd.read_csv(args.data_csv)
    L = pd.read_csv(args.labels_csv)

    for c in id_cols:
        if c not in Xdf.columns:
            raise ValueError(f"Missing id col '{c}' in data_csv. Found: {Xdf.columns.tolist()}")
        if c not in L.columns:
            raise ValueError(f"Missing id col '{c}' in labels_csv. Found: {L.columns.tolist()}")

    if args.label_col not in L.columns:
        raise ValueError(f"label_col '{args.label_col}' not found in labels_csv. Columns: {L.columns.tolist()}")

    # Prevent label/split collisions causing label_x/label_y after merge:
    # We ALWAYS trust labels_csv for label + split.
    for c in [args.label_col, "split"]:
        if c in Xdf.columns and c in L.columns:
            Xdf = Xdf.drop(columns=[c])

    if args.split_mode == "labels":
        if "split" not in L.columns:
            raise ValueError("split_mode=labels but 'split' column not found in labels_csv")
        lab = L[id_cols + [args.label_col, "split"]].drop_duplicates()
    else:
        if not args.split_file:
            raise ValueError("--split_file required when split_mode=file")
        S = pd.read_csv(args.split_file)
        if "location_id" not in S.columns or "split" not in S.columns:
            raise ValueError(f"split_file must have columns location_id, split. Found: {S.columns.tolist()}")
        lab = (
            L[id_cols + [args.label_col]]
            .drop_duplicates()
            .merge(S[["location_id", "split"]].drop_duplicates(), on="location_id", how="inner")
        )

    df = Xdf.merge(lab, on=id_cols, how="inner")

    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' missing after merge. Columns: {df.columns.tolist()}")

    # Filter invalid labels (-1) and NaN labels
    y = df[args.label_col].to_numpy()
    valid = ~pd.isna(y)
    if np.issubdtype(df[args.label_col].dtype, np.number):
        valid = valid & (df[args.label_col].to_numpy() >= 0)
    df = df.loc[valid].copy()

    y = df[args.label_col].to_numpy()
    tr = df["split"].eq("train").to_numpy()
    te = df["split"].eq("test").to_numpy()

    if tr.sum() == 0 or te.sum() == 0:
        raise ValueError(f"Empty train/test after merge. train={int(tr.sum())} test={int(te.sum())}")

    # Select feature columns
    if args.feature_prefix:
        feat_cols = [c for c in df.columns if c.startswith(args.feature_prefix)]
    else:
        blacklist = set(id_cols + [args.label_col, "split", "path", "sample_id", "delta_2019_2023"])
        blacklist |= set(drop_cols)
        feat_cols = [c for c in df.columns if c not in blacklist and pd.api.types.is_numeric_dtype(df[c])]

    if not feat_cols:
        raise ValueError("No feature columns found. Use --feature_prefix e_ for embeddings or check your data file.")

    X = df[feat_cols].to_numpy(dtype=float)

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

    pipe.fit(X[tr], y[tr])
    yp = pipe.predict(X[te])

    acc = accuracy_score(y[te], yp)
    mf1 = f1_score(y[te], yp, average="macro")
    rep = classification_report(y[te], yp, output_dict=True, zero_division=0)
    cm = confusion_matrix(y[te], yp)

    rows = []
    for cls in sorted([k for k in rep.keys() if str(k).isdigit()], key=lambda x: int(x)):
        rows.append({
            "split_mode": args.split_mode,
            "split_file": args.split_file if args.split_file else "",
            "label_col": args.label_col,
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "acc": float(acc),
            "macro_f1": float(mf1),
            "class": int(cls),
            "precision": float(rep[cls]["precision"]),
            "recall": float(rep[cls]["recall"]),
            "f1": float(rep[cls]["f1-score"]),
            "support": int(rep[cls]["support"]),
            "n_features": int(len(feat_cols)),
        })

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(f"Saved: {args.out_csv}")
    print(f"Acc={acc:.4f} MacroF1={mf1:.4f} | n_train={int(tr.sum())} n_test={int(te.sum())} | n_features={len(feat_cols)}")
    print("Confusion matrix:\n", cm)


if __name__ == "__main__":
    main()
import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_csv", required=True)
    p.add_argument("--test_frac", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_pred_csv", default="preds_probe.csv")
    args = p.parse_args()

    df = pd.read_csv(args.features_csv)

    required = {"label", "location_id"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"features_csv must include {required}")

    # Location-based split
    rng = np.random.default_rng(args.seed)
    locs = np.array(sorted(df["location_id"].unique()))
    rng.shuffle(locs)

    n_test = max(1, int(round(len(locs) * args.test_frac)))
    test_locs = set(locs[:n_test])

    df_train = df[~df["location_id"].isin(test_locs)].copy()
    df_test  = df[df["location_id"].isin(test_locs)].copy()

    y_train = df_train["label"].astype(int).values
    y_test  = df_test["label"].astype(int).values

    drop_cols = {"label", "location_id"}
    if "path" in df.columns:
        drop_cols.add("path")

    feat_cols = [c for c in df.columns if c not in drop_cols]

    X_train = df_train[feat_cols].replace([np.inf, -np.inf], np.nan)
    X_test  = df_test[feat_cols].replace([np.inf, -np.inf], np.nan)

    # median impute
    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test  = X_test.fillna(med)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=4000,
            class_weight="balanced"
        ))
    ])

    clf.fit(X_train.values, y_train)
    pred = clf.predict(X_test.values)

    acc = accuracy_score(y_test, pred)
    macro = f1_score(y_test, pred, average="macro")

    print("=== PROBE (features A+) ===")
    print("test_acc:", float(acc))
    print("test_macroF1:", float(macro))
    print(classification_report(y_test, pred, digits=4))

    out = df_test[["sample_id", "location_id", "patch_id", "label"]].copy()
    out["pred"] = pred
    out.to_csv(args.out_pred_csv, index=False)

    print("saved preds:", args.out_pred_csv)
    print("n_train:", len(df_train),
          "n_test:", len(df_test),
          "n_locations:", len(locs),
          "n_test_locations:", len(test_locs))


if __name__ == "__main__":
    main()

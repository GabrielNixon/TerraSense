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
    p.add_argument("--drop_mode", choices=["drop_nd_d02_only", "drop_all_d02", "drop_nd_all_deltas"], default="drop_nd_d02_only")
    args = p.parse_args()

    df = pd.read_csv(args.features_csv)

    rng = np.random.default_rng(args.seed)
    locs = np.array(sorted(df["location_id"].unique()))
    rng.shuffle(locs)
    n_test = max(1, int(round(len(locs) * args.test_frac)))
    test_locs = set(locs[:n_test])

    train = df[~df["location_id"].isin(test_locs)].copy()
    test  = df[df["location_id"].isin(test_locs)].copy()

    y_train = train["label"].astype(int).values
    y_test  = test["label"].astype(int).values

    drop_cols = {"label", "location_id"}
    feat_cols = [c for c in df.columns if c not in drop_cols]

    if args.drop_mode == "drop_nd_d02_only":
        feat_cols = [c for c in feat_cols if c != "nd_d02"]
    elif args.drop_mode == "drop_all_d02":
        feat_cols = [c for c in feat_cols if not c.endswith("_d02")]
    elif args.drop_mode == "drop_nd_all_deltas":
        feat_cols = [c for c in feat_cols if not (c.startswith("nd_") and ("_d" in c))]

    X_train = train[feat_cols].replace([np.inf, -np.inf], np.nan)
    X_test  = test[feat_cols].replace([np.inf, -np.inf], np.nan)

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test  = X_test.fillna(med)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])

    clf.fit(X_train.values, y_train)
    pred = clf.predict(X_test.values)

    acc = accuracy_score(y_test, pred)
    macro = f1_score(y_test, pred, average="macro")

    print("drop_mode:", args.drop_mode)
    print("n_features:", len(feat_cols))
    print("test_acc:", float(acc))
    print("test_macroF1:", float(macro))
    print(classification_report(y_test, pred, digits=4))

if __name__ == "__main__":
    main()

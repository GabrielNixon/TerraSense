import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline

def cosine_dist(a, b, eps=1e-9):
    an = np.linalg.norm(a, axis=1, keepdims=True) + eps
    bn = np.linalg.norm(b, axis=1, keepdims=True) + eps
    return 1.0 - (a * b).sum(axis=1, keepdims=False) / (an[:, 0] * bn[:, 0])

def l2_dist(a, b):
    return np.linalg.norm(a - b, axis=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_csv", required=True)
    ap.add_argument("--split_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--id_cols", default="location_id,patch_id")
    ap.add_argument("--emb_prefix", default="e_")
    ap.add_argument("--label_col", required=True)
    ap.add_argument("--bias_mags", default="0,0.002,0.005,0.01,0.02,0.05")
    ap.add_argument("--drift", choices=["cosine", "l2"], default="cosine")
    ap.add_argument("--do_probe", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    id_cols = [c.strip() for c in args.id_cols.split(",")]
    mags = [float(x) for x in args.bias_mags.split(",")]

    df = pd.read_csv(args.emb_csv)
    split = pd.read_csv(args.split_csv)

    df = df.merge(split[id_cols + ["split"]], on=id_cols, how="inner")

    emb_cols = [c for c in df.columns if c.startswith(args.emb_prefix)]
    X = df[emb_cols].to_numpy(dtype=float)
    y = df[args.label_col].to_numpy()

    tr = df["split"].eq("train").to_numpy()
    te = df["split"].eq("test").to_numpy()

    rng = np.random.default_rng(args.seed)
    v = rng.standard_normal(X.shape[1]).astype(np.float64)
    v = v / (np.linalg.norm(v) + 1e-12)

    rows = []
    for m in mags:
        Xb = X + m * v[None, :]
        if args.drift == "cosine":
            drift = cosine_dist(X, Xb)
        else:
            drift = l2_dist(X, Xb)

        row = {
            "bias_mag": m,
            "drift_mean_all": float(np.mean(drift)),
            "drift_mean_train": float(np.mean(drift[tr])),
            "drift_mean_test": float(np.mean(drift[te])),
            "drift_p50_all": float(np.percentile(drift, 50)),
            "drift_p90_all": float(np.percentile(drift, 90)),
        }

        if args.do_probe:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    multi_class="auto",
                    random_state=args.seed
                ))
            ])
            pipe.fit(Xb[tr], y[tr])
            yp = pipe.predict(Xb[te])
            row["acc"] = float(accuracy_score(y[te], yp))
            row["macro_f1"] = float(f1_score(y[te], yp, average="macro"))
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.labels_csv)

    X = []
    y = []
    split = []

    for _, r in tqdm(df.iterrows(), total=len(df)):
        d = np.load(r["path"], allow_pickle=True)
        x_a = d["x_a"]
        x_b = d["x_b"]

        # simple mean spectral features
        feat = np.concatenate([
            x_a.mean(axis=(0,1)),
            x_b.mean(axis=(0,1)),
            (x_b - x_a).mean(axis=(0,1)),
        ])

        X.append(feat)
        y.append(r["label"])
        split.append(r["split"])

    X = np.stack(X)
    y = np.array(y)

    train_mask = np.array(split) == "train"
    test_mask = np.array(split) == "test"

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X[train_mask], y[train_mask])

    preds = clf.predict(X[test_mask])

    print("Accuracy:", accuracy_score(y[test_mask], preds))
    print(classification_report(y[test_mask], preds))

if __name__ == "__main__":
    main()

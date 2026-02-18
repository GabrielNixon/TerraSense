import argparse
import pandas as pd
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--train_frac", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.labels_csv)

    rng = np.random.default_rng(args.seed)
    locations = df["location_id"].unique()
    rng.shuffle(locations)

    n_train = int(len(locations) * args.train_frac)
    train_locs = set(locations[:n_train])

    df["split"] = df["location_id"].apply(
        lambda x: "train" if x in train_locs else "test"
    )

    df.to_csv(args.out_csv, index=False)

    print("Train locations:", len(train_locs))
    print("Test locations:", len(locations) - len(train_locs))
    print(df["split"].value_counts())

if __name__ == "__main__":
    main()

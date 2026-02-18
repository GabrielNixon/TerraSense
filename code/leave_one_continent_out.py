import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_csv", required=True)
    ap.add_argument("--loc_continent_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--location_id_col", default="location_id")
    args = ap.parse_args()

    patch = pd.read_csv(args.patch_csv)
    loc = pd.read_csv(args.loc_continent_csv)

    patch = patch.merge(
        loc[[args.location_id_col, "continent"]],
        on=args.location_id_col,
        how="left"
    )

    continents = sorted([c for c in patch["continent"].dropna().unique().tolist()])
    for c in continents:
        df = patch[[args.location_id_col]].drop_duplicates().copy()
        df["split"] = "train"
        holdout_locs = patch.loc[patch["continent"].eq(c), args.location_id_col].dropna().unique()
        df.loc[df[args.location_id_col].isin(holdout_locs), "split"] = "test"
        out_path = f"{args.out_dir}/split_leaveout_{c.replace(' ', '_')}.csv"
        df.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()
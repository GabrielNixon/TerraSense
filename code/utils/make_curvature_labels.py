import argparse
import pandas as pd
import numpy as np

def curvature_label(nd0, nd1, nd2, delta):
    d01 = nd1 - nd0
    d12 = nd2 - nd1

    up01 = d01 > delta
    dn01 = d01 < -delta
    up12 = d12 > delta
    dn12 = d12 < -delta

    mono_up = up01 & up12
    mono_dn = dn01 & dn12
    osc = (up01 & dn12) | (dn01 & up12)
    stableish = ~(mono_up | mono_dn | osc)

    y3 = np.full(len(nd0), -1, dtype=int)
    y3[mono_dn] = 0
    y3[stableish] = 1
    y3[mono_up] = 2

    y4 = np.full(len(nd0), -1, dtype=int)
    y4[mono_dn] = 0
    y4[mono_up] = 1
    y4[osc] = 2
    y4[stableish] = 3

    return y3, y4, d01, d12

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--ndvi_cols", default="nd_2019,nd_2021,nd_2023")
    ap.add_argument("--delta", type=float, default=0.05)
    args = ap.parse_args()

    c0, c1, c2 = [c.strip() for c in args.ndvi_cols.split(",")]
    df = pd.read_csv(args.in_csv)

    nd0 = df[c0].to_numpy(dtype=float)
    nd1 = df[c1].to_numpy(dtype=float)
    nd2 = df[c2].to_numpy(dtype=float)

    y3, y4, d01, d12 = curvature_label(nd0, nd1, nd2, args.delta)
    df["d01"] = d01
    df["d12"] = d12
    df["y_curv3"] = y3
    df["y_curv4"] = y4

    df.to_csv(args.out_csv, index=False)

if __name__ == "__main__":
    main()
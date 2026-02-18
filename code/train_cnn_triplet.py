import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score


BANDS6 = ["B02","B03","B04","B08","B11","B12"]


def safe_ndvi(nir, red):
    denom = nir + red
    return (nir - red) / np.where(np.abs(denom) < 1e-6, np.nan, denom)


def mask_from_fmask(x, bands):
    if "Fmask" not in bands:
        return np.ones(x.shape[:2], dtype=bool)
    f = x[..., bands.index("Fmask")]
    f = np.nan_to_num(f, nan=255.0).astype(np.uint16)
    valid = f != 255
    bad = (f == 2) | (f == 3) | (f == 4)
    return valid & (~bad)


class TripletDataset(Dataset):
    def __init__(self, df, mode="diff2", use_fmask=True):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.use_fmask = use_fmask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        d = np.load(r["path"], allow_pickle=True)
        bands = list(d["bands"])

        x0 = d["x_0"].astype(np.float32)
        x1 = d["x_1"].astype(np.float32)
        x2 = d["x_2"].astype(np.float32)

        if "Fmask" in bands:
            x0r = x0[..., :6]
            x1r = x1[..., :6]
            x2r = x2[..., :6]
        else:
            x0r, x1r, x2r = x0, x1, x2

        if self.use_fmask and ("Fmask" in bands):
            m0 = mask_from_fmask(x0, bands)
            m1 = mask_from_fmask(x1, bands)
            m2 = mask_from_fmask(x2, bands)
            m = (m0 & m1 & m2).astype(np.float32)
            x0r = x0r * m[..., None]
            x1r = x1r * m[..., None]
            x2r = x2r * m[..., None]

        if self.mode == "diff2":
            a = (x1r - x0r)
            b = (x2r - x1r)
            x = np.concatenate([a, b], axis=-1)
        elif self.mode == "stack3":
            x = np.concatenate([x0r, x1r, x2r], axis=-1)
        else:
            raise RuntimeError("mode must be diff2 or stack3")

        x = np.transpose(x, (2, 0, 1))
        y = int(r["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


class SmallCNN(nn.Module):
    def __init__(self, cin, nclass=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, nclass)

    def forward(self, x):
        h = self.net(x).squeeze(-1).squeeze(-1)
        return self.head(h)


def split_by_location(df, test_frac=0.30, seed=42):
    rng = np.random.default_rng(seed)
    locs = np.array(sorted(df["location_id"].unique()))
    rng.shuffle(locs)
    n_test = max(1, int(round(len(locs) * test_frac)))
    test_locs = set(locs[:n_test])
    train = df[~df["location_id"].isin(test_locs)].copy()
    test = df[df["location_id"].isin(test_locs)].copy()
    return train, test


@torch.no_grad()
def eval_model(model, loader, device):
    ys, ps = [], []
    model.eval()
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    acc = accuracy_score(y, p)
    macro = f1_score(y, p, average="macro")
    rep = classification_report(y, p, digits=4)
    return acc, macro, rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--mode", choices=["diff2", "stack3"], default="diff2")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_frac", type=float, default=0.30)
    p.add_argument("--no_fmask", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.labels_csv)
    train_df, test_df = split_by_location(df, test_frac=args.test_frac, seed=args.seed)

    cin = 12 if args.mode == "diff2" else 18

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCNN(cin=cin, nclass=3).to(device)

    train_ds = TripletDataset(train_df, mode=args.mode, use_fmask=(not args.no_fmask))
    test_ds = TripletDataset(test_df, mode=args.mode, use_fmask=(not args.no_fmask))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best = (-1, -1, None)

    for ep in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

        acc, macro, rep = eval_model(model, test_loader, device)
        if macro > best[1]:
            best = (acc, macro, rep)

        print(f"epoch {ep:02d} | acc={acc:.4f} macroF1={macro:.4f}")

    print("\n=== BEST (by macro-F1) ===")
    print("test_acc:", float(best[0]))
    print("test_macroF1:", float(best[1]))
    print(best[2])


if __name__ == "__main__":
    main()

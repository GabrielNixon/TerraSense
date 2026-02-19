import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, classification_report


BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "Fmask"]


class PairChangeDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str, mu=None, sigma=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode  # "diff" or "stack"
        self.mu = mu
        self.sigma = sigma

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        d = np.load(r["path"], allow_pickle=True)

        xa = d["x_a"].astype(np.float32)  # H,W,C
        xb = d["x_b"].astype(np.float32)

        if self.mode == "diff":
            x = xb - xa
        elif self.mode == "stack":
            x = np.concatenate([xa, xb, xb - xa], axis=-1)
        else:
            raise ValueError("mode must be diff or stack")

        x = np.transpose(x, (2, 0, 1))  # C,H,W

        if (self.mu is not None) and (self.sigma is not None):
            x = (x - self.mu[:, None, None]) / (self.sigma[:, None, None] + 1e-6)

        y = int(r["label"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def compute_train_stats(df_train: pd.DataFrame, mode: str, max_items: int = 0):
    rows = df_train
    if max_items and max_items > 0 and len(rows) > max_items:
        rows = rows.sample(max_items, random_state=42)

    sums = None
    sqs = None
    n = 0

    for _, r in tqdm(rows.iterrows(), total=len(rows), desc="Computing norm stats"):
        d = np.load(r["path"], allow_pickle=True)
        xa = d["x_a"].astype(np.float32)
        xb = d["x_b"].astype(np.float32)

        if mode == "diff":
            x = xb - xa
        else:  # stack
            x = np.concatenate([xa, xb, xb - xa], axis=-1)

        x = np.transpose(x, (2, 0, 1))  # C,H,W
        c = x.shape[0]
        v = x.reshape(c, -1)

        if sums is None:
            sums = v.sum(axis=1)
            sqs = (v * v).sum(axis=1)
        else:
            sums += v.sum(axis=1)
            sqs += (v * v).sum(axis=1)

        n += v.shape[1]

    mu = sums / n
    var = (sqs / n) - mu * mu
    sigma = np.sqrt(np.maximum(var, 1e-8))
    return mu.astype(np.float32), sigma.astype(np.float32)


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.head(x)


def train_one_epoch(model, loader, opt, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        opt.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    ys = []
    ps = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    return acc, mf1, y_true, y_pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--mode", choices=["diff", "stack"], default="diff")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--stats_max_items", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.labels_csv)
    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()

    mu, sigma = compute_train_stats(df_train, args.mode, max_items=args.stats_max_items)

    train_ds = PairChangeDataset(df_train, args.mode, mu=mu, sigma=sigma)
    test_ds = PairChangeDataset(df_test, args.mode, mu=mu, sigma=sigma)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    in_ch = (7 if args.mode == "diff" else 21)
    model = SmallCNN(in_ch=in_ch, n_classes=3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = (-1.0, -1.0)
    best_state = None

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, opt, device)
        acc, mf1, _, _ = eval_model(model, test_loader, device)

        if mf1 > best[1]:
            best = (acc, mf1)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"epoch {ep:02d} | loss {loss:.4f} | test_acc {acc:.4f} | test_macroF1 {mf1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    acc, mf1, y_true, y_pred = eval_model(model, test_loader, device)
    print("\n=== BEST (by macro-F1) ===")
    print("test_acc:", acc)
    print("test_macroF1:", mf1)
    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()

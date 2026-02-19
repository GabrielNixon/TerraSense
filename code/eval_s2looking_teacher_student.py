import os
import glob
import json
import argparse
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, accuracy_score

from prithvi_teacher.prithvi_mae import PrithviMAE
from train_distill_regime_total import StudentRegime


# NOTE: S2Looking is VHR RGB side-looking imagery, NOT Sentinel-2 reflectance.
# We therefore DO NOT apply Prithvi S2 mean/std normalization here.
# We keep a simple, consistent scale: RGB -> fake-6ch in [0..10000], then /10000 -> [0..1].


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def rgb_to_6ch(x_rgb_0_255):
    # x_rgb_0_255: (B,3,H,W) in [0..255]
    x = x_rgb_0_255.float() / 255.0 * 10000.0
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    # fake mapping to 6-ch expected by Prithvi (B02,B03,B04,B08,B11,B12)
    return torch.cat([b, g, r, r, r, r], dim=1)


class S2Looking(Dataset):
    def __init__(self, root, split="train", crop=256, seed=42, use_label="label"):
        self.root = root
        self.split = split
        self.crop = int(crop)
        self.rng = random.Random(seed)
        self.use_label = use_label  # "label" or "label1" or "label2"

        d1 = os.path.join(root, split, "Image1")
        d2 = os.path.join(root, split, "Image2")
        dl = os.path.join(root, split, use_label)

        self.p1 = sorted(glob.glob(os.path.join(d1, "*.png")) + glob.glob(os.path.join(d1, "*.jpg")) + glob.glob(os.path.join(d1, "*.tif")))
        self.p2 = sorted(glob.glob(os.path.join(d2, "*.png")) + glob.glob(os.path.join(d2, "*.jpg")) + glob.glob(os.path.join(d2, "*.tif")))
        self.pl = sorted(glob.glob(os.path.join(dl, "*.png")) + glob.glob(os.path.join(dl, "*.jpg")) + glob.glob(os.path.join(dl, "*.tif")))

        if not (len(self.p1) == len(self.p2) == len(self.pl) and len(self.p1) > 0):
            raise RuntimeError(f"Bad S2Looking split: {split} | Image1={len(self.p1)} Image2={len(self.p2)} {use_label}={len(self.pl)}")

    def __len__(self):
        return len(self.p1)

    def _img(self, p):
        im = Image.open(p).convert("RGB")
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float()  # (3,H,W)
        return x

    def _mask(self, p):
        im = Image.open(p).convert("L")
        m = torch.from_numpy(np.array(im)).float()
        m = (m > 0).float()
        return m

    def __getitem__(self, idx):
        a = self._img(self.p1[idx])
        b = self._img(self.p2[idx])
        y = self._mask(self.pl[idx])

        _, H, W = a.shape
        if H < self.crop or W < self.crop:
            raise RuntimeError("Image smaller than crop; reduce --crop")

        if self.split == "train":
            top = self.rng.randint(0, H - self.crop)
            left = self.rng.randint(0, W - self.crop)
        else:
            top = (H - self.crop) // 2
            left = (W - self.crop) // 2

        a = a[:, top:top+self.crop, left:left+self.crop]
        b = b[:, top:top+self.crop, left:left+self.crop]
        y = y[top:top+self.crop, left:left+self.crop]

        return a, b, y


def load_teacher(teacher_dir, teacher_ckpt, device, img_size=256, num_frames=3):
    config_path = os.path.join(teacher_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)["pretrained_cfg"]
    cfg = dict(cfg)
    cfg["img_size"] = img_size
    cfg["num_frames"] = num_frames
    cfg["in_chans"] = 6
    cfg["coords_encoding"] = ["time", "location"]

    teacher = PrithviMAE(**cfg).to(device).eval()
    sd = torch.load(teacher_ckpt, map_location=device)
    for k in list(sd.keys()):
        if "pos_embed" in k:
            del sd[k]
    teacher.load_state_dict(sd, strict=False)
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def load_student(student_ckpt, device, img_size, num_frames, embed_dim, depth, heads, K_reg):
    student = StudentRegime(
        img_size=img_size,
        num_frames=num_frames,
        embed_dim=embed_dim,
        depth=depth,
        heads=heads,
        K_reg=K_reg,
    ).to(device).eval()

    sd = torch.load(student_ckpt, map_location=device)
    for k in list(sd.keys()):
        if "pos_embed" in k:
            del sd[k]
    student.load_state_dict(sd, strict=False)
    for p in student.parameters():
        p.requires_grad_(False)
    return student


@torch.no_grad()
def patches_teacher(model, x, tc, lc):
    feats = model.forward_features(x, tc, lc)
    last = feats[-1]
    return last[:, 1:, :]  # (B,N,1024)


@torch.no_grad()
def patches_student(model, x, tc, lc):
    feats = model.enc.forward_features(x, tc, lc)
    last = feats[-1]
    return last[:, 1:, :]  # (B,N,D)


class PatchHead(nn.Module):
    def __init__(self, d_in, grid):
        super().__init__()
        self.grid = grid
        self.lin = nn.Linear(d_in, 1)

    def forward(self, tokens):
        logits = self.lin(tokens).squeeze(-1)  # (B,P)
        return logits.view(tokens.shape[0], self.grid, self.grid)


def mask_to_patches_any(mask, patch=16):
    # mask: (B,H,W)
    m = mask.unsqueeze(1)  # (B,1,H,W)
    m = F.max_pool2d(m, kernel_size=patch, stride=patch)
    return (m > 0).float().squeeze(1)  # (B,gh,gw)


def metrics_from_logits(logits, y):
    # logits,y: (B,gh,gw)
    pred = (torch.sigmoid(logits) > 0.5).float()

    inter = (pred * y).sum(dim=(1, 2))
    union = ((pred + y) > 0).float().sum(dim=(1, 2)).clamp(min=1.0)
    iou = (inter / union).mean().item()

    y_np = y.detach().cpu().numpy().reshape(-1)
    p_np = pred.detach().cpu().numpy().reshape(-1)
    f1 = f1_score(y_np, p_np, average="binary", zero_division=0)
    acc = accuracy_score(y_np, p_np)
    return float(iou), float(f1), float(acc)


def run_eval(kind, backbone, head, loader, device, crop, num_frames):
    head.eval()
    ious, f1s, accs = [], [], []

    for a, b, y in loader:
        a = a.to(device)
        b = b.to(device)
        y = y.to(device)

        a6 = rgb_to_6ch(a)
        b6 = rgb_to_6ch(b)

        # T=3: [before, after, after]
        x = torch.stack([a6, b6, b6], dim=2)  # (B,6,T,H,W)
        x = x / 10000.0

        tc = torch.zeros((x.shape[0], num_frames, 2), device=device)
        lc = torch.zeros((x.shape[0], 2), device=device)

        with torch.no_grad():
            if kind == "teacher":
                tok = patches_teacher(backbone, x, tc, lc)
            else:
                tok = patches_student(backbone, x, tc, lc)

        gh = crop // 16
        P = gh * gh
        D = tok.shape[-1]
        tok = tok.view(x.shape[0], num_frames, P, D)

        diff = tok[:, 1] - tok[:, 0]  # (B,P,D)
        logits = head(diff)  # (B,gh,gw)

        y_patch = mask_to_patches_any(y, 16)
        iou, f1, acc = metrics_from_logits(logits, y_patch)

        ious.append(iou)
        f1s.append(f1)
        accs.append(acc)

    return float(np.mean(ious)), float(np.mean(f1s)), float(np.mean(accs))


def train_head(kind, backbone, head, train_loader, val_loader, device, crop, num_frames, epochs=5, lr=1e-3):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        head.train()
        losses = []

        for a, b, y in train_loader:
            a = a.to(device)
            b = b.to(device)
            y = y.to(device)

            a6 = rgb_to_6ch(a)
            b6 = rgb_to_6ch(b)

            x = torch.stack([a6, b6, b6], dim=2)
            x = x / 10000.0

            tc = torch.zeros((x.shape[0], num_frames, 2), device=device)
            lc = torch.zeros((x.shape[0], 2), device=device)

            with torch.no_grad():
                if kind == "teacher":
                    tok = patches_teacher(backbone, x, tc, lc)
                else:
                    tok = patches_student(backbone, x, tc, lc)

            gh = crop // 16
            P = gh * gh
            D = tok.shape[-1]
            tok = tok.view(x.shape[0], num_frames, P, D)

            diff = tok[:, 1] - tok[:, 0]
            logits = head(diff)

            y_patch = mask_to_patches_any(y, 16)

            pos = y_patch.sum().clamp(min=1.0)
            neg = y_patch.numel() - pos
            pos_weight = (neg / pos).clamp(max=50.0)

            loss = F.binary_cross_entropy_with_logits(logits, y_patch, pos_weight=pos_weight)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        viou, vf1, vacc = run_eval(kind, backbone, head, val_loader, device, crop, num_frames)
        print(f"ep {ep} | loss {np.mean(losses):.4f} | val IoU {viou:.4f} | val F1 {vf1:.4f} | val Acc {vacc:.4f}")

    return head


def patch_pos_rate(ds, n=200):
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
    pos = 0.0
    tot = 0.0
    for i, (_, _, y) in enumerate(dl):
        if i >= n:
            break
        y = y.float()
        yp = mask_to_patches_any(y, 16)  # (1,gh,gw)
        pos += float(yp.sum())
        tot += float(yp.numel())
    return pos / max(tot, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)

    ap.add_argument("--teacher_dir", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, required=True)

    ap.add_argument("--student_ckpts", type=str, default="")

    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_frames", type=int, default=3)

    ap.add_argument("--student_embed_dim", type=int, default=512)
    ap.add_argument("--student_depth", type=int, default=12)
    ap.add_argument("--student_heads", type=int, default=8)
    ap.add_argument("--K_reg", type=int, default=5)

    ap.add_argument("--use_label", type=str, default="label", choices=["label", "label1", "label2"])

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--posrate_n", type=int, default=200)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print("device:", device)

    train_ds = S2Looking(args.data_root, "train", crop=args.crop, seed=args.seed, use_label=args.use_label)
    val_ds   = S2Looking(args.data_root, "val",   crop=args.crop, seed=args.seed, use_label=args.use_label)
    test_ds  = S2Looking(args.data_root, "test",  crop=args.crop, seed=args.seed, use_label=args.use_label)

    print("train patch pos rate:", patch_pos_rate(train_ds, n=args.posrate_n))
    print("val   patch pos rate:", patch_pos_rate(val_ds,   n=args.posrate_n))
    print("test  patch pos rate:", patch_pos_rate(test_ds,  n=args.posrate_n))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    teacher = load_teacher(args.teacher_dir, args.teacher_ckpt, device, img_size=args.crop, num_frames=args.num_frames)
    gh = args.crop // 16

    print("\n== TEACHER ==")
    headT = PatchHead(d_in=1024, grid=gh).to(device)
    headT = train_head("teacher", teacher, headT, train_loader, val_loader, device, args.crop, args.num_frames,
                       epochs=args.epochs, lr=args.lr)
    tiou, tf1, tacc = run_eval("teacher", teacher, headT, test_loader, device, args.crop, args.num_frames)
    print(f"TEACHER TEST | IoU={tiou:.4f} | F1={tf1:.4f} | Acc={tacc:.4f}")

    ckpts = [c.strip() for c in args.student_ckpts.split(",") if c.strip()]
    for ck in ckpts:
        print(f"\n== STUDENT: {ck} ==")
        student = load_student(
            ck, device, args.crop, args.num_frames,
            args.student_embed_dim, args.student_depth, args.student_heads, args.K_reg
        )

        headS = PatchHead(d_in=args.student_embed_dim, grid=gh).to(device)
        headS = train_head("student", student, headS, train_loader, val_loader, device, args.crop, args.num_frames,
                           epochs=args.epochs, lr=args.lr)
        siou, sf1, sacc = run_eval("student", student, headS, test_loader, device, args.crop, args.num_frames)
        print(f"STUDENT TEST | IoU={siou:.4f} | F1={sf1:.4f} | Acc={sacc:.4f} | F1/teacher={(sf1/(tf1+1e-12)):.3f}")


if __name__ == "__main__":
    main()

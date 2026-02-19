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

from sklearn.metrics import f1_score

from prithvi_teacher.prithvi_mae import PrithviMAE
from train_distill_regime_total import StudentRegime


TEACHER_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
TEACHER_STD  = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_pixel_values(x):
    # x: (B,6,T,H,W) in ~0..10000 scale
    mean = torch.tensor(TEACHER_MEAN, device=x.device).view(1, 6, 1, 1, 1)
    std  = torch.tensor(TEACHER_STD,  device=x.device).view(1, 6, 1, 1, 1)
    return (x - mean) / (std + 1e-6)


def rgb_to_6ch(x_rgb_0_255):
    # x_rgb_0_255: (B,3,H,W) in [0..255], channel order is (R,G,B)
    x = x_rgb_0_255.float() / 255.0 * 10000.0
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    # B02,B03,B04,B08,B11,B12
    # We don't have NIR/SWIR, so repeat R (not zeros) to avoid dead channels.
    return torch.cat([b, g, r, r, r, r], dim=1)


class LEVIRCD(Dataset):
    def __init__(self, root, split="train", crop=256, seed=42):
        self.root = root
        self.split = split
        self.crop = int(crop)
        self.rng = random.Random(seed)

        a_dir = os.path.join(root, split, "A")
        b_dir = os.path.join(root, split, "B")
        l_dir = os.path.join(root, split, "label")

        self.a_paths = sorted(glob.glob(os.path.join(a_dir, "*.png")))
        self.b_paths = sorted(glob.glob(os.path.join(b_dir, "*.png")))
        self.l_paths = sorted(glob.glob(os.path.join(l_dir, "*.png")))

        if not (len(self.a_paths) == len(self.b_paths) == len(self.l_paths) and len(self.a_paths) > 0):
            raise RuntimeError(f"Bad split under {root}/{split} (need A,B,label pngs). "
                               f"A={len(self.a_paths)} B={len(self.b_paths)} L={len(self.l_paths)}")

    def __len__(self):
        return len(self.a_paths)

    def _load_rgb(self, p):
        im = Image.open(p).convert("RGB")
        x = torch.from_numpy(np.array(im)).permute(2, 0, 1).float()  # (3,H,W)
        return x

    def _load_mask(self, p):
        im = Image.open(p).convert("L")
        m = torch.from_numpy(np.array(im)).float()                   # (H,W) 0..255
        m = (m > 127).float()                                         # 0/1
        return m

    def __getitem__(self, idx):
        a = self._load_rgb(self.a_paths[idx])
        b = self._load_rgb(self.b_paths[idx])
        y = self._load_mask(self.l_paths[idx])

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
    # permissive on pos_embed
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
def backbone_patches_teacher(model, x, tc, lc):
    feats = model.forward_features(x, tc, lc)
    last = feats[-1]         # (B, 1+N, 1024)
    return last[:, 1:, :]    # (B, N, 1024)


@torch.no_grad()
def backbone_patches_student(model, x, tc, lc):
    feats = model.enc.forward_features(x, tc, lc)
    last = feats[-1]         # (B, 1+N, D)
    return last[:, 1:, :]    # (B, N, D)


class PatchChangeHead(nn.Module):
    def __init__(self, d_in, grid):
        super().__init__()
        self.grid = grid
        self.lin = nn.Linear(d_in, 1)

    def forward(self, diff_tokens):
        # diff_tokens: (B, P, D)
        logits = self.lin(diff_tokens).squeeze(-1)  # (B,P)
        return logits.view(diff_tokens.shape[0], self.grid, self.grid)  # (B,gh,gw)


def downsample_mask_to_patches(mask, patch=16):
    # mask: (B,H,W) float {0,1}
    m = mask.unsqueeze(1)  # (B,1,H,W)
    m = F.max_pool2d(m, kernel_size=patch, stride=patch)
    return (m > 0.0).float().squeeze(1)  # (B,gh,gw)


def iou_f1_acc_from_logits(logits, y):
    # logits,y: (B,gh,gw)
    pred = (torch.sigmoid(logits) > 0.5).float()

    inter = (pred * y).sum(dim=(1,2))
    union = ((pred + y) > 0).float().sum(dim=(1,2)).clamp(min=1.0)
    iou = (inter / union).mean().item()

    y_np = y.detach().cpu().numpy().reshape(-1)
    p_np = pred.detach().cpu().numpy().reshape(-1)
    f1 = f1_score(y_np, p_np, average="binary", zero_division=0)
    acc = float((p_np == y_np).mean())
    return float(iou), float(f1), float(acc)


@torch.no_grad()
def compute_patch_pos_rate(loader, device, patch=16):
    total = 0
    pos = 0
    for _, _, y in loader:
        y = y.to(device)
        y_patch = downsample_mask_to_patches(y, patch=patch)
        pos += int(y_patch.sum().item())
        total += int(y_patch.numel())
    return pos / max(1, total)


def run_eval(backbone_kind, backbone, head, loader, device, img_size, num_frames):
    head.eval()
    ious, f1s, accs = [], [], []

    for a, b, y in loader:
        a = a.to(device)
        b = b.to(device)
        y = y.to(device)

        x6a = rgb_to_6ch(a)
        x6b = rgb_to_6ch(b)

        # T=3 protocol: [before, after, after]
        x = torch.stack([x6a, x6b, x6b], dim=2)  # (B,6,T,H,W)
        x = normalize_pixel_values(x)

        tc = torch.zeros((x.shape[0], num_frames, 2), device=device)
        lc = torch.zeros((x.shape[0], 2), device=device)

        if backbone_kind == "teacher":
            patches = backbone_patches_teacher(backbone, x, tc, lc)
        else:
            patches = backbone_patches_student(backbone, x, tc, lc)

        gh = img_size // 16
        P = gh * gh
        D = patches.shape[-1]
        patches = patches.view(x.shape[0], num_frames, P, D)  # (B,T,P,D)
        diff = patches[:, 1] - patches[:, 0]                  # (B,P,D)

        logits = head(diff)                                   # (B,gh,gw)
        y_patch = downsample_mask_to_patches(y, patch=16)      # (B,gh,gw)

        iou, f1, acc = iou_f1_acc_from_logits(logits, y_patch)
        ious.append(iou); f1s.append(f1); accs.append(acc)

    return float(np.mean(ious)), float(np.mean(f1s)), float(np.mean(accs))


def train_head(backbone_kind, backbone, head, train_loader, val_loader, device, img_size, num_frames,
               pos_weight, epochs=5, lr=1e-3):
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(1, epochs + 1):
        head.train()
        losses = []

        for a, b, y in train_loader:
            a = a.to(device); b = b.to(device); y = y.to(device)

            x6a = rgb_to_6ch(a)
            x6b = rgb_to_6ch(b)
            x = torch.stack([x6a, x6b, x6b], dim=2)  # (B,6,T,H,W)
            x = normalize_pixel_values(x)

            tc = torch.zeros((x.shape[0], num_frames, 2), device=device)
            lc = torch.zeros((x.shape[0], 2), device=device)

            with torch.no_grad():
                if backbone_kind == "teacher":
                    patches = backbone_patches_teacher(backbone, x, tc, lc)
                else:
                    patches = backbone_patches_student(backbone, x, tc, lc)

            gh = img_size // 16
            P = gh * gh
            D = patches.shape[-1]
            patches = patches.view(x.shape[0], num_frames, P, D)
            diff = patches[:, 1] - patches[:, 0]

            logits = head(diff)                                # (B,gh,gw)
            y_patch = downsample_mask_to_patches(y, patch=16)   # (B,gh,gw)

            loss = crit(logits, y_patch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        viou, vf1, vacc = run_eval(backbone_kind, backbone, head, val_loader, device, img_size, num_frames)
        print(f"ep {ep} | loss {np.mean(losses):.4f} | val IoU {viou:.4f} | val F1 {vf1:.4f} | val Acc {vacc:.4f}")

    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levir_root", type=str, required=True)

    ap.add_argument("--teacher_dir", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, required=True)

    ap.add_argument("--student_ckpts", type=str, default="")  # comma-separated

    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_frames", type=int, default=3)

    ap.add_argument("--student_embed_dim", type=int, default=512)
    ap.add_argument("--student_depth", type=int, default=12)
    ap.add_argument("--student_heads", type=int, default=8)
    ap.add_argument("--K_reg", type=int, default=5)

    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print("device:", device)

    train_ds = LEVIRCD(args.levir_root, "train", crop=args.crop, seed=args.seed)
    val_ds   = LEVIRCD(args.levir_root, "val",   crop=args.crop, seed=args.seed)
    test_ds  = LEVIRCD(args.levir_root, "test",  crop=args.crop, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # imbalance stats (patch-level)
    tr_pos = compute_patch_pos_rate(train_loader, device, patch=16)
    va_pos = compute_patch_pos_rate(val_loader,   device, patch=16)
    te_pos = compute_patch_pos_rate(test_loader,  device, patch=16)
    print(f"train patch pos rate: {tr_pos}")
    print(f"val   patch pos rate: {va_pos}")
    print(f"test  patch pos rate: {te_pos}")

    # pos_weight = neg/pos for BCEWithLogitsLoss
    # clamp to avoid exploding if extremely tiny
    eps = 1e-6
    pos_weight_val = float((1.0 - tr_pos) / max(eps, tr_pos))
    pos_weight_val = float(np.clip(pos_weight_val, 1.0, 200.0))
    pos_weight = torch.tensor([pos_weight_val], device=device)
    print(f"pos_weight (neg/pos): {pos_weight_val:.3f}")

    teacher = load_teacher(args.teacher_dir, args.teacher_ckpt, device, img_size=args.crop, num_frames=args.num_frames)
    gh = args.crop // 16

    print("\n== TEACHER ==")
    headT = PatchChangeHead(d_in=1024, grid=gh).to(device)
    headT = train_head("teacher", teacher, headT, train_loader, val_loader, device,
                       args.crop, args.num_frames, pos_weight=pos_weight, epochs=args.epochs, lr=args.lr)
    tiou, tf1, tacc = run_eval("teacher", teacher, headT, test_loader, device, args.crop, args.num_frames)
    print(f"TEACHER TEST | IoU={tiou:.4f} | F1={tf1:.4f} | Acc={tacc:.4f}")

    ckpts = [c.strip() for c in args.student_ckpts.split(",") if c.strip()]
    for ck in ckpts:
        print(f"\n== STUDENT: {ck} ==")
        student = load_student(
            ck, device,
            img_size=args.crop,
            num_frames=args.num_frames,
            embed_dim=args.student_embed_dim,
            depth=args.student_depth,
            heads=args.student_heads,
            K_reg=args.K_reg,
        )

        headS = PatchChangeHead(d_in=args.student_embed_dim, grid=gh).to(device)
        headS = train_head("student", student, headS, train_loader, val_loader, device,
                           args.crop, args.num_frames, pos_weight=pos_weight, epochs=args.epochs, lr=args.lr)
        siou, sf1, sacc = run_eval("student", student, headS, test_loader, device, args.crop, args.num_frames)
        print(f"STUDENT TEST | IoU={siou:.4f} | F1={sf1:.4f} | Acc={sacc:.4f} | F1/teacher={(sf1/(tf1+1e-12)):.3f}")


if __name__ == "__main__":
    main()

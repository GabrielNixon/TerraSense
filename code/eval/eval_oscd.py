import os
import glob
import json
import argparse
import random
import numpy as np

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
    mean = torch.tensor(TEACHER_MEAN, device=x.device).view(1, 6, 1, 1, 1)
    std  = torch.tensor(TEACHER_STD,  device=x.device).view(1, 6, 1, 1, 1)
    return (x - mean) / (std + 1e-6)


def parse_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _find_child_dir(root, must_contain_substrings):
    root = os.path.abspath(root)
    candidates = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        low = name.lower()
        ok = True
        for s in must_contain_substrings:
            if s.lower() not in low:
                ok = False
                break
        if ok:
            candidates.append(p)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 0:
        raise RuntimeError(f"Could not find folder in {root} containing: {must_contain_substrings}. Found: {os.listdir(root)}")
    raise RuntimeError(f"Ambiguous folders for {must_contain_substrings} in {root}: {candidates}")


def _read_npy_or_tif(path):
    # OSCD commonly uses .tif (Sentinel-2) + label as .tif
    # We'll support tif via rasterio if installed, else error.
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext in [".tif", ".tiff"]:
        try:
            import rasterio
        except Exception as e:
            raise RuntimeError("OSCD .tif reading requires rasterio. Install: pip install rasterio") from e
        with rasterio.open(path) as src:
            arr = src.read()  # (C,H,W)
        return arr
    raise RuntimeError(f"Unsupported file type: {path}")


def _to_float_chw(x):
    # Accept (C,H,W) or (H,W,C)
    if x.ndim != 3:
        raise RuntimeError(f"Expected 3D array, got shape {x.shape}")
    if x.shape[0] in (13, 12, 10, 9, 8, 6, 4, 3):
        return x.astype(np.float32)  # (C,H,W)
    return np.transpose(x, (2, 0, 1)).astype(np.float32)


def _extract_bands(chw, band_order, use_bands):
    # chw: (C,H,W) with C=len(band_order)
    band_to_i = {b: i for i, b in enumerate(band_order)}
    idx = [band_to_i[b] for b in use_bands]
    out = chw[idx, :, :]  # (6,H,W)
    return out


def _mask_to01(mask):
    # mask can be (H,W) or (1,H,W)
    if mask.ndim == 3:
        mask = mask[0]
    m = mask.astype(np.float32)
    # OSCD labels are typically 0/255 or 0/1
    if m.max() > 1.5:
        m = (m > 127).astype(np.float32)
    else:
        m = (m > 0.5).astype(np.float32)
    return m


def _center_crop(chw, crop):
    C, H, W = chw.shape
    if H < crop or W < crop:
        raise RuntimeError(f"Image smaller than crop={crop}: got {(H,W)}")
    top = (H - crop) // 2
    left = (W - crop) // 2
    return chw[:, top:top+crop, left:left+crop], top, left


def _random_crop(chw, crop, rng):
    C, H, W = chw.shape
    if H < crop or W < crop:
        raise RuntimeError(f"Image smaller than crop={crop}: got {(H,W)}")
    top = rng.randint(0, H - crop)
    left = rng.randint(0, W - crop)
    return chw[:, top:top+crop, left:left+crop], top, left


class OSCD(Dataset):
    """
    Expects OSCD extracted as:
      oscd_root/
        Onera Satellite Change Detection dataset - Images/
          <loc>/ ... contains two image times (t1,t2) as .tif or .npy
        Onera Satellite Change Detection dataset - Train Labels/
          <loc>/ ... contains label mask
        Onera Satellite Change Detection dataset - Test Labels/
          <loc>/ ... contains label mask

    We auto-detect per-location by pairing the two image files in the location folder.
    """
    def __init__(self, oscd_root, split="train", crop=256, seed=42,
                 band_order=None, use_bands=None):
        self.oscd_root = os.path.abspath(oscd_root)
        self.split = split
        self.crop = int(crop)
        self.rng = random.Random(seed)

        self.images_root = _find_child_dir(self.oscd_root, ["onera", "images"])
        self.train_lbl_root = _find_child_dir(self.oscd_root, ["train", "labels"])
        self.test_lbl_root  = _find_child_dir(self.oscd_root, ["test", "labels"])

        self.band_order = band_order
        self.use_bands = use_bands

        if split not in ["train", "test"]:
            raise ValueError("split must be train or test")

        lbl_root = self.train_lbl_root if split == "train" else self.test_lbl_root

        # locations present in labels are canonical
        self.locations = sorted([d for d in os.listdir(lbl_root) if os.path.isdir(os.path.join(lbl_root, d))])

        # build pairs: for each location, find exactly 2 image tensors + 1 label
        self.samples = []
        for loc in self.locations:
            img_loc = os.path.join(self.images_root, loc)
            lbl_loc = os.path.join(lbl_root, loc)
            if not os.path.isdir(img_loc) or not os.path.isdir(lbl_loc):
                continue

            img_files = sorted(
                glob.glob(os.path.join(img_loc, "*.tif")) +
                glob.glob(os.path.join(img_loc, "*.tiff")) +
                glob.glob(os.path.join(img_loc, "*.npy"))
            )
            lbl_files = sorted(
                glob.glob(os.path.join(lbl_loc, "*.tif")) +
                glob.glob(os.path.join(lbl_loc, "*.tiff")) +
                glob.glob(os.path.join(lbl_loc, "*.png")) +
                glob.glob(os.path.join(lbl_loc, "*.jpg")) +
                glob.glob(os.path.join(lbl_loc, "*.npy"))
            )

            if len(img_files) < 2 or len(lbl_files) < 1:
                continue

            # choose first 2 images deterministically (OSCD typically stores t1, t2)
            t1 = img_files[0]
            t2 = img_files[1]
            y  = lbl_files[0]
            self.samples.append((loc, t1, t2, y))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No OSCD samples found.\n"
                f"images_root={self.images_root}\n"
                f"labels_root={(self.train_lbl_root if split=='train' else self.test_lbl_root)}\n"
                f"Tip: check each location folder has >=2 image files and >=1 label file."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        loc, p1, p2, pl = self.samples[idx]

        a = _to_float_chw(_read_npy_or_tif(p1))
        b = _to_float_chw(_read_npy_or_tif(p2))

        # label might be tif/png/jpg/npy
        ext = os.path.splitext(pl)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            from PIL import Image
            m = np.array(Image.open(pl).convert("L")).astype(np.float32)
        else:
            m = _read_npy_or_tif(pl)
        m = _mask_to01(m)  # (H,W)

        # band select -> 6ch
        if self.band_order is None or self.use_bands is None:
            raise RuntimeError("band_order/use_bands must be provided")
        a6 = _extract_bands(a, self.band_order, self.use_bands)
        b6 = _extract_bands(b, self.band_order, self.use_bands)

        # crop (train=random, test=center) and apply same crop to mask
        if self.split == "train":
            a6, top, left = _random_crop(a6, self.crop, self.rng)
            b6 = b6[:, top:top+self.crop, left:left+self.crop]
            m  = m[top:top+self.crop, left:left+self.crop]
        else:
            a6, top, left = _center_crop(a6, self.crop)
            b6 = b6[:, top:top+self.crop, left:left+self.crop]
            m  = m[top:top+self.crop, left:left+self.crop]

        a6 = torch.from_numpy(a6)          # (6,H,W)
        b6 = torch.from_numpy(b6)          # (6,H,W)
        m  = torch.from_numpy(m)           # (H,W)

        return a6, b6, m


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
    m = mask.unsqueeze(1)  # (B,1,H,W)
    m = F.max_pool2d(m, kernel_size=patch, stride=patch)
    return (m > 0).float().squeeze(1)  # (B,gh,gw)


def metrics_from_logits(logits, y):
    # logits,y: (B,gh,gw)
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()

    inter = (pred * y).sum(dim=(1, 2))
    union = ((pred + y) > 0).float().sum(dim=(1, 2)).clamp(min=1.0)
    iou = (inter / union).mean().item()

    y_np = y.detach().cpu().numpy().reshape(-1)
    p_np = pred.detach().cpu().numpy().reshape(-1)
    f1 = f1_score(y_np, p_np, average="binary", zero_division=0)

    acc = float((pred == y).float().mean().item())
    return float(iou), float(f1), acc


@torch.no_grad()
def run_eval(kind, backbone, head, loader, device, crop, num_frames):
    head.eval()
    ious, f1s, accs = [], [], []

    for a6, b6, y in loader:
        a6 = a6.to(device).float()
        b6 = b6.to(device).float()
        y  = y.to(device).float()

        # T=3: [before, after, after]
        x = torch.stack([a6, b6, b6], dim=2)  # (B,6,T,H,W)
        x = normalize_pixel_values(x)

        tc = torch.zeros((x.shape[0], num_frames, 2), device=device)
        lc = torch.zeros((x.shape[0], 2), device=device)

        if kind == "teacher":
            tok = patches_teacher(backbone, x, tc, lc)
        else:
            tok = patches_student(backbone, x, tc, lc)

        gh = crop // 16
        P = gh * gh
        D = tok.shape[-1]
        tok = tok.view(x.shape[0], num_frames, P, D)

        diff = tok[:, 1] - tok[:, 0]  # (B,P,D)
        logits = head(diff)

        y_patch = mask_to_patches_any(y, 16)
        iou, f1, acc = metrics_from_logits(logits, y_patch)
        ious.append(iou); f1s.append(f1); accs.append(acc)

    return float(np.mean(ious)), float(np.mean(f1s)), float(np.mean(accs))


def estimate_pos_rate(loader, crop):
    gh = crop // 16
    tot = 0.0
    n = 0
    for _, _, y in loader:
        y = y.float()
        y_patch = mask_to_patches_any(y, 16)  # (B,gh,gw)
        tot += float(y_patch.mean().item())
        n += 1
        if n >= 25:
            break
    return tot / max(1, n)


def train_head(kind, backbone, head, train_loader, val_loader, device, crop, num_frames,
               epochs=5, lr=1e-3, pos_weight=None):
    if pos_weight is None:
        bce = nn.BCEWithLogitsLoss()
    else:
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        head.train()
        losses = []

        for a6, b6, y in train_loader:
            a6 = a6.to(device).float()
            b6 = b6.to(device).float()
            y  = y.to(device).float()

            x = torch.stack([a6, b6, b6], dim=2)
            x = normalize_pixel_values(x)

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

            loss = bce(logits, y_patch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        viou, vf1, vacc = run_eval(kind, backbone, head, val_loader, device, crop, num_frames)
        print(f"ep {ep} | loss {np.mean(losses):.4f} | val IoU {viou:.4f} | val F1 {vf1:.4f} | val Acc {vacc:.4f}")

    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oscd_root", type=str, required=True)

    ap.add_argument("--teacher_dir", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, required=True)
    ap.add_argument("--student_ckpts", type=str, default="")

    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_frames", type=int, default=3)

    ap.add_argument("--band_order", type=str, default="B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12")
    ap.add_argument("--use_bands", type=str, default="B02,B03,B04,B08,B11,B12")

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

    band_order = parse_list(args.band_order)
    use_bands = parse_list(args.use_bands)
    if len(use_bands) != 6:
        raise RuntimeError("--use_bands must be exactly 6 bands for Prithvi (B02,B03,B04,B08,B11,B12).")

    # OSCD: train split has labels; test split has labels too (your zip names)
    train_ds = OSCD(args.oscd_root, split="train", crop=args.crop, seed=args.seed,
                    band_order=band_order, use_bands=use_bands)
    test_ds  = OSCD(args.oscd_root, split="test",  crop=args.crop, seed=args.seed,
                    band_order=band_order, use_bands=use_bands)

    # make a small val from train (fixed split)
    n = len(train_ds)
    idx = np.arange(n)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(idx)
    n_val = max(1, int(0.1 * n))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    train_sub = torch.utils.data.Subset(train_ds, tr_idx)
    val_sub = torch.utils.data.Subset(train_ds, val_idx)

    train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_sub,   batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    train_pos = estimate_pos_rate(train_loader, args.crop)
    val_pos   = estimate_pos_rate(val_loader, args.crop)
    test_pos  = estimate_pos_rate(test_loader, args.crop)

    print(f"train patch pos rate: {train_pos}")
    print(f"val   patch pos rate: {val_pos}")
    print(f"test  patch pos rate: {test_pos}")

    pos_weight = (1.0 - train_pos) / max(train_pos, 1e-6)
    print(f"pos_weight (neg/pos): {pos_weight:.3f}")

    teacher = load_teacher(args.teacher_dir, args.teacher_ckpt, device, img_size=args.crop, num_frames=args.num_frames)
    gh = args.crop // 16

    print("\n== TEACHER ==")
    headT = PatchHead(d_in=1024, grid=gh).to(device)
    headT = train_head("teacher", teacher, headT, train_loader, val_loader, device,
                       args.crop, args.num_frames, epochs=args.epochs, lr=args.lr, pos_weight=pos_weight)
    tiou, tf1, tacc = run_eval("teacher", teacher, headT, test_loader, device, args.crop, args.num_frames)
    print(f"TEACHER TEST | IoU={tiou:.4f} | F1={tf1:.4f} | Acc={tacc:.4f}")

    ckpts = [c.strip() for c in args.student_ckpts.split(",") if c.strip()]
    for ck in ckpts:
        print(f"\n== STUDENT: {ck} ==")
        student = load_student(ck, device, args.crop, args.num_frames,
                               args.student_embed_dim, args.student_depth, args.student_heads, args.K_reg)

        headS = PatchHead(d_in=args.student_embed_dim, grid=gh).to(device)
        headS = train_head("student", student, headS, train_loader, val_loader, device,
                           args.crop, args.num_frames, epochs=args.epochs, lr=args.lr, pos_weight=pos_weight)
        siou, sf1, sacc = run_eval("student", student, headS, test_loader, device, args.crop, args.num_frames)
        print(f"STUDENT TEST | IoU={siou:.4f} | F1={sf1:.4f} | Acc={sacc:.4f} | F1/teacher={(sf1/(tf1+1e-12)):.3f}")


if __name__ == "__main__":
    main()

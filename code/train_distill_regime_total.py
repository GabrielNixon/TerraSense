# train_distill_regime_total.py
#
# Teacher→Student distillation + (1) relational KD (RKD), (2) patch-mean KD,
# + (3) radiometric invariance, + (4) regime tokens, + (5) temporal operator heads
# (Stack/Diff/Curv supervised by NDVI-derived weak labels during training).
#
# Assumptions (matches your data):
# - DATA_ROOT has subfolders like accra_01/, bogota_01/ ...
# - Each location folder contains: 2019.npz, 2021.npz, 2023.npz
# - Each npz has keys: x (H,W,7 float32), bands (7,), meta (str)
# - bands include: B02,B03,B04,B08,B11,B12,Fmask (order may vary)
#
# Run:
#   cd ~/Desktop/hls_project
#   source ~/.venv/bin/activate
#   python train_distill_regime_total.py \
#       --data_root ./hls_chips_all \
#       --teacher_dir ./prithvi_distill/prithvi_teacher \
#       --teacher_ckpt ./prithvi_distill/prithvi_teacher/Prithvi_EO_V2_300M_TL.pt \
#       --out_dir ./ckpts_regime \
#       --epochs 10 --batch_size 8
#
# Optional: if you have torchgeo installed, you can evaluate EuroSAT during training:
#   pip install torchgeo rasterio
#   add: --eval_eurosat_every 1
#
import os
import json
import math
import glob
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from prithvi_teacher.prithvi_mae import PrithviMAE, PrithviViT


# ----------------------------
# Constants (teacher normalization)
# ----------------------------
TEACHER_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
TEACHER_STD  = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

YEARS = [2019, 2021, 2023]
REQ_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]  # 6 chans
FMASK = "Fmask"


# ----------------------------
# Utils
# ----------------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_pixel_values(x):
    # x: (B,6,T,H,W)
    mean = torch.tensor(TEACHER_MEAN, device=x.device).view(1,6,1,1,1)
    std  = torch.tensor(TEACHER_STD,  device=x.device).view(1,6,1,1,1)
    return (x - mean) / (std + 1e-6)


def maybe_scale_s2(x):
    # x: float tensor
    mx = float(x.max().detach().cpu())
    if mx <= 2.0:
        return x * 10000.0
    return x


def safe_ndvi_mean(x_hwk, band_map):
    # x_hwk: numpy (H,W,K) float32, contains B04,B08 + Fmask
    red = x_hwk[..., band_map["B04"]].astype(np.float32)
    nir = x_hwk[..., band_map["B08"]].astype(np.float32)
    ndvi = (nir - red) / (nir + red + 1e-6)

    if "Fmask" in band_map:
        fmask = x_hwk[..., band_map["Fmask"]]
        valid = (fmask == 0)
        if np.any(valid):
            m = float(np.mean(ndvi[valid]))
            if np.isfinite(m):
                return m

    m = float(np.mean(ndvi))
    if np.isfinite(m):
        return m
    return np.nan


def labels_from_triplet(nd19, nd21, nd23, delta=0.05):
    # net-change 3-class: down / stable / up (for auxiliary head)
    dnet = nd23 - nd19
    if dnet > delta:
        y_net = 2
    elif dnet < -delta:
        y_net = 0
    else:
        y_net = 1

    # curvature 4-class (your definition)
    d01 = nd21 - nd19
    d12 = nd23 - nd21
    if (d01 < -delta) and (d12 < -delta):
        y_curv = 0  # monotonic down
    elif (d01 > delta) and (d12 > delta):
        y_curv = 1  # monotonic up
    elif ((d01 > delta) and (d12 < -delta)) or ((d01 < -delta) and (d12 > delta)):
        y_curv = 2  # oscillatory
    else:
        y_curv = 3  # stable-ish
    return y_net, y_curv


# ----------------------------
# Radiometric augmentations (invariance)
# ----------------------------
def radiometric_augment(x, p=0.9):
    # x: (B,6,T,H,W), in raw reflectance scale (~0..10000)
    if random.random() > p:
        return x

    out = x

    # global brightness scaling (0.95..1.05)
    s = 0.95 + 0.10 * random.random()
    out = out * s

    # per-band gain jitter (small)
    gains = torch.ones((1,6,1,1,1), device=out.device, dtype=out.dtype)
    gains = gains * (0.98 + 0.04 * torch.rand_like(gains))
    out = out * gains

    # NIR-ish offset (channel 3 corresponds to B08 in our 6-ch input)
    # offset up to ~ ±200 (about 2% of 10000)
    nir_offset = (torch.rand((1,1,1,1,1), device=out.device, dtype=out.dtype) - 0.5) * 400.0
    out[:, 3:4, :, :, :] = out[:, 3:4, :, :, :] + nir_offset

    # clamp to non-negative
    out = torch.clamp(out, min=0.0)
    return out


# ----------------------------
# Dataset
# ----------------------------
class HLSTripletDataset(Dataset):
    def __init__(self, root, delta=0.05):
        self.root = root
        self.delta = float(delta)
        self.items = []

        loc_dirs = [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)]
        for d in sorted(loc_dirs):
            ok = True
            paths = {}
            for y in YEARS:
                p = os.path.join(d, f"{y}.npz")
                if not os.path.exists(p):
                    ok = False
                    break
                paths[y] = p
            if ok:
                self.items.append((os.path.basename(d), paths))

        if not self.items:
            raise RuntimeError(f"No valid triplets found under: {root}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        loc_id, paths = self.items[idx]

        xs = []
        ndvis = []

        for y in YEARS:
            z = np.load(paths[y], allow_pickle=True)
            x = z["x"].astype(np.float32)  # (H,W,K)
            bands = [str(b) for b in z["bands"]]
            band_map = {b: i for i, b in enumerate(bands)}

            # NDVI mean (uses B04,B08,Fmask)
            nd = safe_ndvi_mean(x, band_map)
            ndvis.append(nd)

            # select 6 channels for model
            chans = [band_map[b] for b in REQ_BANDS]
            x6 = x[..., chans]  # (H,W,6)

            # scale to ~0..10000 if needed
            if float(np.nanmax(x6)) <= 2.0:
                x6 = x6 * 10000.0

            xs.append(x6)

        nd19, nd21, nd23 = ndvis
        y_net, y_curv = labels_from_triplet(nd19, nd21, nd23, delta=self.delta)

        # stack to (6,T,H,W)
        x_stack = np.stack(xs, axis=0)  # (T,H,W,6)
        x_stack = np.transpose(x_stack, (3,0,1,2))  # (6,T,H,W)

        # coords (dummy; you can later plug true lat/lon)
        # temporal coords: (T,2) - we encode year fraction in first dim
        tcoords = np.zeros((len(YEARS), 2), dtype=np.float32)
        for i, y in enumerate(YEARS):
            tcoords[i, 0] = (y - 2019) / 4.0  # 2019->0, 2023->1
        lcoords = np.zeros((2,), dtype=np.float32)

        return {
            "loc_id": loc_id,
            "x": torch.from_numpy(x_stack),           # (6,T,H,W)
            "tcoords": torch.from_numpy(tcoords),     # (T,2)
            "lcoords": torch.from_numpy(lcoords),     # (2,)
            "y_net": torch.tensor(y_net, dtype=torch.long),
            "y_curv": torch.tensor(y_curv, dtype=torch.long),
        }


# ----------------------------
# Regime token module (cross-attend to patches)
# ----------------------------
class RegimeAggregator(nn.Module):
    def __init__(self, embed_dim=256, K=4, num_heads=4, dropout=0.0):
        super().__init__()
        self.K = K
        self.tokens = nn.Parameter(torch.randn(1, K, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, patch_tokens):
        # patch_tokens: (B, N, D)  (exclude CLS)
        B, N, D = patch_tokens.shape
        q = self.tokens.expand(B, -1, -1)  # (B,K,D)
        q = self.ln_q(q)
        kv = self.ln_kv(patch_tokens)

        # regimes attend to patches
        out, _ = self.attn(q, kv, kv, need_weights=False)  # (B,K,D)
        out = out + self.ff(out)

        # pooled regime embedding
        pooled = out.mean(dim=1)  # (B,D)
        # also return regime distribution proxy via softmax over regime token norms (optional)
        reg_scores = torch.norm(out, dim=-1)  # (B,K)
        reg_prob = torch.softmax(reg_scores, dim=-1)
        return pooled, reg_prob


# ----------------------------
# Student with fused CLS + regime pool + aux heads
# ----------------------------
class StudentRegime(nn.Module):
    def __init__(self, img_size=256, num_frames=3, embed_dim=256, depth=8, heads=4, K_reg=4):
        super().__init__()
        self.img_size = img_size
        self.num_frames = num_frames

        self.enc = PrithviViT(
            img_size=img_size,
            num_frames=num_frames,
            patch_size=(1,16,16),
            in_chans=6,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=heads,
            mlp_ratio=4.0,
            coords_encoding=["time","location"],
            coords_scale_learn=True,
        )

        self.reg = RegimeAggregator(embed_dim=embed_dim, K=K_reg, num_heads=heads)

        # fuse (cls, regime_pool) -> 1024 (teacher dim)
        self.proj_cls = nn.Linear(2 * embed_dim, 1024)
        self.proj_pm  = nn.Linear(2 * embed_dim, 1024)

        # operator heads (weak supervision)
        self.net_head  = nn.Linear(1024, 3)   # down/stable/up
        self.curv_head = nn.Linear(1024, 4)   # curv4

        # regime regularizers
        self.reg_entropy_weight = 0.0  # can be set >0 for anti-collapse

    def forward_once(self, x, tc, lc):
        # x: (B,6,T,H,W)
        feats = self.enc.forward_features(x, tc, lc)
        last = feats[-1]                   # (B, 1+N, D)
        cls = last[:, 0, :]                # (B,D)
        patches = last[:, 1:, :]           # (B,N,D)
        pm = patches.mean(dim=1)           # (B,D)

        reg_pool, reg_prob = self.reg(patches)  # (B,D), (B,K)

        cls_fused = torch.cat([cls, reg_pool], dim=-1)
        pm_fused  = torch.cat([pm,  reg_pool], dim=-1)

        cls1024 = self.proj_cls(cls_fused)
        pm1024  = self.proj_pm(pm_fused)

        return cls1024, pm1024, reg_prob

    def forward(self, x, tc, lc):
        return self.forward_once(x, tc, lc)


# ----------------------------
# Teacher loader
# ----------------------------
def load_teacher(teacher_dir, teacher_ckpt, device, img_size=256, num_frames=3):
    config_path = os.path.join(teacher_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)["pretrained_cfg"]
    cfg = dict(cfg)
    cfg["img_size"] = img_size
    cfg["num_frames"] = num_frames
    cfg["in_chans"] = 6
    cfg["coords_encoding"] = ["time","location"]

    teacher = PrithviMAE(**cfg).to(device).eval()
    sd = torch.load(teacher_ckpt, map_location=device)
    # permissive about positional embeddings
    for k in list(sd.keys()):
        if "pos_embed" in k:
            del sd[k]
    teacher.load_state_dict(sd, strict=False)
    return teacher


@torch.no_grad()
def teacher_embed(teacher, x, tc, lc):
    feats = teacher.forward_features(x, tc, lc)
    last = feats[-1]
    cls = last[:, 0, :]
    pm = last[:, 1:, :].mean(dim=1)
    return cls, pm


# ----------------------------
# RKD (Relational Knowledge Distillation)
# ----------------------------
def pdist(e, squared=False):
    # e: (B,D)
    # returns (B,B)
    prod = torch.mm(e, e.t())
    norm = prod.diag().unsqueeze(1)
    dist = norm - 2 * prod + norm.t()
    dist = torch.clamp(dist, min=0.0)
    if not squared:
        dist = torch.sqrt(dist + 1e-12)
    dist = dist - torch.diag(dist.diag())
    return dist


def rkd_distance_loss(student, teacher):
    # normalize distances by mean
    with torch.no_grad():
        t_d = pdist(teacher, squared=False)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / (mean_td + 1e-12)

    s_d = pdist(student, squared=False)
    mean_sd = s_d[s_d > 0].mean()
    s_d = s_d / (mean_sd + 1e-12)

    return F.smooth_l1_loss(s_d, t_d)


def rkd_angle_loss(student, teacher):
    # angle loss over all triplets (i,j,k) approximated by random sampling
    B = student.shape[0]
    if B < 3:
        return student.sum() * 0.0

    def sample_triplets(B, m=256):
        out = []
        for _ in range(m):
            i = random.randrange(B)
            j = random.randrange(B)
            k = random.randrange(B)
            if len({i,j,k}) < 3:
                continue
            out.append((i,j,k))
        return out

    triplets = sample_triplets(B, m=min(256, B*B))
    if not triplets:
        return student.sum() * 0.0

    s = student
    t = teacher
    loss = 0.0
    for (i,j,k) in triplets:
        vs1 = F.normalize(s[i] - s[j], dim=0)
        vs2 = F.normalize(s[i] - s[k], dim=0)
        vt1 = F.normalize(t[i] - t[j], dim=0)
        vt2 = F.normalize(t[i] - t[k], dim=0)
        asim = (vs1 * vs2).sum()
        atim = (vt1 * vt2).sum()
        loss = loss + F.smooth_l1_loss(asim, atim)
    return loss / float(len(triplets))


def rkd_loss(student, teacher, w_dist=1.0, w_angle=2.0):
    return w_dist * rkd_distance_loss(student, teacher) + w_angle * rkd_angle_loss(student, teacher)


# ----------------------------
# Optional: EuroSAT FAST eval (with pos_embed interpolation for student)
# ----------------------------
def interpolate_pos_embed_2d(pos_embed, new_grid_h, new_grid_w):
    # pos_embed: (1, 1+N, D) with 1 cls + N grid tokens
    # returns resized pos_embed (1, 1+newN, D)
    cls = pos_embed[:, :1, :]
    tok = pos_embed[:, 1:, :]  # (1,N,D)
    N = tok.shape[1]
    D = tok.shape[2]
    gs = int(math.sqrt(N))
    if gs * gs != N:
        return pos_embed  # cannot safely reshape

    tok = tok.reshape(1, gs, gs, D).permute(0,3,1,2)  # (1,D,gs,gs)
    tok = F.interpolate(tok, size=(new_grid_h, new_grid_w), mode="bilinear", align_corners=False)
    tok = tok.permute(0,2,3,1).reshape(1, new_grid_h * new_grid_w, D)
    return torch.cat([cls, tok], dim=1)


@torch.no_grad()
def eval_eurosat_fast(teacher, student, device, img_size=64, num_frames=3, batch_size=8, limit=None):
    try:
        from torchgeo.datasets import EuroSAT
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader
    except Exception as e:
        print("EuroSAT eval skipped (missing deps torchgeo/sklearn):", str(e))
        return None

    EUROSAT_BANDS = ("B02","B03","B04","B08","B11","B12")
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_eurosat")

    ds = EuroSAT(root=root, split="train", bands=EUROSAT_BANDS, download=True)
    if limit is not None and limit < len(ds):
        idx = np.random.RandomState(42).choice(len(ds), limit, replace=False)
        ds = torch.utils.data.Subset(ds, idx)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

    # Build an evaluation-time student clone at img_size=64 and interpolate pos_embed
    # so we don't trash performance by dropping pos_embed.
    stud_eval = StudentRegime(img_size=img_size, num_frames=num_frames).to(device).eval()
    sd = student.state_dict()

    # interpolate student pos_embed if present
    key = "enc.pos_embed"
    if key in sd:
        pe = sd[key]  # (1, 1+N, 256)
        # grid for img_size=64, patch=16 => 4x4 per frame? prithvi uses (1,16,16) patch in time too,
        # but pos_embed in their ViT usually only spatial tokens per-frame are flattened across time.
        # Empirically your 64x64+T=3 leads to 49 tokens => 1 + 48 => 16 per frame -> 4x4.
        new_grid = 4  # 64/16
        pe2 = interpolate_pos_embed_2d(pe, new_grid, new_grid)
        sd[key] = pe2

    stud_eval.load_state_dict(sd, strict=False)

    def norm_x(x):
        x = x.float()
        mx = float(x.max().detach().cpu())
        if mx <= 2.0:
            x = x * 10000.0
        # x: (B,6,64,64) -> (B,6,T,64,64)
        x = x.unsqueeze(2).repeat(1,1,num_frames,1,1)
        tc = torch.zeros((x.shape[0], num_frames, 2), device=device)
        lc = torch.zeros((x.shape[0], 2), device=device)
        x = normalize_pixel_values(x)
        return x, tc, lc

    Temb = []
    Semb = []
    Y = []
    for batch in loader:
        img = batch["image"].to(device)
        y = batch["label"].cpu().numpy()
        x, tc, lc = norm_x(img)

        tcls, _ = teacher_embed(teacher, x, tc, lc)
        scls, _, _ = stud_eval.forward_once(x, tc, lc)

        Temb.append(tcls.cpu().numpy())
        Semb.append(scls.cpu().numpy())
        Y.append(y)

    Temb = np.concatenate(Temb, axis=0)
    Semb = np.concatenate(Semb, axis=0)
    Y = np.concatenate(Y, axis=0)

    def probe(X, y):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        clf = LogisticRegression(max_iter=4000)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        return float(accuracy_score(yte, pred))

    accT = probe(Temb, Y)
    accS = probe(Semb, Y)
    print(f"[EuroSAT FAST] Teacher acc={accT:.4f} | Student acc={accS:.4f}")
    return accT, accS


# ----------------------------
# Training
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--teacher_dir", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./ckpts_regime")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)

    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--num_frames", type=int, default=3)

    ap.add_argument("--K_reg", type=int, default=4)

    # loss weights
    ap.add_argument("--w_kd_cls", type=float, default=1.0)
    ap.add_argument("--w_kd_pm", type=float, default=1.0)
    ap.add_argument("--w_rkd", type=float, default=2.0)
    ap.add_argument("--w_inv", type=float, default=0.2)
    ap.add_argument("--w_net", type=float, default=0.5)
    ap.add_argument("--w_curv", type=float, default=1.0)
    ap.add_argument("--w_reg_entropy", type=float, default=0.05)

    ap.add_argument("--embed_dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=4)


    # eval
    ap.add_argument("--eval_eurosat_every", type=int, default=0)  # 0 disables
    ap.add_argument("--eurosat_limit", type=int, default=8000)    # speed; None for full
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = pick_device()
    print("device:", device)
    print("data_root:", args.data_root)

    # dataset
    ds = HLSTripletDataset(args.data_root, delta=args.delta)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # models
    teacher = load_teacher(args.teacher_dir, args.teacher_ckpt, device, img_size=args.img_size, num_frames=args.num_frames)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = StudentRegime(
        img_size=args.img_size,
        num_frames=args.num_frames,
        embed_dim=args.embed_dim,
        depth=args.depth,
        heads=args.heads,
        K_reg=args.K_reg,
    ).to(device).train()

    student.reg_entropy_weight = args.w_reg_entropy

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training
    global_step = 0
    for ep in range(1, args.epochs + 1):
        student.train()
        losses = []

        for batch in dl:
            x = batch["x"].to(device).float()              # (B,6,T,H,W)
            tc = batch["tcoords"].to(device).float()       # (B,T,2)
            lc = batch["lcoords"].to(device).float()       # (B,2)
            y_net = batch["y_net"].to(device)
            y_curv = batch["y_curv"].to(device)

            # reflectance scaling + normalization
            x = maybe_scale_s2(x)
            x = normalize_pixel_values(x)

            # teacher embeddings (triplet)
            with torch.no_grad():
                t_cls, t_pm = teacher_embed(teacher, x, tc, lc)

            # student embeddings (triplet)
            s_cls, s_pm, reg_prob = student(x, tc, lc)

            # KD losses
            kd_cls = F.mse_loss(s_cls, t_cls)
            kd_pm  = F.mse_loss(s_pm,  t_pm)

            # RKD relational loss on CLS (structure preservation)
            rkd = rkd_loss(s_cls, t_cls)

            # invariance loss (radiometric)
            x_raw = maybe_scale_s2(batch["x"].to(device).float())
            x_aug = radiometric_augment(x_raw, p=0.9)
            x_raw = normalize_pixel_values(x_raw)
            x_aug = normalize_pixel_values(x_aug)

            s_raw, _, _ = student.forward_once(x_raw, tc, lc)
            s_aug, _, _ = student.forward_once(x_aug, tc, lc)
            inv = (1.0 - F.cosine_similarity(s_raw, s_aug, dim=-1)).mean()

            # temporal operator heads (weak supervision from NDVI triplet)
            # We need per-year embeddings. Easiest: run 3 passes with that year repeated across frames.
            # This is slower but simple and stable.
            def year_embed(frame_idx):
                xx = batch["x"].to(device).float()  # raw
                xx = maybe_scale_s2(xx)
                # pick the frame (6, T, H, W) -> (6,1,H,W) then repeat to T
                f = xx[:, :, frame_idx:frame_idx+1, :, :]         # (B,6,1,H,W)
                f = f.repeat(1, 1, args.num_frames, 1, 1)         # (B,6,T,H,W)
                f = normalize_pixel_values(f)
                # keep same coords (or could zero tc); we keep tc for consistency
                e, _, _ = student.forward_once(f, tc, lc)         # (B,1024)
                return e

            e19 = year_embed(0)
            e21 = year_embed(1)
            e23 = year_embed(2)

            # net delta and curvature embedding
            net_vec  = (e23 - e19)
            curv_vec = (e23 - e21) - (e21 - e19)

            logits_net  = student.net_head(net_vec)
            logits_curv = student.curv_head(curv_vec)

            loss_net  = F.cross_entropy(logits_net, y_net)
            loss_curv = F.cross_entropy(logits_curv, y_curv)

            # regime anti-collapse regularizer (encourage non-degenerate regime usage)
            # maximize entropy => minimize negative entropy
            ent = -(reg_prob * torch.log(reg_prob + 1e-8)).sum(dim=-1).mean()
            reg_loss = -ent  # minimize -entropy -> maximize entropy

            loss = (
                args.w_kd_cls * kd_cls +
                args.w_kd_pm  * kd_pm +
                args.w_rkd    * rkd +
                args.w_inv    * inv +
                args.w_net    * loss_net +
                args.w_curv   * loss_curv +
                args.w_reg_entropy * reg_loss
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            global_step += 1

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        ckpt_path = os.path.join(args.out_dir, f"student_regime_ep{ep}.pt")
        torch.save(student.state_dict(), ckpt_path)
        print(f"epoch {ep} | loss {mean_loss:.4f} | saved {ckpt_path}")

        # optional EuroSAT eval (FAST, smaller image size + pos_embed interpolation)
        if args.eval_eurosat_every and (ep % args.eval_eurosat_every == 0):
            student.eval()
            _ = eval_eurosat_fast(
                teacher=teacher,
                student=student,
                device=device,
                img_size=64,
                num_frames=args.num_frames,
                batch_size=8,
                limit=args.eurosat_limit
            )

    print("done.")


if __name__ == "__main__":
    main()


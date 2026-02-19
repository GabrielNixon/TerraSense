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
from torch.utils.data import Dataset, DataLoader, Subset

import optuna
from optuna.pruners import MedianPruner

from prithvi_teacher.prithvi_mae import PrithviMAE, PrithviViT

TEACHER_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
TEACHER_STD  = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

YEARS = [2019, 2021, 2023]
REQ_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def normalize_pixel_values(x):
    mean = torch.tensor(TEACHER_MEAN, device=x.device).view(1,6,1,1,1)
    std  = torch.tensor(TEACHER_STD,  device=x.device).view(1,6,1,1,1)
    return (x - mean) / (std + 1e-6)

def maybe_scale_s2(x):
    mx = float(x.max().detach().cpu())
    if mx <= 2.0:
        return x * 10000.0
    return x

def safe_ndvi_mean(x_hwk, band_map):
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
    dnet = nd23 - nd19
    if dnet > delta:
        y_net = 2
    elif dnet < -delta:
        y_net = 0
    else:
        y_net = 1

    d01 = nd21 - nd19
    d12 = nd23 - nd21
    if (d01 < -delta) and (d12 < -delta):
        y_curv = 0
    elif (d01 > delta) and (d12 > delta):
        y_curv = 1
    elif ((d01 > delta) and (d12 < -delta)) or ((d01 < -delta) and (d12 > delta)):
        y_curv = 2
    else:
        y_curv = 3
    return y_net, y_curv

def radiometric_augment(x, p=0.9):
    if random.random() > p:
        return x
    out = x
    s = 0.95 + 0.10 * random.random()
    out = out * s
    gains = torch.ones((1,6,1,1,1), device=out.device, dtype=out.dtype)
    gains = gains * (0.98 + 0.04 * torch.rand_like(gains))
    out = out * gains
    nir_offset = (torch.rand((1,1,1,1,1), device=out.device, dtype=out.dtype) - 0.5) * 400.0
    out[:, 3:4, :, :, :] = out[:, 3:4, :, :, :] + nir_offset
    out = torch.clamp(out, min=0.0)
    return out

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
            x = z["x"].astype(np.float32)
            bands = [str(b) for b in z["bands"]]
            band_map = {b: i for i, b in enumerate(bands)}
            nd = safe_ndvi_mean(x, band_map)
            ndvis.append(nd)
            chans = [band_map[b] for b in REQ_BANDS]
            x6 = x[..., chans]
            if float(np.nanmax(x6)) <= 2.0:
                x6 = x6 * 10000.0
            xs.append(x6)

        nd19, nd21, nd23 = ndvis
        y_net, y_curv = labels_from_triplet(nd19, nd21, nd23, delta=self.delta)

        x_stack = np.stack(xs, axis=0)             # (T,H,W,6)
        x_stack = np.transpose(x_stack, (3,0,1,2)) # (6,T,H,W)

        tcoords = np.zeros((len(YEARS), 2), dtype=np.float32)
        for i, y in enumerate(YEARS):
            tcoords[i, 0] = (y - 2019) / 4.0
        lcoords = np.zeros((2,), dtype=np.float32)

        return {
            "loc_id": loc_id,
            "x": torch.from_numpy(x_stack),
            "tcoords": torch.from_numpy(tcoords),
            "lcoords": torch.from_numpy(lcoords),
            "y_net": torch.tensor(y_net, dtype=torch.long),
            "y_curv": torch.tensor(y_curv, dtype=torch.long),
        }

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
        B, N, D = patch_tokens.shape
        q = self.tokens.expand(B, -1, -1)
        q = self.ln_q(q)
        kv = self.ln_kv(patch_tokens)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = out + self.ff(out)
        pooled = out.mean(dim=1)
        reg_scores = torch.norm(out, dim=-1)
        reg_prob = torch.softmax(reg_scores, dim=-1)
        return pooled, reg_prob

class StudentRegime(nn.Module):
    def __init__(self, img_size=256, num_frames=3, embed_dim=256, depth=8, heads=4, K_reg=4):
        super().__init__()
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
        self.proj_cls = nn.Linear(2 * embed_dim, 1024)
        self.proj_pm  = nn.Linear(2 * embed_dim, 1024)
        self.net_head  = nn.Linear(1024, 3)
        self.curv_head = nn.Linear(1024, 4)

    def forward_once(self, x, tc, lc):
        feats = self.enc.forward_features(x, tc, lc)
        last = feats[-1]
        cls = last[:, 0, :]
        patches = last[:, 1:, :]
        pm = patches.mean(dim=1)
        reg_pool, reg_prob = self.reg(patches)
        cls1024 = self.proj_cls(torch.cat([cls, reg_pool], dim=-1))
        pm1024  = self.proj_pm(torch.cat([pm,  reg_pool], dim=-1))
        return cls1024, pm1024, reg_prob

    def forward(self, x, tc, lc):
        return self.forward_once(x, tc, lc)

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

def pdist(e, squared=False):
    prod = torch.mm(e, e.t())
    norm = prod.diag().unsqueeze(1)
    dist = norm - 2 * prod + norm.t()
    dist = torch.clamp(dist, min=0.0)
    if not squared:
        dist = torch.sqrt(dist + 1e-12)
    dist = dist - torch.diag(dist.diag())
    return dist

def rkd_distance_loss(student, teacher):
    with torch.no_grad():
        t_d = pdist(teacher, squared=False)
        mean_td = t_d[t_d > 0].mean()
        t_d = t_d / (mean_td + 1e-12)
    s_d = pdist(student, squared=False)
    mean_sd = s_d[s_d > 0].mean()
    s_d = s_d / (mean_sd + 1e-12)
    return F.smooth_l1_loss(s_d, t_d)

def rkd_angle_loss(student, teacher):
    B = student.shape[0]
    if B < 3:
        return student.sum() * 0.0

    m = min(256, B * B)
    triplets = []
    for _ in range(m):
        i = random.randrange(B)
        j = random.randrange(B)
        k = random.randrange(B)
        if len({i, j, k}) < 3:
            continue
        triplets.append((i, j, k))
    if not triplets:
        return student.sum() * 0.0

    loss = 0.0
    for (i, j, k) in triplets:
        vs1 = F.normalize(student[i] - student[j], dim=0)
        vs2 = F.normalize(student[i] - student[k], dim=0)
        vt1 = F.normalize(teacher[i] - teacher[j], dim=0)
        vt2 = F.normalize(teacher[i] - teacher[k], dim=0)
        loss = loss + F.smooth_l1_loss((vs1 * vs2).sum(), (vt1 * vt2).sum())
    return loss / float(len(triplets))

def rkd_loss(student, teacher, w_dist=1.0, w_angle=2.0):
    return w_dist * rkd_distance_loss(student, teacher) + w_angle * rkd_angle_loss(student, teacher)

def split_indices(n, val_frac=0.15, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    nval = int(round(val_frac * n))
    val_idx = idx[:nval]
    tr_idx = idx[nval:]
    return tr_idx.tolist(), val_idx.tolist()

@torch.no_grad()
def eval_val_loss(student, teacher, dl, device):
    student.eval()
    losses = []
    for batch in dl:
        x = maybe_scale_s2(batch["x"].to(device).float())
        x = normalize_pixel_values(x)
        tc = batch["tcoords"].to(device).float()
        lc = batch["lcoords"].to(device).float()
        t_cls, t_pm = teacher_embed(teacher, x, tc, lc)
        s_cls, s_pm, _ = student(x, tc, lc)
        kd = F.mse_loss(s_cls, t_cls) + F.mse_loss(s_pm, t_pm)
        losses.append(float(kd.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")

def train_one_epoch(student, teacher, dl, device, opt, args):
    student.train()
    losses = []
    for batch in dl:
        x_raw = maybe_scale_s2(batch["x"].to(device).float())
        tc = batch["tcoords"].to(device).float()
        lc = batch["lcoords"].to(device).float()
        y_net = batch["y_net"].to(device)
        y_curv = batch["y_curv"].to(device)

        x = normalize_pixel_values(x_raw)

        with torch.no_grad():
            t_cls, t_pm = teacher_embed(teacher, x, tc, lc)

        s_cls, s_pm, reg_prob = student(x, tc, lc)
        kd_cls = F.mse_loss(s_cls, t_cls)
        kd_pm  = F.mse_loss(s_pm,  t_pm)
        rkd = rkd_loss(s_cls, t_cls)

        x_aug = radiometric_augment(x_raw, p=args.inv_p)
        s_raw, _, _ = student.forward_once(normalize_pixel_values(x_raw), tc, lc)
        s_aug, _, _ = student.forward_once(normalize_pixel_values(x_aug), tc, lc)
        inv = (1.0 - F.cosine_similarity(s_raw, s_aug, dim=-1)).mean()

        def year_embed(frame_idx):
            f = x_raw[:, :, frame_idx:frame_idx+1, :, :].repeat(1, 1, args.num_frames, 1, 1)
            e, _, _ = student.forward_once(normalize_pixel_values(f), tc, lc)
            return e

        e19 = year_embed(0)
        e21 = year_embed(1)
        e23 = year_embed(2)

        net_vec  = (e23 - e19)
        curv_vec = (e23 - e21) - (e21 - e19)

        logits_net  = student.net_head(net_vec)
        logits_curv = student.curv_head(curv_vec)

        loss_net  = F.cross_entropy(logits_net, y_net)
        loss_curv = F.cross_entropy(logits_curv, y_curv)

        ent = -(reg_prob * torch.log(reg_prob + 1e-8)).sum(dim=-1).mean()
        reg_loss = -ent

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
    return float(np.mean(losses)) if losses else float("inf")

MODEL_SCALES = {
    "tiny":  dict(embed_dim=192, depth=6,  heads=3),
    "small": dict(embed_dim=256, depth=8,  heads=4),
    "base":  dict(embed_dim=384, depth=10, heads=6),
    "large": dict(embed_dim=512, depth=12, heads=8),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--teacher_dir", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="./ckpts_regime_optuna")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--num_frames", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--trial_epochs", type=int, default=3)
    ap.add_argument("--train_limit", type=int, default=0)
    ap.add_argument("--val_limit", type=int, default=0)

    ap.add_argument("--study_name", type=str, default="distill_regime")
    ap.add_argument("--storage", type=str, default="sqlite:///optuna_distill_regime.db")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print("device:", device)

    ds_full = HLSTripletDataset(args.data_root, delta=0.05)
    tr_idx, val_idx = split_indices(len(ds_full), val_frac=args.val_frac, seed=args.seed)

    if args.train_limit and args.train_limit < len(tr_idx):
        tr_idx = tr_idx[:args.train_limit]
    if args.val_limit and args.val_limit < len(val_idx):
        val_idx = val_idx[:args.val_limit]

    ds_tr = Subset(ds_full, tr_idx)
    ds_val = Subset(ds_full, val_idx)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    teacher = load_teacher(args.teacher_dir, args.teacher_ckpt, device, img_size=args.img_size, num_frames=args.num_frames)
    for p in teacher.parameters():
        p.requires_grad_(False)

    def objective(trial: optuna.Trial):
        scale = trial.suggest_categorical("model_scale", ["tiny", "small", "base", "large"])
        sconf = MODEL_SCALES[scale]

        K_reg = trial.suggest_int("K_reg", 2, 6)
        lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
        wd = trial.suggest_float("weight_decay", 0.01, 0.1, log=True)

        w_kd_cls = trial.suggest_float("w_kd_cls", 0.5, 2.0)
        w_kd_pm  = trial.suggest_float("w_kd_pm",  0.5, 2.0)
        w_rkd    = trial.suggest_float("w_rkd",    0.5, 4.0)
        w_inv    = trial.suggest_float("w_inv",    0.0, 0.5)
        w_net    = trial.suggest_float("w_net",    0.0, 1.0)
        w_curv   = trial.suggest_float("w_curv",   0.0, 2.0)
        w_reg_entropy = trial.suggest_float("w_reg_entropy", 0.0, 0.2)
        inv_p = trial.suggest_float("inv_p", 0.6, 0.95)

        student = StudentRegime(
            img_size=args.img_size,
            num_frames=args.num_frames,
            embed_dim=sconf["embed_dim"],
            depth=sconf["depth"],
            heads=sconf["heads"],
            K_reg=K_reg,
        ).to(device)

        opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)

        trial_args = argparse.Namespace(
            num_frames=args.num_frames,
            w_kd_cls=w_kd_cls, w_kd_pm=w_kd_pm, w_rkd=w_rkd,
            w_inv=w_inv, w_net=w_net, w_curv=w_curv, w_reg_entropy=w_reg_entropy,
            inv_p=inv_p,
        )

        for ep in range(1, args.trial_epochs + 1):
            tr_loss = train_one_epoch(student, teacher, dl_tr, device, opt, trial_args)
            val_loss = eval_val_loss(student, teacher, dl_val, device)

            trial.report(val_loss, step=ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return val_loss

    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    study.optimize(objective, n_trials=args.trials)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": study.best_trial.number,
    }
    outp = os.path.join(args.out_dir, "optuna_best.json")
    with open(outp, "w") as f:
        json.dump(best, f, indent=2)
    print("BEST:", json.dumps(best, indent=2))
    print("saved:", outp)

if __name__ == "__main__":
    main()

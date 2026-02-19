import os
import json
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import snapshot_download
from sklearn.metrics import accuracy_score, f1_score

from prithvi_teacher.prithvi_mae import PrithviMAE

TEACHER_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
TEACHER_STD  = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

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

def load_teacher(teacher_dir, teacher_ckpt, device, img_size=64, num_frames=3):
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

# def load_student(student_ckpt, device, img_size=64, num_frames=3, K_reg=4,
#                  embed_dim=256, depth=8, heads=4):
#     from train_distill_regime_total import StudentRegime
#     student = StudentRegime(
#         img_size=img_size,
#         num_frames=num_frames,
#         embed_dim=embed_dim,
#         depth=depth,
#         heads=heads,
#         K_reg=K_reg,
#     ).to(device).eval()
#     sd = torch.load(student_ckpt, map_location=device)
#     student.load_state_dict(sd, strict=False)
#     return student

def load_student(student_ckpt, device, img_size=64, num_frames=3, K_reg=4,
                 embed_dim=256, depth=8, heads=4):
    from train_distill_regime_total import StudentRegime

    student = StudentRegime(
        img_size=img_size,
        num_frames=num_frames,
        embed_dim=embed_dim,
        depth=depth,
        heads=heads,
        K_reg=K_reg,
    ).to(device).eval()

    sd = torch.load(student_ckpt, map_location=device)

    # 🔥 Minimal fix: remove positional embedding (size mismatch)
    for k in list(sd.keys()):
        if "pos_embed" in k:
            del sd[k]

    student.load_state_dict(sd, strict=False)
    return student


@torch.no_grad()
def student_embed(student, x, tc, lc):
    s_cls, s_pm, _ = student(x, tc, lc)
    return s_cls, s_pm

def parse_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def download_timesen2crop(local_dir):
    repo_id = "monster-monash/TimeSen2Crop"
    allow = [
        "TimeSen2Crop_X.npy",
        "TimeSen2Crop_y.npy",
        "TimeSen2Crop_meta.npy",
        "test_indices_fold_*.txt",
        "README.md",
    ]
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow,
    )
    return path

def load_split(repo_path, fold):
    X_path = os.path.join(repo_path, "TimeSen2Crop_X.npy")
    y_path = os.path.join(repo_path, "TimeSen2Crop_y.npy")
    idx_path = os.path.join(repo_path, f"test_indices_fold_{fold}.txt")

    X = np.load(X_path, mmap_mode="r")  # (N, 9, 365) likely
    y = np.load(y_path, mmap_mode="r")  # (N,)

    test_idx = np.loadtxt(idx_path, dtype=np.int64)
    test_idx = np.unique(test_idx)

    N = y.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[test_idx] = False
    train_idx = np.nonzero(mask)[0]

    return X, y, train_idx, test_idx

def build_batch_from_timeseries(X_batch, days, band_idx, img_size, device):
    # X_batch: (B, 9, 365)
    B = X_batch.shape[0]
    T = len(days)

    xs = []
    for d in days:
        v = X_batch[:, band_idx, d]  # (B,6)
        xs.append(v)
    x = np.stack(xs, axis=2)  # (B,6,T)

    x = torch.from_numpy(x).float().to(device)
    x = x[:, :, :, None, None].expand(B, 6, T, img_size, img_size).contiguous()

    tc = torch.zeros((B, T, 2), device=device)
    lc = torch.zeros((B, 2), device=device)

    x = maybe_scale_s2(x)
    x = normalize_pixel_values(x)
    return x, tc, lc

def collect_embeddings(embed_fn, X, y, indices, batch_size, days, band_idx, img_size, device, use="cls", n_samples=0):
    if n_samples and n_samples < len(indices):
        indices = indices[:n_samples]

    Xs = []
    ys = []

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i:i+batch_size]
        Xb = np.asarray(X[batch_idx])  # materialize this batch
        yb = np.asarray(y[batch_idx]).astype(np.int64)

        x, tc, lc = build_batch_from_timeseries(Xb, days, band_idx, img_size, device)
        cls, pm = embed_fn(x, tc, lc)

        if use == "cls":
            E = cls
        elif use == "pm":
            E = pm
        else:
            E = torch.cat([cls, pm], dim=1)

        Xs.append(E.detach().cpu())
        ys.append(torch.from_numpy(yb))

    Xs = torch.cat(Xs, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    return Xs, ys

def train_linear_probe(Xtr, ytr, Xte, yte, num_classes, epochs=8, lr=1e-3, wd=1e-4, device="cpu"):
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr).long().to(device)
    Xte_t = torch.from_numpy(Xte).float().to(device)

    D = Xtr.shape[1]
    head = nn.Linear(D, num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    head.train()
    for _ in range(epochs):
        logits = head(Xtr_t)
        loss = F.cross_entropy(logits, ytr_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        pred = head(Xte_t).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro", zero_division=0)
    return float(acc), float(f1m)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", type=str, default="./timesen2crop_cache")
    ap.add_argument("--fold", type=int, default=0, choices=[0,1,2,3,4])

    ap.add_argument("--train_n", type=int, default=50000)  # 0 = all (slow)
    ap.add_argument("--test_n", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=64)

    ap.add_argument("--days", type=str, default="60,180,300")         # choose 3 timesteps
    ap.add_argument("--band_idx", type=str, default="0,1,2,6,7,8")    # map to your 6 bands
    ap.add_argument("--use", type=str, default="cls", choices=["cls","pm","cat"])

    ap.add_argument("--teacher_dir", type=str, required=True)
    ap.add_argument("--teacher_ckpt", type=str, required=True)

    ap.add_argument("--student_ckpts", type=str, default="")  # comma-separated
    ap.add_argument("--student_embed_dim", type=int, default=256)
    ap.add_argument("--student_depth", type=int, default=8)
    ap.add_argument("--student_heads", type=int, default=4)
    ap.add_argument("--K_reg", type=int, default=4)

    ap.add_argument("--probe_epochs", type=int, default=8)
    ap.add_argument("--probe_lr", type=float, default=1e-3)
    ap.add_argument("--probe_wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device()
    print("device:", device)

    days = parse_int_list(args.days)
    band_idx = parse_int_list(args.band_idx)
    if len(days) != 3:
        raise ValueError("--days must have exactly 3 indices (T=3).")
    if len(band_idx) != 6:
        raise ValueError("--band_idx must have exactly 6 indices (your 6-band input).")

    repo_path = args.cache_dir
    X, y, train_idx, test_idx = load_split(repo_path, args.fold)

    num_classes = int(np.max(y)) + 1
    print("N:", y.shape[0], "classes:", num_classes, "train:", len(train_idx), "test:", len(test_idx))

    teacher = load_teacher(args.teacher_dir, args.teacher_ckpt, device,
                           img_size=args.img_size, num_frames=3)

    def teacher_fn(x, tc, lc):
        return teacher_embed(teacher, x, tc, lc)

    print("\n[Teacher] embedding train...")
    Xtr_T, ytr = collect_embeddings(teacher_fn, X, y, train_idx, args.batch_size,
                                    days, band_idx, args.img_size, device, use=args.use, n_samples=args.train_n)
    print("[Teacher] embedding test...")
    Xte_T, yte = collect_embeddings(teacher_fn, X, y, test_idx, args.batch_size,
                                    days, band_idx, args.img_size, device, use=args.use, n_samples=args.test_n)

    print("[Teacher] probe...")
    accT, f1T = train_linear_probe(Xtr_T, ytr, Xte_T, yte, num_classes,
                                   epochs=args.probe_epochs, lr=args.probe_lr, wd=args.probe_wd, device=device)
    print(f"TEACHER | acc={accT:.4f} | macroF1={f1T:.4f}")

    ckpts = [c.strip() for c in args.student_ckpts.split(",") if c.strip()]
    for ck in ckpts:
        print(f"\n[Student] {ck}")
        student = load_student(
            student_ckpt=ck,
            device=device,
            img_size=args.img_size,
            num_frames=3,
            K_reg=args.K_reg,
            embed_dim=args.student_embed_dim,
            depth=args.student_depth,
            heads=args.student_heads,
        )

        def student_fn(x, tc, lc):
            return student_embed(student, x, tc, lc)

        print("[Student] embedding train...")
        Xtr_S, _ = collect_embeddings(student_fn, X, y, train_idx, args.batch_size,
                                      days, band_idx, args.img_size, device, use=args.use, n_samples=args.train_n)
        print("[Student] embedding test...")
        Xte_S, _ = collect_embeddings(student_fn, X, y, test_idx, args.batch_size,
                                      days, band_idx, args.img_size, device, use=args.use, n_samples=args.test_n)

        print("[Student] probe...")
        accS, f1S = train_linear_probe(Xtr_S, ytr, Xte_S, yte, num_classes,
                                       epochs=args.probe_epochs, lr=args.probe_lr, wd=args.probe_wd, device=device)
        print(f"STUDENT | acc={accS:.4f} | macroF1={f1S:.4f} | acc/teacher={(accS/(accT+1e-12)):.3f}")

if __name__ == "__main__":
    main()

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from torchgeo.datasets import EuroSAT  # MSI (13-band) dataset
from prithvi_teacher.prithvi_mae import PrithviMAE, PrithviViT

# ----------------------------
# Paths (edit if needed)
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEACHER_DIR = os.path.join(SCRIPT_DIR, "prithvi_distill", "prithvi_teacher")
TEACHER_CKPT = os.path.join(TEACHER_DIR, "Prithvi_EO_V2_300M_TL.pt")
TEACHER_CONFIG = os.path.join(TEACHER_DIR, "config.json")

STUDENT_CKPT = os.path.join(SCRIPT_DIR, "ckpts", "student_ep5.pt")
EUROSAT_ROOT = os.path.join(SCRIPT_DIR, "data_eurosat")  # dataset will be downloaded here

# ----------------------------
# Model expectations
# ----------------------------
IMG_SIZE = 256
NUM_FRAMES = 3

# We will pull these from EuroSAT MSI, then map to teacher channels
EUROSAT_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
# Map to teacher’s 6 “channels” (Prithvi-EO-2.0-TL expects B02,B03,B04,B05,B06,B07)
# Using your earlier mapping: B08->B05, B11->B06, B12->B07
# So effectively, our 6 channels are: [B02,B03,B04,B08,B11,B12] but interpreted as [B02,B03,B04,B05,B06,B07]

TEACHER_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
TEACHER_STD  = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

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
    # TorchGeo EuroSAT MSI often returns reflectance as integers 0..10000.
    # If values look like 0..1 floats, scale by 10000.
    mx = float(x.max().detach().cpu())
    if mx <= 2.0:
        return x * 10000.0
    return x

def upsample_to_256(x):
    # x: (B, C, H, W)
    if x.shape[-1] == IMG_SIZE and x.shape[-2] == IMG_SIZE:
        return x
    return F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

# ----------------------------
# Teacher + Student loaders
# ----------------------------
import json

def load_teacher(device):
    with open(TEACHER_CONFIG, "r") as f:
        cfg = json.load(f)["pretrained_cfg"]
    cfg = dict(cfg)
    cfg["img_size"] = IMG_SIZE
    cfg["num_frames"] = NUM_FRAMES
    cfg["in_chans"] = 6
    cfg["coords_encoding"] = ["time", "location"]

    teacher = PrithviMAE(**cfg).to(device).eval()
    sd = torch.load(TEACHER_CKPT, map_location=device)
    # drop pos_embed if present (common for size mismatches)
    for k in list(sd.keys()):
        if "pos_embed" in k:
            del sd[k]
    teacher.load_state_dict(sd, strict=False)
    return teacher

class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = PrithviViT(
            img_size=IMG_SIZE,
            num_frames=NUM_FRAMES,
            patch_size=(1,16,16),
            in_chans=6,
            embed_dim=256,
            depth=8,
            num_heads=4,
            mlp_ratio=4.0,
            coords_encoding=["time","location"],
            coords_scale_learn=True,
        )
        self.proj = torch.nn.Linear(256, 1024)

    def forward(self, x, tc, lc):
        feats = self.enc.forward_features(x, tc, lc)
        last = feats[-1]
        cls = last[:, 0, :]
        pm = last[:, 1:, :].mean(dim=1)
        return self.proj(cls), self.proj(pm)

def load_student(device):
    m = Student().to(device).eval()
    m.load_state_dict(torch.load(STUDENT_CKPT, map_location=device))
    return m

# ----------------------------
# Embedding extraction
# ----------------------------
@torch.no_grad()
def embed_dataset(teacher, student, loader, device):
    t_embs = []
    s_embs = []
    ys = []

    for batch in loader:
        # TorchGeo returns dict with keys: "image" (C,H,W), "label" (int)
        img = batch["image"].to(device)  # (B,6,H,W) because we selected 6 bands
        y = batch["label"].cpu().numpy()

        img = maybe_scale_s2(img)
        img = upsample_to_256(img)

        # Make T=3 frames by repeating
        # (B,6,H,W) -> (B,6,T,H,W)
        x = img.unsqueeze(2).repeat(1, 1, NUM_FRAMES, 1, 1)

        # Dummy coords (works fine for linear probe)
        tc = torch.zeros((x.shape[0], NUM_FRAMES, 2), device=device)  # (B,T,2)
        lc = torch.zeros((x.shape[0], 2), device=device)              # (B,2)

        x = normalize_pixel_values(x)

        # Teacher CLS
        t_feats = teacher.forward_features(x, tc, lc)
        t_last = t_feats[-1]
        t_cls = t_last[:, 0, :]

        # Student projected CLS (1024-dim to match teacher)
        s_cls, _ = student(x, tc, lc)

        t_embs.append(t_cls.cpu().numpy())
        s_embs.append(s_cls.cpu().numpy())
        ys.append(y)

    T = np.concatenate(t_embs, axis=0)
    S = np.concatenate(s_embs, axis=0)
    Y = np.concatenate(ys, axis=0)
    return T, S, Y

def linear_probe(X, y, name):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=4000, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"{name} EuroSAT MSI linear-probe accuracy: {acc:.4f}")

def main():
    device = pick_device()
    print("device:", device)

    # EuroSAT MSI via TorchGeo, selecting the bands we need
    # TorchGeo EuroSAT supports 'bands=' argument. :contentReference[oaicite:3]{index=3}
    ds = EuroSAT(root=EUROSAT_ROOT, split="train", bands=EUROSAT_BANDS, download=True)

    # EuroSAT has only "train" split in TorchGeo; we'll do our own train/test split after embedding.
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2, drop_last=False)

    teacher = load_teacher(device)
    student = load_student(device)

    print("Embedding EuroSAT MSI…")
    T, S, y = embed_dataset(teacher, student, loader, device)

    print("\n=== EuroSAT MSI Benchmark (10 classes) ===")
    linear_probe(T, y, "Teacher")
    linear_probe(S, y, "Student")

if __name__ == "__main__":
    main()

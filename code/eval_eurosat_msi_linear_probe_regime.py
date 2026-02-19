import os, json, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from torchgeo.datasets import EuroSAT
from prithvi_teacher.prithvi_mae import PrithviMAE
from models_student_regime import StudentRegime

TEACHER_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
TEACHER_STD  = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

IMG_SIZE = 64
NUM_FRAMES = 3
BATCH_SIZE = 8
EUROSAT_BANDS = ("B02","B03","B04","B08","B11","B12")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEACHER_DIR = os.path.join(SCRIPT_DIR, "prithvi_distill", "prithvi_teacher")
TEACHER_CKPT = os.path.join(TEACHER_DIR, "Prithvi_EO_V2_300M_TL.pt")
TEACHER_CONFIG = os.path.join(TEACHER_DIR, "config.json")
STUDENT_CKPT = os.path.join(SCRIPT_DIR, "ckpts_regime", "student_regime_ep10.pt")

EUROSAT_ROOT = os.path.join(SCRIPT_DIR, "data_eurosat")

def pick_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
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

def interpolate_pos_embed_2d(pos_embed, new_grid_h, new_grid_w):
    cls = pos_embed[:, :1, :]
    tok = pos_embed[:, 1:, :]
    N = tok.shape[1]
    D = tok.shape[2]
    gs = int(math.sqrt(N))
    if gs * gs != N:
        return pos_embed
    tok = tok.reshape(1, gs, gs, D).permute(0,3,1,2)
    tok = F.interpolate(tok, size=(new_grid_h, new_grid_w), mode="bilinear", align_corners=False)
    tok = tok.permute(0,2,3,1).reshape(1, new_grid_h * new_grid_w, D)
    return torch.cat([cls, tok], dim=1)

def load_teacher(device):
    with open(TEACHER_CONFIG, "r") as f:
        cfg = json.load(f)["pretrained_cfg"]
    cfg = dict(cfg)
    cfg["img_size"] = IMG_SIZE
    cfg["num_frames"] = NUM_FRAMES
    cfg["in_chans"] = 6
    cfg["coords_encoding"] = ["time","location"]
    teacher = PrithviMAE(**cfg).to(device).eval()
    sd = torch.load(TEACHER_CKPT, map_location=device)
    for k in list(sd.keys()):
        if "pos_embed" in k:
            del sd[k]
    teacher.load_state_dict(sd, strict=False)
    return teacher

def load_student(device):
    student = StudentRegime(img_size=IMG_SIZE, num_frames=NUM_FRAMES, K_reg=4).to(device).eval()
    sd = torch.load(STUDENT_CKPT, map_location=device)

    key = "enc.pos_embed"
    if key in sd:
        pe = sd[key]  # (1, 1 + T*oldN, D) e.g. (1,769,256)
        cls = pe[:, :1, :]
        tok = pe[:, 1:, :]  # (1, T*oldN, D)

        T = NUM_FRAMES
        D = tok.shape[-1]
        oldN_total = tok.shape[1]
        assert oldN_total % T == 0, f"pos tokens {oldN_total} not divisible by T={T}"
        oldN = oldN_total // T  # e.g. 768/3=256

        old_grid = int(math.sqrt(oldN))
        assert old_grid * old_grid == oldN, f"oldN={oldN} not square"

        new_grid = IMG_SIZE // 16  # 64->4
        # reshape into (T, old_grid, old_grid)
        tok = tok.reshape(1, T, old_grid, old_grid, D)          # (1,T,og,og,D)
        tok = tok.permute(0,1,4,2,3)                            # (1,T,D,og,og)

        tok_rs = []
        for t in range(T):
            tt = tok[:, t]                                      # (1,D,og,og)
            tt = F.interpolate(tt, size=(new_grid, new_grid), mode="bilinear", align_corners=False)
            tt = tt.permute(0,2,3,1).reshape(1, new_grid*new_grid, D)  # (1,newN,D)
            tok_rs.append(tt)

        tok_new = torch.cat(tok_rs, dim=1)                      # (1, T*newN, D)
        pe_new = torch.cat([cls, tok_new], dim=1)               # (1, 1+T*newN, D)

        sd[key] = pe_new

    student.load_state_dict(sd, strict=False)
    return student


@torch.no_grad()
def embed_dataset(teacher, student, loader, device):
    t_embs, s_embs, ys = [], [], []
    for batch in loader:
        img = batch["image"].to(device).float()
        y = batch["label"].cpu().numpy()

        img = maybe_scale_s2(img)
        x = img.unsqueeze(2).repeat(1,1,NUM_FRAMES,1,1)

        tc = torch.zeros((x.shape[0], NUM_FRAMES, 2), device=device)
        lc = torch.zeros((x.shape[0], 2), device=device)

        x = normalize_pixel_values(x)

        t_feats = teacher.forward_features(x, tc, lc)
        t_cls = t_feats[-1][:,0,:]

        s_cls, _, _ = student.forward_once(x, tc, lc)

        t_embs.append(t_cls.cpu().numpy())
        s_embs.append(s_cls.cpu().numpy())
        ys.append(y)

    return np.concatenate(t_embs), np.concatenate(s_embs), np.concatenate(ys)

def probe(X, y, name):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=4000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"{name} EuroSAT MSI linear-probe accuracy: {acc:.4f}")

def main():
    device = pick_device()
    print("device:", device)
    ds = EuroSAT(root=EUROSAT_ROOT, split="train", bands=EUROSAT_BANDS, download=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=False)

    teacher = load_teacher(device)
    student = load_student(device)

    print("Embedding EuroSAT MSI (FAST)…")
    T, S, y = embed_dataset(teacher, student, loader, device)

    print("\n=== EuroSAT MSI Benchmark (10 classes) ===")
    probe(T, y, "Teacher")
    probe(S, y, "Student")

if __name__ == "__main__":
    main()


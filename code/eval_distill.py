import os, json, glob, datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from prithvi_teacher.prithvi_mae import PrithviMAE, PrithviViT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(SCRIPT_DIR, "hls_chips_all")
TEACHER_DIR = os.path.join(SCRIPT_DIR, "prithvi_distill", "prithvi_teacher")
CKPT_TEACHER = os.path.join(TEACHER_DIR, "Prithvi_EO_V2_300M_TL.pt")
CKPT_STUDENT = os.path.join(SCRIPT_DIR, "ckpts", "student_ep5.pt")

CHIP_SIZE = 256
YEARS = (2019, 2021, 2023)
NUM_FRAMES = 3
SCALE_FACTOR = 10000.0

TEACHER_BANDS = ["B02","B03","B04","B05","B06","B07"]
S30_TO_TEACHER = {"B02":"B02","B03":"B03","B04":"B04","B08":"B05","B11":"B06","B12":"B07"}

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
    return (x - mean) / std

def parse_meta(meta_arr):
    if isinstance(meta_arr, np.ndarray) and meta_arr.shape == ():
        meta_arr = meta_arr.item()
    if isinstance(meta_arr, (bytes, bytearray)):
        meta_arr = meta_arr.decode("utf-8")
    if isinstance(meta_arr, str):
        try: return json.loads(meta_arr)
        except Exception: return {"raw_meta": meta_arr}
    if isinstance(meta_arr, dict):
        return meta_arr
    return {}

def julian_day(meta):
    s = meta.get("item_datetime") or meta.get("datetime") or None
    if s:
        s = str(s).replace("Z","")
        try:
            dt = datetime.datetime.fromisoformat(s.split("+")[0])
            return int(dt.timetuple().tm_yday)
        except Exception:
            pass
    if "julian_day" in meta:
        try: return int(meta["julian_day"])
        except Exception: pass
    return 180

def band_index(bands):
    return {str(b): i for i,b in enumerate(list(bands))}

def fmask_valid_mask(fmask):
    return (fmask == 0)

class TripletDataset(Dataset):
    def __init__(self, root):
        locs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
        self.items = []
        for loc in locs:
            paths = []
            ok = True
            for y in YEARS:
                p = os.path.join(loc, f"{y}.npz")
                if not os.path.exists(p):
                    ok = False
                    break
                paths.append(p)
            if ok:
                self.items.append((os.path.basename(loc), paths))
        if not self.items:
            raise RuntimeError(f"No triplets found under {root}.")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        loc_id, paths = self.items[i]
        frames = []
        temporal = []
        lat = 0.0
        lon = 0.0

        for p in paths:
            z = np.load(p, allow_pickle=True)
            x = z["x"].astype(np.float32)   # (256,256,7)
            bands = z["bands"]
            meta = parse_meta(z["meta"])
            bidx = band_index(bands)

            fmask = x[..., bidx["Fmask"]] if "Fmask" in bidx else None

            idxs = []
            for tb in TEACHER_BANDS:
                if tb in bidx:
                    idxs.append(bidx[tb])
                else:
                    inv = {v:k for k,v in S30_TO_TEACHER.items()}
                    s30 = inv.get(tb, None)
                    if s30 is None or s30 not in bidx:
                        raise RuntimeError(f"Missing band for {tb} in {p}.")
                    idxs.append(bidx[s30])

            x6 = x[..., idxs] * SCALE_FACTOR

            if fmask is not None:
                valid = fmask_valid_mask(fmask)
                x6[~valid] = 0.0

            x6 = x6[:CHIP_SIZE, :CHIP_SIZE, :]
            frames.append(x6)

            year = int(os.path.basename(p).split(".")[0])
            temporal.append([year, julian_day(meta)])
            lat = float(meta.get("lat", lat))
            lon = float(meta.get("lon", lon))

        x = np.stack(frames, axis=0)     # (T,H,W,6)
        x = np.moveaxis(x, -1, 1)        # (T,6,H,W)
        x = torch.from_numpy(x)

        temporal = torch.tensor(temporal, dtype=torch.float32)   # (T,2)
        location = torch.tensor([lat, lon], dtype=torch.float32)

        return {"id": loc_id, "pixel_values": x, "temporal_coords": temporal, "location_coords": location}

def load_teacher(device):
    with open(os.path.join(TEACHER_DIR, "config.json"), "r") as f:
        cfg = json.load(f)["pretrained_cfg"]
    cfg = dict(cfg)
    cfg["num_frames"] = NUM_FRAMES
    cfg["in_chans"] = 6
    cfg["coords_encoding"] = ["time", "location"]
    cfg["img_size"] = CHIP_SIZE

    teacher = PrithviMAE(**cfg).to(device).eval()
    sd = torch.load(CKPT_TEACHER, map_location=device)
    for k in list(sd.keys()):
        if "pos_embed" in k:
            del sd[k]
    teacher.load_state_dict(sd, strict=False)
    return teacher

class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = PrithviViT(
            img_size=CHIP_SIZE,
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
        cls = last[:,0,:]
        pm = last[:,1:,:].mean(dim=1)
        return self.proj(cls), self.proj(pm)

def knn_agreement(teacher_emb, student_emb, k=5):
    # teacher_emb, student_emb: (N,D) torch
    t = F.normalize(teacher_emb, dim=-1)
    s = F.normalize(student_emb, dim=-1)

    t_sim = t @ t.T
    s_sim = s @ s.T

    # exclude self
    eye = torch.eye(t_sim.shape[0], device=t_sim.device).bool()
    t_sim = t_sim.masked_fill(eye, -1e9)
    s_sim = s_sim.masked_fill(eye, -1e9)

    t_knn = torch.topk(t_sim, k=k, dim=1).indices
    s_knn = torch.topk(s_sim, k=k, dim=1).indices

    # overlap per row
    overlaps = []
    for i in range(t_knn.shape[0]):
        overlaps.append(len(set(t_knn[i].tolist()).intersection(set(s_knn[i].tolist()))) / k)
    return float(np.mean(overlaps))

def main():
    device = pick_device()
    print("device:", device)
    print("student ckpt:", CKPT_STUDENT)

    ds = TripletDataset(DATA_ROOT)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, drop_last=False)

    teacher = load_teacher(device)
    student = Student().to(device)
    student.load_state_dict(torch.load(CKPT_STUDENT, map_location=device))
    student.eval()

    ids = []
    t_cls_all, t_pm_all = [], []
    s_cls_all, s_pm_all = [], []
    meta_all = []

    with torch.no_grad():
        for batch in dl:
            ids.extend(batch["id"])
            x = batch["pixel_values"].to(device)        # (B,T,6,H,W)
            x = x.permute(0,2,1,3,4).contiguous()       # (B,6,T,H,W)
            tc = batch["temporal_coords"].to(device)
            lc = batch["location_coords"].to(device)

            x = normalize_pixel_values(x)

            t_feats = teacher.forward_features(x, tc, lc)
            t_last = t_feats[-1]
            t_cls = t_last[:,0,:]
            t_pm = t_last[:,1:,:].mean(dim=1)

            s_cls, s_pm = student(x, tc, lc)

            t_cls_all.append(t_cls.cpu())
            t_pm_all.append(t_pm.cpu())
            s_cls_all.append(s_cls.cpu())
            s_pm_all.append(s_pm.cpu())

            meta_all.append(torch.cat([lc.cpu(), tc[:,0,:].cpu()], dim=1))  # [lat,lon,year0,jday0]

    t_cls_all = torch.cat(t_cls_all, dim=0)
    t_pm_all  = torch.cat(t_pm_all, dim=0)
    s_cls_all = torch.cat(s_cls_all, dim=0)
    s_pm_all  = torch.cat(s_pm_all, dim=0)

    # Cosine similarity stats
    cls_cos = F.cosine_similarity(F.normalize(s_cls_all, dim=-1), F.normalize(t_cls_all, dim=-1), dim=-1)
    pm_cos  = F.cosine_similarity(F.normalize(s_pm_all,  dim=-1), F.normalize(t_pm_all,  dim=-1), dim=-1)

    print("\nTeacher-Student Cosine Similarity")
    print(f"CLS  mean={cls_cos.mean().item():.4f}  std={cls_cos.std().item():.4f}  min={cls_cos.min().item():.4f}")
    print(f"PM   mean={pm_cos.mean().item():.4f}   std={pm_cos.std().item():.4f}   min={pm_cos.min().item():.4f}")

    # KNN agreement (structure preservation)
    knn5 = knn_agreement(t_cls_all, s_cls_all, k=5)
    print(f"\nKNN@5 agreement (teacher vs student, CLS): {knn5:.3f}")

    # Export embeddings for downstream
    out = os.path.join(SCRIPT_DIR, "student_embeddings.npz")
    np.savez(
        out,
        ids=np.array(ids),
        student_cls=s_cls_all.numpy(),
        student_pm=s_pm_all.numpy(),
        teacher_cls=t_cls_all.numpy(),
        teacher_pm=t_pm_all.numpy(),
    )
    print("\nsaved:", out)

if __name__ == "__main__":
    main()

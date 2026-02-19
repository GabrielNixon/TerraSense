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
        except Exception: return {}
    if isinstance(meta_arr, dict): return meta_arr
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
    return 180

def band_index(bands):
    return {str(b): i for i,b in enumerate(list(bands))}

def fmask_valid_mask(fmask):
    return (fmask == 0)

def load_chip_6bands(npz_path):
    z = np.load(npz_path, allow_pickle=True)
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
            idxs.append(bidx[s30])

    x6 = x[..., idxs] * SCALE_FACTOR
    if fmask is not None:
        valid = fmask_valid_mask(fmask)
        x6[~valid] = 0.0

    x6 = x6[:CHIP_SIZE, :CHIP_SIZE, :]      # (H,W,6)
    x6 = np.moveaxis(x6, -1, 0)             # (6,H,W)
    return x6, meta

class SingleYearTripletDataset(Dataset):
    """
    For each (location, year), create a 3-frame input by repeating that year's chip.
    This keeps NUM_FRAMES=3 so the student checkpoint loads cleanly.
    """
    def __init__(self, root):
        locs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
        self.items = []
        for loc in locs:
            loc_id = os.path.basename(loc)
            for y in YEARS:
                p = os.path.join(loc, f"{y}.npz")
                if os.path.exists(p):
                    self.items.append((loc_id, y, p))
        if not self.items:
            raise RuntimeError(f"No .npz found under {root}")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        loc_id, year, p = self.items[i]
        x6, meta = load_chip_6bands(p)  # (6,H,W)

        # Repeat same chip across frames -> (6,T,H,W)
        x = np.repeat(x6[:, None, :, :], repeats=NUM_FRAMES, axis=1)  # (6,3,H,W)
        x = torch.from_numpy(x)

        jd = float(julian_day(meta))
        tc = torch.tensor([[float(year), jd]] * NUM_FRAMES, dtype=torch.float32)  # (3,2)
        lc = torch.tensor([float(meta.get("lat", 0.0)), float(meta.get("lon", 0.0))], dtype=torch.float32)

        return {"id": loc_id, "year": int(year), "pixel_values": x, "temporal_coords": tc, "location_coords": lc}

def load_teacher(device):
    with open(os.path.join(TEACHER_DIR, "config.json"), "r") as f:
        cfg = json.load(f)["pretrained_cfg"]
    cfg = dict(cfg)
    cfg["num_frames"] = NUM_FRAMES
    cfg["in_chans"] = 6
    cfg["coords_encoding"] = ["time","location"]
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

def main():
    device = pick_device()
    print("device:", device)

    ds = SingleYearTripletDataset(DATA_ROOT)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, drop_last=False)

    teacher = load_teacher(device)
    student = Student().to(device)
    student.load_state_dict(torch.load(CKPT_STUDENT, map_location=device))
    student.eval()

    ids, years = [], []
    t_cls_all, s_cls_all = [], []

    with torch.no_grad():
        for b in dl:
            ids.extend(b["id"])
            years.extend([int(y) for y in b["year"]])

            x = b["pixel_values"].to(device)          # (B,6,3,H,W)
            tc = b["temporal_coords"].to(device)      # (B,3,2)
            lc = b["location_coords"].to(device)      # (B,2)

            x = normalize_pixel_values(x)

            t_feats = teacher.forward_features(x, tc, lc)
            t_last = t_feats[-1]
            t_cls = t_last[:,0,:]

            s_cls, _ = student(x, tc, lc)

            t_cls_all.append(t_cls.cpu())
            s_cls_all.append(s_cls.cpu())

    t_cls_all = torch.cat(t_cls_all, dim=0).numpy()
    s_cls_all = torch.cat(s_cls_all, dim=0).numpy()

    out = os.path.join(SCRIPT_DIR, "year_embeddings.npz")
    np.savez(out, ids=np.array(ids), years=np.array(years), teacher_cls=t_cls_all, student_cls=s_cls_all)
    print("saved:", out)

if __name__ == "__main__":
    main()

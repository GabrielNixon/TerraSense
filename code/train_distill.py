import os, glob, json, datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from prithvi_teacher.prithvi_mae import PrithviMAE, PrithviViT

# -----------------------
# Paths (EDIT if needed)
# -----------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(SCRIPT_DIR, "hls_chips_all")
TEACHER_DIR = os.path.join(SCRIPT_DIR, "prithvi_distill", "prithvi_teacher")
CKPT_PATH = os.path.join(TEACHER_DIR, "Prithvi_EO_V2_300M_TL.pt")

# Your chips: reflectance-like ~0..1 => teacher expects ~1000..3000
SCALE_FACTOR = 10000.0

# We will use full 256x256 chips (no crop)
CHIP_SIZE = 256

# years you have per location folder
YEARS = (2019, 2021, 2023)
NUM_FRAMES = 3

# Teacher uses 6 bands in this order
TEACHER_BANDS = ["B02","B03","B04","B05","B06","B07"]
# Your bands are HLS S30 style
S30_TO_TEACHER = {"B02":"B02","B03":"B03","B04":"B04","B08":"B05","B11":"B06","B12":"B07"}

# Teacher normalization stats (from Prithvi config.json)
TEACHER_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
TEACHER_STD  = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

# -----------------------
# Device
# -----------------------
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# -----------------------
# Utils
# -----------------------
def normalize_pixel_values(x_bcthw):
    mean = torch.tensor(TEACHER_MEAN, device=x_bcthw.device).view(1,6,1,1,1)
    std  = torch.tensor(TEACHER_STD,  device=x_bcthw.device).view(1,6,1,1,1)
    return (x_bcthw - mean) / std

def parse_meta(meta_arr):
    # meta is np.ndarray shape () with dtype <U... (a JSON string)
    if isinstance(meta_arr, np.ndarray) and meta_arr.shape == ():
        meta_arr = meta_arr.item()
    if isinstance(meta_arr, (bytes, bytearray)):
        meta_arr = meta_arr.decode("utf-8")
    if isinstance(meta_arr, str):
        try:
            return json.loads(meta_arr)
        except Exception:
            return {"raw_meta": meta_arr}
    if isinstance(meta_arr, dict):
        return meta_arr
    return {}

def julian_day_from_meta(meta):
    s = meta.get("item_datetime") or meta.get("datetime") or None
    if s:
        s = str(s).replace("Z","")
        try:
            dt = datetime.datetime.fromisoformat(s.split("+")[0])
            return int(dt.timetuple().tm_yday)
        except Exception:
            pass
    if "julian_day" in meta:
        try:
            return int(meta["julian_day"])
        except Exception:
            pass
    return 180

def band_index(bands):
    return {str(b): i for i, b in enumerate(list(bands))}

def fmask_valid_mask(fmask):
    """
    We don't know your exact bit-codes, but we do know:
    - clear pixels are usually coded as 0 in many HLS pipelines
    - your mean is 144, so you have lots of non-zero values too

    Safe default:
    - keep pixels where fmask == 0 (clear)
    - if that yields too few pixels in practice, we can loosen it later

    Returns boolean mask (H,W): True = valid.
    """
    return (fmask == 0)

# -----------------------
# Dataset
# -----------------------
class TripletDataset(Dataset):
    """
    Expects each location folder to contain:
      location_xx/2019.npz
      location_xx/2021.npz
      location_xx/2023.npz
    """
    def __init__(self, root, years=YEARS, chip=CHIP_SIZE, scale=SCALE_FACTOR):
        self.root = root
        self.years = list(years)
        self.chip = int(chip)
        self.scale = float(scale)

        locs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
        items = []
        for loc in locs:
            paths = []
            ok = True
            for y in self.years:
                p = os.path.join(loc, f"{y}.npz")
                if not os.path.exists(p):
                    ok = False
                    break
                paths.append(p)
            if ok:
                items.append((loc, paths))

        if not items:
            raise RuntimeError(f"No triplets found under {root}. "
                               f"Expected location folders with {years} .npz files.")
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        loc, paths = self.items[i]
        frames = []
        temporal = []
        lat = 0.0
        lon = 0.0

        for p in paths:
            z = np.load(p, allow_pickle=True)
            x = z["x"].astype(np.float32)            # (H,W,7)
            bands = z["bands"]                       # (7,)
            meta = parse_meta(z["meta"])

            bidx = band_index(bands)

            # separate fmask
            fmask = x[..., bidx["Fmask"]] if "Fmask" in bidx else None

            # select 6 spectral bands mapped to teacher order
            idxs = []
            for tb in TEACHER_BANDS:
                if tb in bidx:
                    idxs.append(bidx[tb])
                else:
                    inv = {v:k for k,v in S30_TO_TEACHER.items()}
                    s30 = inv.get(tb, None)
                    if s30 is None or s30 not in bidx:
                        raise RuntimeError(f"Missing band for {tb} in {p}. bands={list(bands)}")
                    idxs.append(bidx[s30])

            x6 = x[..., idxs] * self.scale           # (H,W,6) -> scaled

            # apply fmask: zero-out invalid pixels
            if fmask is not None:
                valid = fmask_valid_mask(fmask)
                x6[~valid] = 0.0

            # ensure chip size
            x6 = x6[:self.chip, :self.chip, :]       # (256,256,6)

            year = int(os.path.basename(p).split(".")[0])
            jday = julian_day_from_meta(meta)

            lat = float(meta.get("lat", lat))
            lon = float(meta.get("lon", lon))

            frames.append(x6)
            temporal.append([year, jday])

        # stack -> (T,H,W,6) then to torch as (T,6,H,W)
        x = np.stack(frames, axis=0)
        x = np.moveaxis(x, -1, 1)
        x = torch.from_numpy(x)                      # (T,6,H,W)

        temporal = torch.tensor(temporal, dtype=torch.float32)   # (T,2)
        location = torch.tensor([lat, lon], dtype=torch.float32) # (2,)

        return {"pixel_values": x, "temporal_coords": temporal, "location_coords": location}

# -----------------------
# Teacher + Student
# -----------------------
def load_teacher(device):
    with open(os.path.join(TEACHER_DIR, "config.json"), "r") as f:
        cfg = json.load(f)["pretrained_cfg"]

    cfg = dict(cfg)
    cfg["num_frames"] = NUM_FRAMES
    cfg["in_chans"] = 6
    cfg["coords_encoding"] = ["time", "location"]
    cfg["img_size"] = CHIP_SIZE

    teacher = PrithviMAE(**cfg).to(device).eval()
    sd = torch.load(CKPT_PATH, map_location=device)

    # match official inference: remove pos_embed keys
    for k in list(sd.keys()):
        if "pos_embed" in k:
            del sd[k]
    teacher.load_state_dict(sd, strict=False)
    return teacher

class Student(torch.nn.Module):
    def __init__(self, device):
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

    def forward(self, x_bcthw, tc, lc):
        feats = self.enc.forward_features(x_bcthw, tc, lc)
        last = feats[-1]
        cls = last[:, 0, :]
        pm = last[:, 1:, :].mean(dim=1)
        return self.proj(cls), self.proj(pm)

def distill_loss(s_cls, s_pm, t_cls, t_pm):
    s1 = F.normalize(s_cls, dim=-1); t1 = F.normalize(t_cls, dim=-1)
    s2 = F.normalize(s_pm, dim=-1);  t2 = F.normalize(t_pm, dim=-1)
    l1 = 1.0 - (s1 * t1).sum(dim=-1).mean()
    l2 = 1.0 - (s2 * t2).sum(dim=-1).mean()
    return 0.7*l1 + 0.3*l2

# -----------------------
# Train
# -----------------------
def main():
    device = pick_device()
    print("device:", device)
    print("data:", DATA_ROOT)
    print("teacher ckpt:", CKPT_PATH)

    ds = TripletDataset(DATA_ROOT)
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=False, drop_last=True)

    teacher = load_teacher(device)
    student = Student(device).to(device).train()

    opt = torch.optim.AdamW(student.parameters(), lr=2e-4, weight_decay=0.05)

    os.makedirs("ckpts", exist_ok=True)

    for ep in range(1, 6):
        running = 0.0
        n = 0
        for batch in dl:
            x = batch["pixel_values"].to(device)          # (B,T,6,H,W)
            x = x.permute(0,2,1,3,4).contiguous()         # -> (B,6,T,H,W)
            tc = batch["temporal_coords"].to(device)      # (B,T,2)
            lc = batch["location_coords"].to(device)      # (B,2)

            x = normalize_pixel_values(x)

            with torch.no_grad():
                t_feats = teacher.forward_features(x, tc, lc)
                t_last = t_feats[-1]
                t_cls = t_last[:, 0, :]
                t_pm  = t_last[:, 1:, :].mean(dim=1)

            s_cls, s_pm = student(x, tc, lc)
            loss = distill_loss(s_cls, s_pm, t_cls, t_pm)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            n += 1

        ck = f"ckpts/student_ep{ep}.pt"
        torch.save(student.state_dict(), ck)
        print(f"epoch {ep} | loss {running/max(n,1):.4f} | saved {ck}")

if __name__ == "__main__":
    main()

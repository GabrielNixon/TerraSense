import os
import glob
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "hls_chips_all")
EMB_FILE = os.path.join(SCRIPT_DIR, "student_embeddings.npz")

D = np.load(EMB_FILE, allow_pickle=True)
ids = D["ids"]
teacher = D["teacher_cls"]
student = D["student_cls"]

# --- compute NDVI per location per year ---
def compute_ndvi(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    x = z["x"].astype(np.float32)
    bands = z["bands"]
    band_map = {str(b): i for i, b in enumerate(bands)}

    red = x[..., band_map["B04"]]
    nir = x[..., band_map["B08"]]
    ndvi = (nir - red) / (nir + red + 1e-6)

    # Mask clouds if available
    if "Fmask" in band_map:
        fmask = x[..., band_map["Fmask"]]
        valid = (fmask == 0)
        if np.any(valid):
            ndvi_valid = ndvi[valid]
            m = float(np.mean(ndvi_valid))
            if np.isfinite(m):
                return m

    # Fallback: unmasked mean
    m = float(np.mean(ndvi))
    if np.isfinite(m):
        return m

    return np.nan


# Build NDVI change target (2019 → 2023)
targets = []
valid_ids = []

for loc in sorted(os.listdir(DATA_ROOT)):
    loc_dir = os.path.join(DATA_ROOT, loc)
    if not os.path.isdir(loc_dir):
        continue

    p2019 = os.path.join(loc_dir, "2019.npz")
    p2023 = os.path.join(loc_dir, "2023.npz")
    if not (os.path.exists(p2019) and os.path.exists(p2023)):
        continue

    ndvi_2019 = compute_ndvi(p2019)
    ndvi_2023 = compute_ndvi(p2023)

    y = ndvi_2023 - ndvi_2019
    if np.isfinite(y):
        targets.append(y)
        valid_ids.append(loc)

targets = np.array(targets, dtype=np.float32)
print("usable samples:", len(targets))

# Align embeddings (triplet embeddings)
mask = np.array([i in valid_ids for i in ids])
teacher_X = teacher[mask]
student_X = student[mask]
y = targets

print("samples:", len(y))

def evaluate(X, name):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(Xtr, ytr)
    preds = model.predict(Xte)
    r2 = r2_score(yte, preds)
    print(f"{name} R²: {r2:.4f}")

print("\nVegetation Change Regression (NDVI Δ 2019→2023)")
evaluate(teacher_X, "Teacher")
evaluate(student_X, "Student")

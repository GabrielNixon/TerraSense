import os, glob, numpy as np

ROOT = os.path.expanduser("~/Desktop/hls_project/hls_chips_all")

cands = glob.glob(os.path.join(ROOT, "**", "*.npz"), recursive=True)
print("found npz:", len(cands))
print("first 5:", cands[:5])

p = sorted(cands)[0]
z = np.load(p, allow_pickle=True)
print("\nSAMPLE:", p)
print("keys:", z.files)
for k in z.files:
    v = z[k]
    print(k, type(v), getattr(v, "shape", None), getattr(v, "dtype", None))

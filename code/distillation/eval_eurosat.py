import os
import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

# load EuroSAT (RGB)
ds = load_dataset("eurosat/rgb", split="train")

def preprocess(sample):
    # sample["image"] is PIL.Image
    img = np.array(sample["image"])
    img = img / 255.0
    # resize to 256x256 if needed
    import cv2
    img = cv2.resize(img, (256,256))
    # reorder to (C,H,W)
    img = np.transpose(img, (2,0,1))
    return img

X, y = [], []
for i in range(len(ds)):
    sample = ds[i]
    img = preprocess(sample)
    X.append(img)
    y.append(sample["label"])

X = np.stack(X)
y = np.array(y)

# batch embed via your student encoder
import torch.nn.functional as F
from prithvi_teacher.prithvi_mae import PrithviViT

student = ... # load your student model
student.eval()

def embed_batch(imgs):
    with torch.no_grad():
        imgs = torch.tensor(imgs).float().to(device)
        # your student expects shape (B,6,T,H,W)
        # for EuroSAT we can pad to 3 identical frames
        imgs = imgs.unsqueeze(2).repeat(1,1,3,1,1)
        # dummy coords
        tc = torch.zeros(imgs.shape[0], 3, 2).to(device)
        lc = torch.zeros(imgs.shape[0], 2).to(device)

        _, emb = student(imgs, tc, lc)
        return emb.cpu().numpy()

# get embeddings
embs = embed_batch(X)

# train/test split
from sklearn.model_selection import train_test_split
Xe_tr, Xe_te, y_tr, y_te = train_test_split(embs, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=2000)
clf.fit(Xe_tr, y_tr)

preds = clf.predict(Xe_te)
acc = accuracy_score(y_te, preds)
print("EuroSAT accuracy:", acc)

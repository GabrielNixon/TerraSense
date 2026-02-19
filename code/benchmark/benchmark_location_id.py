import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data = np.load(os.path.join(SCRIPT_DIR, "year_embeddings.npz"), allow_pickle=True)

ids = data["ids"]
teacher = data["teacher_cls"]
student = data["student_cls"]

labels = ids  # each location has 3 samples (2019/2021/2023)
uniq = sorted(set(labels))
m = {u:i for i,u in enumerate(uniq)}
y = np.array([m[l] for l in labels])

def eval(X, name):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=5000)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    print(f"{name} accuracy: {accuracy_score(yte, pred):.4f}")

print("\nLocation-ID Classification Benchmark (3 samples/class)")
eval(teacher, "Teacher")
eval(student, "Student")

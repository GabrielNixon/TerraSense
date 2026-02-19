import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_FILE = os.path.join(SCRIPT_DIR, "student_embeddings.npz")

data = np.load(EMB_FILE, allow_pickle=True)

ids = data["ids"]
teacher = data["teacher_cls"]
student = data["student_cls"]

# Extract country label from folder name
# Example: "accra_01" → "accra"
labels = np.array([i.split("_")[0] for i in ids])

# Encode labels as integers
unique = sorted(set(labels))
label_map = {u:i for i,u in enumerate(unique)}
y = np.array([label_map[l] for l in labels])

def evaluate(X, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} accuracy: {acc:.4f}")

print("\nCountry Classification Benchmark")
evaluate(teacher, "Teacher")
evaluate(student, "Student")

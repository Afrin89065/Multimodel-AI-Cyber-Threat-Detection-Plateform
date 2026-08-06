import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from backend.services.nlp_services import extract_url_features

# -----------------------------
# Change this path
# -----------------------------
DATASET = "datasets/raw/email/CEAS_08.csv"

df = pd.read_csv(DATASET)

# -----------------------------
# Auto-detect URL column
# -----------------------------
url_col = None

for c in df.columns:
    if "url" in c.lower():
        url_col = c
        break

if url_col is None:
    raise Exception("No URL column found.")

# -----------------------------
# Auto-detect label column
# -----------------------------
label_col = None

for c in df.columns:
    if "label" in c.lower():
        label_col = c
        break

if label_col is None:
    raise Exception("No label column found.")

X = []
y = []

for _, row in df.iterrows():
    X.append(extract_url_features(str(row[url_col])))
    y.append(int(row[label_col]))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

save_dir = Path("models_store/nlp")
save_dir.mkdir(parents=True, exist_ok=True)

with open(save_dir / "nlp_lite.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nSaved:")
print(save_dir / "nlp_lite.pkl")
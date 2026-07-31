import json
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from loguru import logger

# -------------------------------
# Configuration
# -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_FILE = PROJECT_ROOT / "datasets/processed/fusion/test.jsonl"
MODEL_FILE = PROJECT_ROOT / "backend/models/fusion_model.pt"

THREAT_CLASSES = [
    "CLEAN",
    "PHISHING",
    "BEC",
    "MALWARE",
    "NETWORK_ATTACK"
]

# -------------------------------
# Same model as backend
# -------------------------------

import torch.nn as nn

class AttentionFusionEngine(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Linear(3,64)

        self.attention = nn.MultiheadAttention(
            64,
            num_heads=4,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(64*4,128),
            nn.ReLU(),
            nn.Linear(128,5)
        )

    def forward(self,x):

        emb=self.embedding(x)

        attn,_=self.attention(emb,emb,emb)

        out=self.classifier(attn.reshape(attn.size(0),-1))

        return out


# -------------------------------
# Load Dataset
# -------------------------------

logger.info("Loading fusion test dataset...")

X=[]
Y=[]

with open(DATA_FILE) as f:

    for line in f:

        sample=json.loads(line)

        X.append(sample["features"])
        Y.append(sample["label"])

X=np.array(X,dtype=np.float32)
Y=np.array(Y)

# convert 12 -> (4,3)

X=X.reshape(-1,4,3)

# -------------------------------
# Load model
# -------------------------------

model=AttentionFusionEngine()

model.load_state_dict(
    torch.load(MODEL_FILE,map_location="cpu")
)

model.eval()

# -------------------------------
# Prediction
# -------------------------------

with torch.no_grad():

    logits=model(torch.tensor(X))

    preds=torch.argmax(logits,dim=1).numpy()

# -------------------------------
# Metrics
# -------------------------------

acc=accuracy_score(Y,preds)

logger.info(f"Accuracy : {acc:.4f}")

print()

print(classification_report(
    Y,
    preds,
    target_names=THREAT_CLASSES,
    zero_division=0
))

print()

print("Confusion Matrix")

print(confusion_matrix(Y,preds))

print()

logger.success("Fusion evaluation complete.")
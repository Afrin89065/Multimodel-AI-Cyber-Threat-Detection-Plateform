"""
=========================================================
AIDTECT v3.0
Attention Fusion Training
=========================================================
Fuses predictions from

1. NLP
2. Vision
3. Malware
4. Network

Output:
backend/models/fusion_model.pt
=========================================================
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from loguru import logger


##############################################################
# PROJECT PATHS
##############################################################

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets/processed/fusion"

TRAIN_FILE = DATASET_DIR / "train.jsonl"
VAL_FILE = DATASET_DIR / "val.jsonl"
TEST_FILE = DATASET_DIR / "test.jsonl"

MODEL_DIR = PROJECT_ROOT / "backend/models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "fusion_model.pt"

PLOT_DIR = PROJECT_ROOT / "backend/models"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

##############################################################
# SETTINGS
##############################################################

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_CLASSES = 5

FEATURE_DIM = 3

MODULES = 4

HIDDEN_DIM = 64

BATCH_SIZE = 64

EPOCHS = 40

LR = 1e-3

WEIGHT_DECAY = 1e-5

EARLY_STOPPING = 8

logger.info(f"Device : {DEVICE}")

##############################################################
# LABELS
##############################################################

CLASS_NAMES = [
    "CLEAN",
    "PHISHING",
    "BEC",
    "MALWARE",
    "NETWORK_ATTACK",
]
##############################################################
# DATASET
##############################################################

class FusionDataset(Dataset):

    def __init__(self, file_path):

        self.features = []
        self.labels = []

        logger.info(f"Loading {file_path}")

        with open(file_path, "r") as f:

            for line in f:

                sample = json.loads(line)

                feature = np.array(
                    sample["features"],
                    dtype=np.float32
                ).reshape(4, 3)

                label = sample["label"]

                self.features.append(feature)
                self.labels.append(label)

        self.features = np.asarray(self.features)
        self.labels = np.asarray(self.labels)

        logger.success(
            f"Loaded {len(self.labels)} samples"
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return (

            torch.tensor(
                self.features[idx],
                dtype=torch.float32,
            ),

            torch.tensor(
                self.labels[idx],
                dtype=torch.long,
            ),
        )
    ##############################################################
# ATTENTION FUSION MODEL
##############################################################

class AttentionFusionEngine(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Linear(
            FEATURE_DIM,
            HIDDEN_DIM,
        )

        self.attention = nn.MultiheadAttention(

            embed_dim=HIDDEN_DIM,

            num_heads=4,

            batch_first=True,

            dropout=0.2,

        )

        self.norm = nn.LayerNorm(HIDDEN_DIM)

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(

            nn.Linear(
                HIDDEN_DIM * MODULES,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                NUM_CLASSES,
            ),
        )

    def forward(self, x):

        x = self.embedding(x)

        attn_output, attention = self.attention(
            x,
            x,
            x,
        )

        x = self.norm(x + attn_output)

        x = self.dropout(x)

        x = x.reshape(
            x.size(0),
            -1,
        )

        logits = self.classifier(x)

        return logits, attention
    ##############################################################
# LOAD DATASETS
##############################################################

train_dataset = FusionDataset(TRAIN_FILE)
val_dataset = FusionDataset(VAL_FILE)
test_dataset = FusionDataset(TEST_FILE)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

##############################################################
# BUILD MODEL
##############################################################

model = AttentionFusionEngine().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

best_accuracy = 0.0
early_counter = 0

train_losses = []
val_losses = []

train_accs = []
val_accs = []
##############################################################
# TRAINING
##############################################################

def train_one_epoch():

    model.train()

    running_loss = 0

    predictions = []
    labels_all = []

    for features, labels in train_loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(features)

        loss = criterion(logits, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        predictions.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = accuracy_score(
        labels_all,
        predictions
    )

    return epoch_loss, epoch_acc
##############################################################
# VALIDATION
##############################################################

@torch.no_grad()
def validate(loader):

    model.eval()

    running_loss = 0

    predictions = []
    labels_all = []

    attention_weights = []

    for features, labels in loader:

        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        logits, attention = model(features)

        loss = criterion(
            logits,
            labels
        )

        running_loss += loss.item()

        preds = torch.argmax(
            logits,
            dim=1
        )

        predictions.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        attention_weights.append(
            attention.cpu().numpy()
        )

    epoch_loss = running_loss / len(loader)

    epoch_acc = accuracy_score(
        labels_all,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_all,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "loss": epoch_loss,
        "accuracy": epoch_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": labels_all,
        "predictions": predictions,
        "attention": attention_weights,
    }
##############################################################
# SAVE MODEL
##############################################################

def save_checkpoint():

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    logger.success(
        f"Model saved -> {MODEL_PATH}"
    )
    ##############################################################
# TRAINING LOOP
##############################################################

logger.info("=" * 70)
logger.info("Starting Fusion Training")
logger.info("=" * 70)

for epoch in range(EPOCHS):

    ##########################################################
    # Train
    ##########################################################

    train_loss, train_acc = train_one_epoch()

    ##########################################################
    # Validation
    ##########################################################

    val_result = validate(val_loader)

    scheduler.step(val_result["accuracy"])

    train_losses.append(train_loss)
    val_losses.append(val_result["loss"])

    train_accs.append(train_acc)
    val_accs.append(val_result["accuracy"])

    logger.info("")
    logger.info(f"Epoch {epoch+1}/{EPOCHS}")
    logger.info("-" * 40)
    logger.info(f"Train Loss : {train_loss:.4f}")
    logger.info(f"Train Acc  : {train_acc:.4f}")
    logger.info(f"Val Loss   : {val_result['loss']:.4f}")
    logger.info(f"Val Acc    : {val_result['accuracy']:.4f}")
    logger.info(f"Precision  : {val_result['precision']:.4f}")
    logger.info(f"Recall     : {val_result['recall']:.4f}")
    logger.info(f"F1 Score   : {val_result['f1']:.4f}")

    ##########################################################
    # Save Best Model
    ##########################################################

    if val_result["accuracy"] > best_accuracy:

        best_accuracy = val_result["accuracy"]

        early_counter = 0

        save_checkpoint()

        logger.success(
            f"Best Validation Accuracy : {best_accuracy:.4f}"
        )

    else:

        early_counter += 1

        logger.info(
            f"Early Stopping Counter : {early_counter}/{EARLY_STOPPING}"
        )

    ##########################################################
    # Early Stopping
    ##########################################################

    if early_counter >= EARLY_STOPPING:

        logger.success(
            "Early stopping activated."
        )

        break

logger.info("=" * 70)
logger.success(
    f"Training Finished | Best Accuracy = {best_accuracy:.4f}"
)
logger.info("=" * 70)
##############################################################
# LOAD BEST MODEL
##############################################################

logger.info("Loading Best Fusion Model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

##############################################################
# TEST EVALUATION
##############################################################

logger.info("=" * 70)
logger.info("Testing Fusion Model")
logger.info("=" * 70)

test_result = validate(test_loader)

print("\n")
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(f"Test Loss      : {test_result['loss']:.4f}")
print(f"Test Accuracy  : {test_result['accuracy']:.4f}")
print(f"Precision      : {test_result['precision']:.4f}")
print(f"Recall         : {test_result['recall']:.4f}")
print(f"F1 Score       : {test_result['f1']:.4f}")

##############################################################
# CLASSIFICATION REPORT
##############################################################

print("\nClassification Report\n")

print(
    classification_report(
        test_result["labels"],
        test_result["predictions"],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

##############################################################
# CONFUSION MATRIX
##############################################################

cm = confusion_matrix(
    test_result["labels"],
    test_result["predictions"]
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(8,8))

disp.plot(
    cmap="Blues",
    ax=ax,
    colorbar=False
)

plt.title("Fusion Model Confusion Matrix")

plt.tight_layout()

cm_path = MODEL_DIR / "fusion_confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300
)

plt.close()

logger.success(
    f"Confusion Matrix Saved -> {cm_path}"
)
##############################################################
# TRAINING CURVES
##############################################################

logger.info("Saving training graphs...")

##############################################################
# LOSS CURVE
##############################################################

plt.figure(figsize=(8,5))

plt.plot(
    train_losses,
    label="Train Loss",
    linewidth=2
)

plt.plot(
    val_losses,
    label="Validation Loss",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Fusion Training Loss")

plt.legend()

plt.grid(True)

loss_path = MODEL_DIR / "fusion_loss_curve.png"

plt.tight_layout()

plt.savefig(
    loss_path,
    dpi=300
)

plt.close()

##############################################################
# ACCURACY CURVE
##############################################################

plt.figure(figsize=(8,5))

plt.plot(
    train_accs,
    label="Train Accuracy",
    linewidth=2
)

plt.plot(
    val_accs,
    label="Validation Accuracy",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Fusion Training Accuracy")

plt.legend()

plt.grid(True)

acc_path = MODEL_DIR / "fusion_accuracy_curve.png"

plt.tight_layout()

plt.savefig(
    acc_path,
    dpi=300
)

plt.close()

logger.success(f"Loss Curve Saved      : {loss_path}")
logger.success(f"Accuracy Curve Saved  : {acc_path}")

##############################################################
# SAVE TRAINING METRICS
##############################################################

metrics = {
    "best_accuracy": float(best_accuracy),
    "epochs_completed": len(train_losses),
    "train_loss": [float(x) for x in train_losses],
    "val_loss": [float(x) for x in val_losses],
    "train_accuracy": [float(x) for x in train_accs],
    "val_accuracy": [float(x) for x in val_accs]
}

metrics_file = MODEL_DIR / "fusion_metrics.json"

with open(metrics_file, "w") as f:
    json.dump(metrics, f, indent=4)

logger.success(f"Training Metrics Saved : {metrics_file}")

##############################################################
# FINISHED
##############################################################

print("\n" + "=" * 70)
print("✅ Fusion Model Training Completed Successfully")
print("=" * 70)

print(f"Model                : {MODEL_PATH}")
print(f"Metrics              : {metrics_file}")
print(f"Confusion Matrix     : {cm_path}")
print(f"Loss Curve           : {loss_path}")
print(f"Accuracy Curve       : {acc_path}")
print(f"Best Validation Acc  : {best_accuracy:.4f}")
print("=" * 70)
"""Generate synthetic fusion training data. RUN: python scripts\training\generate_fusion_data.py"""
import json, sys, numpy as np, random
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import os; os.chdir(PROJECT_ROOT)

OUT = PROJECT_ROOT / "datasets/processed/fusion"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["CLEAN", "PHISHING", "BEC", "MALWARE", "NETWORK_ATTACK"]
rng = np.random.RandomState(42)

def gen_sample(label_idx):
    label = LABELS[label_idx]
    nlp_s = rng.uniform(0.7, 0.98) if label in ("PHISHING", "BEC") else rng.uniform(0, 0.25)
    vis_s = rng.uniform(0.6, 0.95) if label == "PHISHING" else rng.uniform(0, 0.2)
    net_s = rng.uniform(0.7, 0.98) if label == "NETWORK_ATTACK" else rng.uniform(0, 0.2)
    mal_s = rng.uniform(0.7, 0.98) if label == "MALWARE" else rng.uniform(0, 0.15)
    return {
        "features": [
            float(nlp_s), float(vis_s), float(net_s), float(mal_s),
            float(rng.uniform(0.7, 0.99)), float(rng.uniform(0.5, 0.99)),
            float(rng.uniform(0.7, 0.99)), float(rng.uniform(0.7, 0.99)),
            1.0 if label != "CLEAN" and nlp_s > 0.5 else 0.0,
            1.0 if label == "PHISHING" and vis_s > 0.5 else 0.0,
            1.0 if label == "NETWORK_ATTACK" and net_s > 0.5 else 0.0,
            1.0 if label == "MALWARE" and mal_s > 0.5 else 0.0,
        ],
        "label": label_idx,
    }

for split, n in [("train", 20000), ("val", 4000), ("test", 4000)]:
    samples = []
    per_class = n // len(LABELS)
    for label_idx in range(len(LABELS)):
        for _ in range(per_class):
            samples.append(gen_sample(label_idx))
    random.shuffle(samples)
    with open(OUT / f"{split}.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    logger.info(f"{split}: {len(samples)} fusion samples")

logger.info("Fusion data generation complete!")
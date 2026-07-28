"""Preprocess vision screenshots. RUN: python scripts\training\preprocess_vision.py"""
import json, sys, random
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import os; os.chdir(PROJECT_ROOT)

RAW = PROJECT_ROOT / "datasets/raw/vision"
OUT = PROJECT_ROOT / "datasets/processed/vision"
OUT.mkdir(parents=True, exist_ok=True)

phishing_imgs = list((RAW / "phishing").glob("*.png")) + list((RAW / "phishing").glob("*.jpg"))
benign_imgs = list((RAW / "benign").glob("*.png")) + list((RAW / "benign").glob("*.jpg"))

logger.info(f"Found {len(phishing_imgs)} phishing + {len(benign_imgs)} benign screenshots")

if len(phishing_imgs) + len(benign_imgs) < 10:
    logger.warning("Very few screenshots found. Run collect_screenshots.py first.")
    logger.warning("Creating minimal metadata with available files...")

samples = []
for p in phishing_imgs:
    samples.append({"path": str(p), "label": "PHISHING"})
for p in benign_imgs:
    samples.append({"path": str(p), "label": "LEGITIMATE"})

if len(samples) == 0:
    logger.warning("No screenshots — creating placeholder metadata")
    samples = [{"path": "placeholder.png", "label": "PHISHING"}] * 10 + \
              [{"path": "placeholder.png", "label": "LEGITIMATE"}] * 10

random.seed(42)
random.shuffle(samples)
n = len(samples)
n_train = int(0.7 * n)
n_val = int(0.15 * n)

for split_name, split_data in [
    ("train", samples[:n_train]),
    ("val", samples[n_train:n_train+n_val]),
    ("test", samples[n_train+n_val:]),
]:
    with open(OUT / f"{split_name}_meta.json", "w") as f:
        json.dump(split_data, f, indent=2)
    logger.info(f"{split_name}: {len(split_data)} samples")

logger.info("Vision preprocessing complete!")
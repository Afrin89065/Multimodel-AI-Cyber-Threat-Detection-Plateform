"""Preprocess NLP dataset. RUN: python scripts\training\preprocess_nlp.py"""
import json, re, csv, bz2, sys, random
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import os; os.chdir(PROJECT_ROOT)

RAW = PROJECT_ROOT / "datasets/raw/nlp"
OUT = PROJECT_ROOT / "datasets/processed/nlp"
OUT.mkdir(parents=True, exist_ok=True)

samples = []

# PhishTank
phishtank = RAW / "phishtank_10k.csv"
if phishtank.exists():
    with open(phishtank, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("url", row.get("phish_url", ""))
            if url:
                samples.append({"text": f"Suspicious link detected: {url}", "url": url, "label": "PHISHING"})
    logger.info(f"PhishTank: {len(samples)} samples")

# OpenPhish
openphish = RAW / "openphish.txt"
if openphish.exists():
    urls = [l.strip() for l in openphish.read_text().splitlines() if l.strip()]
    for url in urls[:5000]:
        samples.append({"text": f"Suspicious URL: {url}", "url": url, "label": "PHISHING"})
    logger.info(f"OpenPhish: {len(urls)} URLs added")

# ENRON emails (benign)
enron_dir = RAW / "enron"
if enron_dir.exists():
    count = 0
    for fp in list(enron_dir.rglob("*.txt"))[:10000]:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")[:500]
            if len(text) > 50:
                samples.append({"text": text, "url": "", "label": "CLEAN"})
                count += 1
        except Exception:
            pass
    logger.info(f"ENRON: {count} emails added")

# Fallback synthetic data if nothing downloaded yet
if len(samples) < 100:
    logger.warning("No real data found — creating minimal synthetic dataset for testing")
    phishing_texts = [
        ("URGENT: Your PayPal account has been suspended! Verify now.", "paypa1-secure.xyz/verify"),
        ("Your bank account needs immediate attention. Click here.", "secure-bank-login.tk/account"),
        ("Congratulations! You won $1000. Claim your prize now.", "prize-claim.xyz/winner"),
        ("Your Microsoft account password expired. Update now.", "microsoftupdate.cc/password"),
        ("ALERT: Unusual activity on your Amazon account.", "amazon-secure.ml/verify"),
    ] * 200
    clean_texts = [
        ("Please find attached the quarterly report for your review.", ""),
        ("The meeting has been scheduled for 3pm on Friday.", ""),
        ("Thank you for your order. It will arrive in 3-5 days.", ""),
        ("Here is the project update you requested.", ""),
        ("Happy to connect and discuss the proposal further.", ""),
    ] * 200
    for text, url in phishing_texts:
        samples.append({"text": text, "url": url, "label": "PHISHING"})
    for text, url in clean_texts:
        samples.append({"text": text, "url": url, "label": "CLEAN"})

random.seed(42)
random.shuffle(samples)
n = len(samples)
n_train, n_val = int(0.7 * n), int(0.15 * n)
splits = {"train": samples[:n_train], "val": samples[n_train:n_train+n_val], "test": samples[n_train+n_val:]}

for split_name, split_data in splits.items():
    with open(OUT / f"{split_name}.jsonl", "w") as f:
        for s in split_data:
            f.write(json.dumps(s) + "\n")
    logger.info(f"{split_name}: {len(split_data)} samples")

logger.info(f"NLP preprocessing complete: {n} total samples")
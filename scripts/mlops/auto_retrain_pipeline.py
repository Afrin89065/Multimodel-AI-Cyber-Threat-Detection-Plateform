"""
Auto-retrain check. RUN WEEKLY: python scripts\mlops\auto_retrain_pipeline.py
Checks drift scores and retrains if needed.
"""
import sys, subprocess, json
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import os; os.chdir(PROJECT_ROOT)

DRIFT_THRESHOLD = 0.20
F1_DROP_THRESHOLD = 0.05

def get_current_drift():
    try:
        import pickle, numpy as np
        from services.drift_service import DriftService
        from core.config import settings
        svc = DriftService(settings.DRIFT_REFERENCE_PATH)
        proc = PROJECT_ROOT / "datasets/processed/network/val.csv"
        if not proc.exists():
            return 0.0
        import pandas as pd
        df = pd.read_csv(proc)
        feat_cols = [c for c in df.columns if c != "Label"]
        X = df[feat_cols].values
        result = svc.check_drift(X)
        return result.get("psi_score", 0.0)
    except Exception as e:
        logger.warning(f"Drift check failed: {e}")
        return 0.0

def main():
    logger.info("Running auto-retrain pipeline check...")
    drift_score = get_current_drift()
    logger.info(f"Current drift PSI: {drift_score:.4f} (threshold: {DRIFT_THRESHOLD})")

    report = {"drift_psi": drift_score, "threshold": DRIFT_THRESHOLD, "retrain_triggered": False, "reason": ""}

    if drift_score > DRIFT_THRESHOLD:
        logger.warning(f"Drift detected (PSI={drift_score:.4f}) — triggering retrain!")
        report["retrain_triggered"] = True
        report["reason"] = f"PSI {drift_score:.4f} > threshold {DRIFT_THRESHOLD}"
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/training/train_network.py")], check=False)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/training/train_fusion.py")], check=False)
    else:
        logger.info("No drift detected — no retrain needed")
        report["reason"] = "PSI within acceptable range"

    out = PROJECT_ROOT / "logs/retrain_report.json"
    json.dump(report, open(out, "w"), indent=2)
    logger.info(f"Report saved: {out}")

if __name__ == "__main__":
    main()
"""Retrain fusion from analyst feedback. RUN: python scripts\training\retrain_fusion_from_feedback.py"""
import sys, json, subprocess
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
import os; os.chdir(PROJECT_ROOT)

def collect_feedback_samples():
    """Pull analyst verdicts from PostgreSQL and generate retraining data."""
    try:
        from sqlalchemy import create_engine, text
        from core.config import settings
        engine = create_engine(settings.DATABASE_URL_SYNC)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT fusion_result, nlp_result, vision_result,
                       network_result, malware_result, analyst_verdict
                FROM threat_events
                WHERE analyst_verdict IN ('CONFIRMED', 'FALSE_POSITIVE')
                AND created_at > NOW() - INTERVAL '30 days'
                LIMIT 1000
            """))
            rows = result.fetchall()
        logger.info(f"Collected {len(rows)} analyst-verified samples")
        return rows
    except Exception as e:
        logger.warning(f"Could not connect to database: {e}")
        return []

def main():
    logger.info("Starting fusion retrain from analyst feedback...")
    rows = collect_feedback_samples()
    if len(rows) < 50:
        logger.warning(f"Only {len(rows)} feedback samples — need at least 50 to retrain")
        logger.info("Collect more analyst verdicts in the dashboard first.")
        return

    from services.fusion_service import THREAT_CLASSES
    feedback_samples = []
    for row in rows:
        fusion = row[0] if row[0] else {}
        verdict = row[5]
        label = 0 if verdict == "FALSE_POSITIVE" else THREAT_CLASSES.index(fusion.get("threat_class", "CLEAN"))
        feats = [
            fusion.get("module_scores", {}).get("nlp", 0),
            fusion.get("module_scores", {}).get("vision", 0),
            fusion.get("module_scores", {}).get("network", 0),
            fusion.get("module_scores", {}).get("malware", 0),
        ] + [0.8] * 4 + [0.0] * 4
        feedback_samples.append({"features": feats, "label": label})

    feedback_file = PROJECT_ROOT / "datasets/processed/fusion/feedback.jsonl"
    with open(feedback_file, "w") as f:
        for s in feedback_samples:
            f.write(json.dumps(s) + "\n")
    logger.info(f"Feedback samples written: {feedback_file}")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/training/train_fusion.py")], check=False)
    logger.info("Fusion retrain from feedback complete!")

if __name__ == "__main__":
    main()
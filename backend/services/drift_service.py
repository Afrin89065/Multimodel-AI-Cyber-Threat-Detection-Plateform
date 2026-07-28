import numpy as np
import pickle
from pathlib import Path
from loguru import logger
from utils.metrics import DRIFT_SCORE


class DriftService:
    """
    Population Stability Index (PSI) drift detection.
    PSI < 0.1: no drift
    PSI 0.1–0.2: slight drift (monitor)
    PSI > 0.2: significant drift (retrain!)
    
    No v3 modifications - same as v2
    """

    def __init__(self, reference_path: str):
        self.reference = None
        if Path(reference_path).exists():
            self.reference = pickle.load(open(reference_path, "rb"))
            logger.info("Drift reference loaded")

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        eps = 1e-8
        expected_pct = np.histogram(expected, bins=bins)[0] / (len(expected) + eps)
        actual_pct = np.histogram(actual, bins=bins)[0] / (len(actual) + eps)
        expected_pct = np.clip(expected_pct, eps, 1.0)
        actual_pct = np.clip(actual_pct, eps, 1.0)
        return float(np.sum(
            (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        ))

    def check_drift(
        self, current_features: np.ndarray, module: str = "network"
    ) -> dict:
        if self.reference is None:
            return {"drift_detected": False, "psi_score": 0.0, "module": module}

        ref_mean = np.array(self.reference["mean"])
        ref_std = np.array(self.reference["std"])
        feat_names = self.reference.get("feature_names", [])

        n_features = min(current_features.shape[1], len(ref_mean))
        psi_scores = []
        drifted = []

        for i in range(n_features):
            ref_dist = np.random.normal(
                ref_mean[i], max(ref_std[i], 1e-6), 1000
            )
            psi = self._psi(ref_dist, current_features[:, i])
            psi_scores.append(psi)
            if psi > 0.2:
                name = feat_names[i] if i < len(feat_names) else f"f{i}"
                drifted.append({"feature": name, "psi": round(psi, 4)})

        avg_psi = float(np.mean(psi_scores))
        DRIFT_SCORE.labels(module=module).set(avg_psi)

        return {
            "drift_detected": avg_psi > 0.2,
            "psi_score": round(avg_psi, 4),
            "drifted_features": drifted[:10],
            "module": module,
            "recommendation": (
                "RETRAIN IMMEDIATELY" if avg_psi > 0.25
                else "MONITOR" if avg_psi > 0.1
                else "OK"
            )
        }
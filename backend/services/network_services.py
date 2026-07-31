import pickle
import numpy as np
import shap
import torch
import torch.nn as nn
from pathlib import Path
from loguru import logger
from utils.metrics import timer

# v3 NEW: TabTransformer imports
try:
    from tab_transformer_pytorch import TabTransformer as TabTransformerModel
    TAB_AVAILABLE = True
except ImportError:
    TAB_AVAILABLE = False
    logger.warning("tab-transformer-pytorch not installed — XGBoost only")

# v3 FIX: previously hardcoded to 4 classes. The trained model in
# models_store/network/xgb_nids_v2.pkl reports 7 classes via
# `.classes_`. We don't have ground-truth names for classes beyond the
# original 4 (the label→name mapping lived in whatever raw dataset produced
# network_model.pkl, which isn't in this repo) — guessing wrong names here
# would be actively dangerous (a SOC analyst could see a mislabeled attack
# class and dismiss it as benign). So class names default to this list for
# backward compatibility, and NetworkService._resolve_class_names() extends
# it with clearly-marked placeholders for any classes the model reports
# beyond what we have real names for. Replace the placeholders with real
# names in this list as soon as you know the original label encoding.
ATTACK_CLASSES = ["BENIGN", "DDoS", "PortScan", "BruteForce"]


class NetworkService:
    def __init__(self, xgb_path: str, iso_path: str,
                 scaler_path: str, features_path: str):
        self.xgb = (
            pickle.load(open(xgb_path, "rb"))
            if Path(xgb_path).exists() else None
        )
        self.class_names = self._resolve_class_names()
        self.iso = (
            pickle.load(open(iso_path, "rb"))
            if Path(iso_path).exists() else None
        )
        self.scaler = (
            pickle.load(open(scaler_path, "rb"))
            if Path(scaler_path).exists() else None
        )
        # Load feature names
        feat_file = Path(features_path)
        self.feature_names = (
            feat_file.read_text().splitlines()
            if feat_file.exists() else [f"f{i}" for i in range(80)]
        )
        self.explainer = (
            shap.TreeExplainer(
                self.xgb.calibrated_classifiers_[0].estimator
                if hasattr(self.xgb, "calibrated_classifiers_")
                else self.xgb
            ) if self.xgb else None
        )

        # v3 NEW: TabTransformer initialization
        self.tab_transformer = None
        tab_path = Path(xgb_path).parent / "tab_transformer_v2.pt"
        if TAB_AVAILABLE and tab_path.exists():
            try:
                self.tab_transformer = TabTransformerModel(
                    categories=(),
                    num_continuous=len(self.feature_names),
                    dim=32, dim_out=4, depth=4, heads=8,
                    attn_dropout=0.1, ff_dropout=0.1,
                    mlp_hidden_mults=(4, 2), mlp_act=nn.ReLU()
                )
                self.tab_transformer.load_state_dict(
                    torch.load(tab_path, map_location="cpu")
                )
                self.tab_transformer.eval()
                logger.info("TabTransformer loaded")
            except Exception as e:
                logger.warning(f"TabTransformer load failed: {e}")

        logger.info("Network Service ready")

    def _resolve_class_names(self) -> list:
        """
        Build a class-name list matching self.xgb.classes_'s length.
        Uses the known ATTACK_CLASSES names for indices we're confident
        about, and clearly-flagged placeholders for anything beyond that,
        rather than silently guessing.
        """
        n_classes = (
            len(self.xgb.classes_)
            if self.xgb is not None and hasattr(self.xgb, "classes_")
            else len(ATTACK_CLASSES)
        )
        if n_classes == len(ATTACK_CLASSES):
            return list(ATTACK_CLASSES)

        names = list(ATTACK_CLASSES[:n_classes])
        while len(names) < n_classes:
            names.append(f"UNVERIFIED_CLASS_{len(names)}")
        logger.warning(
            f"Loaded network model reports {n_classes} classes but only "
            f"{len(ATTACK_CLASSES)} have confirmed names ({ATTACK_CLASSES}). "
            f"Classes {names[len(ATTACK_CLASSES):]} are placeholders — verify "
            f"the real label encoding used when this model was trained and "
            f"update ATTACK_CLASSES in network_services.py accordingly. "
            f"Until then, do not treat UNVERIFIED_CLASS_N predictions as "
            f"benign or dismiss them in the SOC dashboard."
        )
        return names

    def analyse(self, features: list, compute_shap: bool = True) -> dict:
        with timer("network"):
            X = np.array(features, dtype=np.float32).reshape(1, -1)

            # Pad or trim to match training feature count
            expected = len(self.feature_names)
            if X.shape[1] < expected:
                X = np.pad(X, ((0, 0), (0, expected - X.shape[1])))
            else:
                X = X[:, :expected]

            if self.scaler:
                X = self.scaler.transform(X)

            # XGBoost prediction
            if self.xgb:
                probs = self.xgb.predict_proba(X)[0]
            else:
                probs = np.array([0.7, 0.1, 0.1, 0.1])

            # v3 NEW: TabTransformer ensemble
            if self.tab_transformer:
                try:
                    x_tensor = torch.tensor(X, dtype=torch.float32)
                    with torch.no_grad():
                        tab_logits = self.tab_transformer(None, x_tensor)
                        tab_probs = torch.softmax(tab_logits, dim=1).squeeze().numpy()
                    # Ensemble: 60% XGBoost + 40% TabTransformer
                    probs = 0.60 * probs + 0.40 * tab_probs
                except Exception as e:
                    logger.warning(f"TabTransformer inference failed: {e}")

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])

            # Isolation Forest anomaly score (zero-day fallback)
            iso_score = 0.0
            if confidence < 0.65 and self.iso:
                raw_score = float(-self.iso.score_samples(X)[0])
                iso_score = max(0.0, min(raw_score, 1.0))

            # SHAP feature importance
            top_features = []
            if compute_shap and self.explainer:
                try:
                    sv = self.explainer.shap_values(X)
                    if isinstance(sv, list):
                        sv_arr = sv[pred_idx][0]
                    else:
                        sv_arr = sv[0]
                    top_idx = np.argsort(np.abs(sv_arr))[-10:][::-1]
                    top_features = [
                        {
                            "feature": self.feature_names[i]
                            if i < len(self.feature_names) else f"f{i}",
                            "value": round(float(X[0, i]), 4),
                            "shap": round(float(sv_arr[i]), 4),
                            "direction": "attack" if sv_arr[i] > 0 else "benign"
                        }
                        for i in top_idx
                    ]
                except Exception as e:
                    logger.warning(f"SHAP computation failed: {e}")

            net_score = max(float(1.0 - probs[0]), iso_score)

            return {
                "network_score": round(net_score, 4),
                "attack_class": self.class_names[pred_idx],
                "confidence": round(confidence, 4),
                "uncertainty": round(1.0 - confidence, 4),
                "class_probs": {
                    self.class_names[i]: round(float(p), 4)
                    for i, p in enumerate(probs)
                },
                "iso_anomaly_score": round(iso_score, 4),
                "zero_day_flag": iso_score > 0.5,
                "top_features": top_features
            }
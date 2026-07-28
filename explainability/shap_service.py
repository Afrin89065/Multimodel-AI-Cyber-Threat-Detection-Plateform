import shap
import torch
import numpy as np
from loguru import logger
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from services.nlp_service import extract_url_features, URL_FEATURE_NAMES


class NLPSHAPExplainer:
    def __init__(self, nlp_service):
        self.svc = nlp_service

    def explain(self, text: str, url: str = "") -> dict:
        enc = self.svc.tokenizer(
            text[:500], max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )

        def predict_fn(url_features_array):
            results = []
            for row in url_features_array:
                uf = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = self.svc.model(enc["input_ids"], enc["attention_mask"], uf)
                    results.append(torch.softmax(logits, dim=1).squeeze().numpy())
            return np.array(results)

        background = np.zeros((1, 25), dtype=np.float32)
        explainer = shap.KernelExplainer(predict_fn, background)
        url_arr = extract_url_features(url).reshape(1, -1)

        try:
            shap_vals = explainer.shap_values(url_arr, nsamples=100)
            pred_class = int(np.argmax(predict_fn(url_arr)[0]))
            sv = shap_vals[pred_class][0]
            contributions = [
                {
                    "feature": name,
                    "feature_value": round(float(fv), 4),
                    "shap_value": round(float(sv_), 4),
                    "direction": "threat" if sv_ > 0 else "clean",
                    "magnitude": round(abs(float(sv_)), 4),
                }
                for name, sv_, fv in zip(URL_FEATURE_NAMES, sv, url_arr[0])
                if abs(sv_) > 0.005
            ]
            contributions.sort(key=lambda x: x["magnitude"], reverse=True)
            return {
                "explanation_type": "SHAP_KernelExplainer",
                "module": "nlp",
                "top_features": contributions[:10],
            }
        except Exception as e:
            return {"explanation_type": "SHAP", "module": "nlp", "error": str(e)}


class NetworkSHAPExplainer:
    def __init__(self, network_service):
        self.svc = network_service

    def explain(self, features: list) -> dict:
        if self.svc.xgb is None:
            return {"error": "XGBoost not loaded"}

        X = np.array(features, dtype=np.float32).reshape(1, -1)
        expected = len(self.svc.feature_names)
        if X.shape[1] < expected:
            X = np.pad(X, ((0, 0), (0, expected - X.shape[1])))
        else:
            X = X[:, :expected]
        if self.svc.scaler:
            X = self.svc.scaler.transform(X)

        try:
            base_est = (
                self.svc.xgb.calibrated_classifiers_[0].estimator
                if hasattr(self.svc.xgb, "calibrated_classifiers_")
                else self.svc.xgb
            )
            explainer = shap.TreeExplainer(base_est)
            sv = explainer.shap_values(X)
            probs = self.svc.xgb.predict_proba(X)[0]
            pred_class = int(np.argmax(probs))
            sv_arr = sv[pred_class][0] if isinstance(sv, list) else sv[0]

            contributions = [
                {
                    "feature": self.svc.feature_names[i] if i < len(self.svc.feature_names) else f"f{i}",
                    "value": round(float(X[0, i]), 4),
                    "shap": round(float(sv_arr[i]), 4),
                    "direction": "attack" if sv_arr[i] > 0 else "benign",
                    "magnitude": round(abs(float(sv_arr[i])), 4),
                }
                for i in range(len(sv_arr))
                if abs(sv_arr[i]) > 0.01
            ]
            contributions.sort(key=lambda x: x["magnitude"], reverse=True)
            return {
                "explanation_type": "SHAP_TreeExplainer",
                "module": "network",
                "top_features": contributions[:10],
                "predicted_class": ["BENIGN", "DDoS", "PortScan", "BruteForce"][pred_class],
            }
        except Exception as e:
            return {"explanation_type": "SHAP", "module": "network", "error": str(e)}
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from loguru import logger
from utils.metrics import timer, THREAT_DETECTIONS

THREAT_CLASSES = ["CLEAN", "PHISHING", "BEC", "MALWARE", "NETWORK_ATTACK"]


# v3 NEW: AttentionFusionEngine replaces FusionMLP
class AttentionFusionEngine(nn.Module):
    """
    Replaces FusionMLP. Same input/output contract, but uses
    self-attention so attention weights = native module-trust explanation.
    Input shape: [B, 4, 3] where 4 modules, 3 features each (score, confidence, threat_flag)
    """
    def __init__(self, n_modules=4, feature_dim=3, hidden_dim=64,
                 n_classes=5, dropout=0.2):
        super().__init__()
        self.module_embedding = nn.Linear(feature_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4,
            dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * n_modules, 128),
            nn.LayerNorm(128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: [B, 4, 3]
        emb = self.module_embedding(x)
        attended, attn_weights = self.attention(emb, emb, emb)
        x_out = self.dropout(self.layer_norm(emb + attended))
        return self.classifier(x_out.reshape(x_out.size(0), -1)), attn_weights

    def forward_mc(self, x, n_samples=20):
        self.train()
        logits_list, attn_list = [], []
        with torch.no_grad():
            for _ in range(n_samples):
                logits, attn = self.forward(x)
                logits_list.append(torch.softmax(logits, dim=1))
                attn_list.append(attn)
        self.eval()
        return torch.stack(logits_list).mean(0), torch.stack(logits_list).std(0), torch.stack(attn_list).mean(0)


class FusionService:
    def __init__(self, model_path: str, calibrator_path: str = ""):
        # v3 NEW: Use AttentionFusionEngine instead of FusionMLP
        self.model = AttentionFusionEngine()
        if Path(model_path).exists():
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
        self.model.eval()
        self.calibrator = (
            pickle.load(open(calibrator_path, "rb"))
            if calibrator_path and Path(calibrator_path).exists()
            else None
        )
        logger.info("Fusion Service ready (AttentionFusionEngine v3)")

    # v3 NEW: Renamed and updated
    def _build_module_tensor(self, nlp, vision, network, malware):
        """Build [B,4,3] tensor from module outputs."""
        return torch.tensor([[
            [nlp.get("nlp_score", 0.0), nlp.get("confidence", 0.5),
             0.0 if nlp.get("threat_type", "CLEAN") == "CLEAN" else 1.0],
            [vision.get("cv_score", 0.0), vision.get("vit_score", 0.5),
             0.0 if vision.get("pred_class", "LEGITIMATE") == "LEGITIMATE" else 1.0],
            [network.get("network_score", 0.0), network.get("confidence", 0.5),
             0.0 if network.get("attack_class", "BENIGN") == "BENIGN" else 1.0],
            [malware.get("malware_score", 0.0), malware.get("confidence", 0.5),
             0.0 if malware.get("verdict", "BENIGN") == "BENIGN" else 1.0],
        ]], dtype=torch.float32)

    def fuse(
        self,
        nlp: dict = None, vision: dict = None,
        network: dict = None, malware: dict = None
    ) -> dict:
        with timer("fusion"):
            nlp = nlp or {}
            vision = vision or {}
            network = network or {}
            malware = malware or {}

            scores = [
                nlp.get("nlp_score", 0.0),
                vision.get("cv_score", 0.0),
                network.get("network_score", 0.0),
                malware.get("malware_score", 0.0),
            ]
            uncertainties = [
                nlp.get("uncertainty", 0.0),
                vision.get("uncertainty", 0.0),
                network.get("uncertainty", 0.0),
                malware.get("uncertainty", 0.0),
            ]
            max_score = max(scores)

            # Hard override: very high-confidence single modality
            if max_score > 0.97:
                risk_score = max_score
                mean_class_probs = np.zeros(len(THREAT_CLASSES))
                mean_class_probs[scores.index(max_score) + 1] = max_score
                mean_class_probs[0] = 1 - max_score
                fusion_uncertainty = 0.0
                attn_weights_dict = {}
            else:
                # v3 NEW: Use AttentionFusionEngine with attention weights
                module_tensor = self._build_module_tensor(nlp, vision, network, malware)
                mean_p, std_p, attn = self.model.forward_mc(module_tensor, n_samples=20)
                mean_class_probs = mean_p.squeeze().numpy()
                fusion_uncertainty = float(std_p.squeeze().numpy().max())
                risk_score = float(1.0 - mean_class_probs[0])

                # v3 NEW: Extract attention weights for explainability
                attn_np = attn.squeeze().numpy()
                module_names = ["nlp", "vision", "network", "malware"]
                attn_weights_dict = {
                    module_names[i]: round(float(attn_np[:, i].mean()), 4) for i in range(4)
                }

                # Calibrate if available
                if self.calibrator:
                    try:
                        cal_probs = self.calibrator.predict_proba(
                            module_tensor.numpy()
                        )[0]
                        risk_score = float(1.0 - cal_probs[0])
                        mean_class_probs = cal_probs
                    except Exception:
                        pass

            # Severity classification
            if risk_score >= 0.85:
                severity = "CRITICAL"
            elif risk_score >= 0.65:
                severity = "HIGH"
            elif risk_score >= 0.35:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            # Threat class
            threat_idx = int(np.argmax(mean_class_probs))
            threat_class = THREAT_CLASSES[threat_idx]

            # Human-readable reasons
            reasons = []
            if scores[0] > 0.5:
                reasons.append(
                    f"Email/URL phishing detected "
                    f"({scores[0]:.0%} confidence) [{nlp.get('threat_type', '?')}]"
                )
            if scores[1] > 0.5:
                reasons.append(
                    f"Fake login page detected "
                    f"({scores[1]:.0%} confidence)"
                )
            if scores[2] > 0.5:
                reasons.append(
                    f"Network attack: {network.get('attack_class', '?')} "
                    f"({scores[2]:.0%} confidence)"
                )
            if scores[3] > 0.5:
                reasons.append(
                    f"Malware detected: {malware.get('family', 'unknown')} "
                    f"({scores[3]:.0%} confidence) "
                    f"via {malware.get('method', '?')}"
                )

            # Track Prometheus metrics
            THREAT_DETECTIONS.labels(
                severity=severity, threat_class=threat_class
            ).inc()

            # Aggregate uncertainty
            all_uncertainties = uncertainties + [fusion_uncertainty]
            avg_uncertainty = float(np.mean([u for u in all_uncertainties if u > 0]))

            return {
                "risk_score": round(risk_score, 4),
                "severity": severity,
                "threat_class": threat_class,
                "reason": " | ".join(reasons) if reasons else "No threat detected",
                "module_scores": {
                    "nlp": round(scores[0], 4),
                    "vision": round(scores[1], 4),
                    "network": round(scores[2], 4),
                    "malware": round(scores[3], 4),
                },
                "class_probs": {
                    THREAT_CLASSES[i]: round(float(p), 4)
                    for i, p in enumerate(mean_class_probs)
                },
                "uncertainties": {
                    "nlp": round(uncertainties[0], 4),
                    "vision": round(uncertainties[1], 4),
                    "network": round(uncertainties[2], 4),
                    "malware": round(uncertainties[3], 4),
                    "fusion": round(fusion_uncertainty, 4),
                    "aggregate": round(avg_uncertainty, 4),
                },
                "is_uncertain": avg_uncertainty > 0.15,
                "needs_human_review": avg_uncertainty > 0.20 or severity == "CRITICAL",
                # v3 NEW: Attention weights for explainability
                "attention_weights": attn_weights_dict,
                "fusion_architecture": "AttentionFusion_v3",
            }
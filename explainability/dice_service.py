from itertools import product
from loguru import logger


class DiCEService:
    def __init__(self, fusion_service):
        self.fusion = fusion_service

    def explain(
        self,
        nlp=None, vision=None, network=None, malware=None,
        target_severity: str = "LOW"
    ) -> dict:
        nlp = nlp or {}
        vision = vision or {}
        network = network or {}
        malware = malware or {}

        original = self.fusion.fuse(nlp, vision, network, malware)
        original_severity = original["severity"]

        if original_severity == target_severity:
            return {
                "counterfactual_found": True,
                "already_at_target": True,
                "original_severity": original_severity,
                "target_severity": target_severity,
                "changes_required": [],
                "interpretation": f"Already at {target_severity} severity.",
            }

        best_cf = None
        best_dist = float("inf")
        DELTAS = [0.0, -0.1, -0.2, -0.3, -0.4, -0.6, -0.8]

        for d0, d1, d2, d3 in product(DELTAS, repeat=4):
            if abs(d0) + abs(d1) + abs(d2) + abs(d3) == 0:
                continue
            cf_nlp = {**nlp, "nlp_score": max(0.0, nlp.get("nlp_score", 0) + d0)}
            cf_vis = {**vision, "cv_score": max(0.0, vision.get("cv_score", 0) + d1)}
            cf_net = {**network, "network_score": max(0.0, network.get("network_score", 0) + d2)}
            cf_mal = {**malware, "malware_score": max(0.0, malware.get("malware_score", 0) + d3)}

            result = self.fusion.fuse(cf_nlp, cf_vis, cf_net, cf_mal)
            dist = abs(d0) + abs(d1) + abs(d2) + abs(d3)

            if result["severity"] == target_severity and dist < best_dist:
                best_dist = dist
                best_cf = {
                    "resulting_severity": result["severity"],
                    "resulting_risk_score": round(result["risk_score"], 4),
                    "counterfactual_scores": {
                        "nlp": round(cf_nlp["nlp_score"], 3),
                        "vision": round(cf_vis["cv_score"], 3),
                        "network": round(cf_net["network_score"], 3),
                        "malware": round(cf_mal["malware_score"], 3),
                    },
                    "changes": {
                        k: round(v, 3)
                        for k, v in [("nlp_delta", d0), ("vision_delta", d1), ("network_delta", d2), ("malware_delta", d3)]
                        if abs(v) > 0.01
                    },
                    "total_change": round(best_dist, 3),
                }

        LABELS = {
            "nlp_delta": "Email/URL score",
            "vision_delta": "Visual score",
            "network_delta": "Network score",
            "malware_delta": "Malware score",
        }
        changes_required = []
        if best_cf:
            for key, label in LABELS.items():
                if key in best_cf["changes"]:
                    changes_required.append({
                        "module": key.replace("_delta", ""),
                        "description": label,
                        "required_change": best_cf["changes"][key],
                        "direction": "decrease" if best_cf["changes"][key] < 0 else "increase",
                    })

        if best_cf and changes_required:
            parts = [
                f"{c['description']} needs to {c['direction']} by {abs(c['required_change']):.0%}"
                for c in changes_required
            ]
            interpretation = (
                f"To reduce from {original_severity} to {target_severity}: " + "; ".join(parts)
            )
        else:
            interpretation = f"No counterfactual found within search range to reach {target_severity}."

        return {
            "explanation_type": "DiCE_Counterfactual",
            "module": "fusion",
            "original_severity": original_severity,
            "original_risk_score": round(original["risk_score"], 4),
            "target_severity": target_severity,
            "counterfactual_found": best_cf is not None,
            "counterfactual": best_cf,
            "changes_required": changes_required,
            "interpretation": interpretation,
        }
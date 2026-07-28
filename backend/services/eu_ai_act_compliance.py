"""AIDTECT v3.0 — EU AI Act Compliance Service"""
import json
from datetime import datetime
from pathlib import Path
from loguru import logger


class EUAIActComplianceService:
    SYSTEM_INFO = {
        "system_name": "AIDTECT v3.0",
        "version": "3.0.0",
        "intended_purpose": "Automated cyber threat detection across email, web pages, network traffic, and files",
        "risk_category": "High-Risk (Article 6 + Annex III — critical infrastructure protection)",
        "intended_users": "Security Operations Centre analysts",
    }

    TRAINING_DATASETS = {
        "nlp": {"name": "PhishTank + ENRON", "size": "20,000 samples", "date_range": "2004-2024",
                "bias": "English-dominant, may miss non-English phishing patterns"},
        "network": {"name": "CICIDS-2017", "size": "100,000 network flows", "date_range": "2017",
                    "bias": "2017 dataset may not reflect modern attack patterns"},
        "malware": {"name": "EMBER", "size": "10,000 PE files", "date_range": "2018",
                    "bias": "Windows PE focus; misses fileless/mobile malware"},
        "vision": {"name": "Custom screenshots + CLIP", "size": "2,000 screenshots + zero-shot",
                   "date_range": "2024", "bias": "Small dataset; CLIP generalises but may miss niche styles"},
    }

    KNOWN_LIMITATIONS = [
        "Network module trained on 2017 data — may miss novel attack patterns",
        "Fusion model partially trained on synthetic data",
        "No fileless or living-off-the-land attack detection",
        "English-language bias in NLP module",
        "No field validation in live SOC environment",
    ]

    HUMAN_OVERSIGHT = [
        "All CRITICAL severity alerts flagged for mandatory human review",
        "Uncertainty > 0.20 triggers automatic review flag",
        "Analyst verdict system: CONFIRMED / FALSE_POSITIVE / INVESTIGATING",
        "All decisions logged to PostgreSQL for forensic audit",
        "API endpoint for analyst verdict override",
    ]

    def generate_transparency_report(self, event_id: str, fusion_result: dict,
                                      shap_explanation: dict = None,
                                      counterfactual: dict = None,
                                      attack_tags: dict = None,
                                      analyst_id: str = None) -> dict:
        return {
            "regulation": "EU AI Act Article 13 — Transparency",
            "system": self.SYSTEM_INFO["system_name"],
            "event_id": event_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "analyst_id": analyst_id,
            "ai_decision": {
                "verdict": fusion_result.get("severity"),
                "risk_score": fusion_result.get("risk_score"),
                "threat_class": fusion_result.get("threat_class"),
                "confidence": round(1 - fusion_result.get("uncertainties", {}).get("aggregate", 0), 4),
                "uncertainty": fusion_result.get("uncertainties", {}).get("aggregate"),
            },
            "explanation": {
                "plain_language": fusion_result.get("reason"),
                "feature_attribution": shap_explanation.get("top_features", []) if shap_explanation else [],
                "counterfactual": counterfactual.get("interpretation") if counterfactual else None,
                "attention_weights": fusion_result.get("attention_weights"),
                "mitre_att_ck": attack_tags.get("mitre_tags", []) if attack_tags else [],
                "methods": ["SHAP", "GradCAM", "DiCE", "Attention"],
            },
            "human_oversight": {
                "human_review_required": fusion_result.get("needs_human_review"),
                "analyst_can_override": True,
                "override_options": ["CONFIRMED", "FALSE_POSITIVE", "INVESTIGATING"],
            },
            "data_sources": {k: v["name"] for k, v in self.TRAINING_DATASETS.items()},
            "known_limitations": self.KNOWN_LIMITATIONS[:3],
        }

    def generate_technical_documentation(self, accuracy_metrics: dict = None) -> dict:
        return {
            "document_type": "EU AI Act Article 11 Technical Documentation",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "system_description": self.SYSTEM_INFO,
            "architecture": {
                "type": "Multi-modal late-fusion ensemble",
                "modules": {
                    "nlp": "SecureBERT + URL features + SHAP",
                    "vision": "MobileNetV3 + CLIP ViT-B/32 + GradCAM",
                    "network": "XGBoost + TabTransformer + Isolation Forest + SHAP",
                    "malware": "YARA + MalBERT + ResNet18 + XGBoost",
                    "fusion": "Attention-based MLP + DiCE + Calibration",
                },
                "explainability": ["SHAP", "GradCAM", "DiCE", "Attention weights"],
                "uncertainty": "Monte Carlo Dropout (n=15-20 samples)",
            },
            "training_data": self.TRAINING_DATASETS,
            "performance_metrics": accuracy_metrics or {"note": "See MLflow experiment logs"},
            "known_limitations": self.KNOWN_LIMITATIONS,
            "human_oversight": self.HUMAN_OVERSIGHT,
            "compliance_status": {
                "article_9_risk_mgmt": "✅ Drift detection + retraining pipeline",
                "article_10_data": "✅ Dataset documentation above",
                "article_11_tech_docs": "✅ This document",
                "article_13_transparency": "✅ SHAP + GradCAM + DiCE explanations",
                "article_14_oversight": "✅ Analyst verdict + human review flags",
                "article_15_robustness": "✅ Adversarial testing + uncertainty estimation",
            },
        }
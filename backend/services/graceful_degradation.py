"""
Wraps each AI model with a circuit breaker.
If a model fails, it returns a safe fallback instead of crashing the whole app.
PATH: backend/services/graceful_degradation.py
"""
from loguru import logger
from typing import Any


class FallbackModel:
    """Safe fallback when a model fails to load or throws an error."""

    def __init__(self, module_name: str):
        self.module_name = module_name

    def analyse(self, *args, **kwargs) -> dict:
        return {
            "error": f"{self.module_name} unavailable",
            "fallback": True,
            f"{self.module_name}_score": 0.0,
            "confidence": 0.0,
            "uncertainty": 1.0,
        }

    def fuse(self, *args, **kwargs) -> dict:
        return {
            "risk_score": 0.0,
            "severity": "LOW",
            "threat_class": "CLEAN",
            "reason": "Fusion model unavailable — fallback response",
            "module_scores": {"nlp": 0, "vision": 0, "network": 0, "malware": 0},
            "uncertainties": {"aggregate": 1.0},
            "attention_weights": None,
            "fusion_architecture": "FALLBACK",
            "needs_human_review": True,
            "is_uncertain": True,
        }

    def check_drift(self, *args, **kwargs) -> dict:
        return {"drift_detected": False, "psi_score": 0.0, "module": self.module_name}

    def _build_module_tensor(self, *args, **kwargs):
        import torch
        return torch.zeros(1, 4, 3)


class SafeModel:
    """Wraps a real model. On error, logs and returns fallback."""

    def __init__(self, model: Any, module_name: str):
        self.model = model
        self.module_name = module_name
        self.fallback = FallbackModel(module_name)

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            try:
                return getattr(self.model, name)(*args, **kwargs)
            except Exception as e:
                logger.error(f"[{self.module_name}] {name} failed: {e}")
                fallback_fn = getattr(self.fallback, name, None)
                if fallback_fn:
                    return fallback_fn(*args, **kwargs)
                return {"error": str(e), "fallback": True}
        return wrapper


def wrap_all_models(models: dict) -> dict:
    """Wrap every model with SafeModel for graceful degradation."""
    wrapped = {}
    for name, model in models.items():
        if model is not None:
            wrapped[name] = SafeModel(model, name)
            logger.info(f"✅ {name} model wrapped with SafeModel")
        else:
            wrapped[name] = FallbackModel(name)
            logger.warning(f"⚠️  {name} using FallbackModel (model not loaded)")
    return wrapped
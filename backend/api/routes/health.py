from fastapi import APIRouter, Request, Depends
from pathlib import Path
from core.security import get_current_user, TokenData
from core.config import settings

router = APIRouter(tags=["health"])

@router.get("/health")
async def health(request: Request):
    models = getattr(request.app.state, "models", {})
    return {"status": "healthy", "version": "3.0.0",
            "models_loaded": bool(models), "modules": list(models.keys()) if models else []}

@router.get("/health/ready")
async def readiness(request: Request, current_user: TokenData = Depends(get_current_user)):
    models = getattr(request.app.state, "models", {})
    required = ["nlp", "vision", "network", "malware", "fusion", "drift"]
    missing = [m for m in required if m not in models]
    return {
        "ready": len(missing) == 0,
        "missing": missing,
        "model_files": {
            "nlp":     Path(settings.NLP_MODEL_PATH).exists(),
            "vision":  Path(settings.VISION_CLF_PATH).exists(),
            "network": Path(settings.NETWORK_XGB_PATH).exists(),
            "malware": Path(settings.MALWARE_CNN_PATH).exists(),
            "fusion":  Path(settings.FUSION_PATH).exists(),
        },
        "services": {
            "attack_mapper":    getattr(request.app.state, "attack_mapper", None) is not None,
            "knowledge_graph":  getattr(request.app.state, "knowledge_graph", None) is not None,
            "threat_intel":     getattr(request.app.state, "threat_intel", None) is not None,
            "virustotal":       getattr(request.app.state, "virustotal", None) is not None,
        }
    }
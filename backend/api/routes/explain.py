from fastapi import APIRouter, Request, Depends, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from core.security import get_current_user, TokenData

router = APIRouter(tags=["explain"])

class ExplainFusionRequest(BaseModel):
    nlp_result: Optional[dict] = None
    vision_result: Optional[dict] = None
    network_result: Optional[dict] = None
    malware_result: Optional[dict] = None
    target_severity: str = "LOW"

class ExplainNLPRequest(BaseModel):
    text: str
    url: Optional[str] = ""

class ExplainNetworkRequest(BaseModel):
    features: list

class CrossModalRequest(BaseModel):
    email_text: str
    url: Optional[str] = ""

@router.post("/explain/fusion")
async def explain_fusion(request: Request, body: ExplainFusionRequest, current_user: TokenData = Depends(get_current_user)):
    from explainability.dice_service import DiCEService
    return DiCEService(request.app.state.models["fusion"]).explain(
        nlp=body.nlp_result or {}, vision=body.vision_result or {},
        network=body.network_result or {}, malware=body.malware_result or {},
        target_severity=body.target_severity)

@router.post("/explain/nlp")
async def explain_nlp(request: Request, body: ExplainNLPRequest, current_user: TokenData = Depends(get_current_user)):
    from explainability.shap_service import NLPSHAPExplainer
    return NLPSHAPExplainer(request.app.state.models["nlp"]).explain(body.text, body.url or "")

@router.post("/explain/network")
async def explain_network(request: Request, body: ExplainNetworkRequest, current_user: TokenData = Depends(get_current_user)):
    from explainability.shap_service import NetworkSHAPExplainer
    return NetworkSHAPExplainer(request.app.state.models["network"]).explain(body.features)

@router.post("/explain/vision")
async def explain_vision(request: Request, file: UploadFile = File(...), current_user: TokenData = Depends(get_current_user)):
    from explainability.gradcam_service import GradCAMService
    image_bytes = await file.read()
    return GradCAMService(request.app.state.models["vision"].clf).generate(image_bytes)

@router.post("/explain/cross_modal")
async def explain_cross_modal(request: Request, body: CrossModalRequest, file: Optional[UploadFile] = File(None), current_user: TokenData = Depends(get_current_user)):
    vision_svc = request.app.state.models["vision"]
    if not hasattr(vision_svc, "clip") or not vision_svc.clip.available:
        return {"error": "CLIP not available", "fix": "pip install open-clip-torch"}
    if file:
        image_bytes = await file.read()
    elif body.url:
        svc = getattr(request.app.state, "screenshot_svc", None)
        if svc:
            image_bytes = await svc.take_screenshot(body.url)
            if not image_bytes:
                return {"error": "Screenshot failed"}
        else:
            return {"error": "No image and no screenshot service"}
    else:
        return {"error": "Provide a file or URL"}
    result = vision_svc.clip.check_consistency(body.email_text, image_bytes)
    result["brand_detection"] = vision_svc.clip.detect_brand(image_bytes)
    return result

@router.post("/explain/attention")
async def explain_attention(request: Request, body: ExplainFusionRequest, current_user: TokenData = Depends(get_current_user)):
    fusion_svc = request.app.state.models["fusion"]
    module_tensor = fusion_svc._build_module_tensor(
        body.nlp_result or {}, body.vision_result or {},
        body.network_result or {}, body.malware_result or {})
    mean_p, std_p, attn = fusion_svc.model.forward_mc(module_tensor, n_samples=20)
    attn_np = attn.squeeze().numpy()
    mnames = ["nlp", "vision", "network", "malware"]
    attention_received = {mnames[i]: round(float(attn_np[:, i].mean()), 4) for i in range(4)}
    most_trusted = max(attention_received, key=attention_received.get)
    trust_pct = round(attention_received[most_trusted] * 100, 1)
    desc = {"nlp": "email/URL analysis", "vision": "visual page analysis",
            "network": "network traffic pattern", "malware": "file/binary analysis"}
    return {
        "explanation_type": "AttentionWeights", "module": "fusion",
        "attention_received": attention_received, "attention_matrix": attn_np.tolist(),
        "most_trusted_module": most_trusted,
        "interpretation": f"Fusion relied most on {most_trusted.upper()} ({trust_pct}%) — {desc.get(most_trusted)}.",
        "uncertainty": float(std_p.squeeze().numpy().max()),
        "risk_score": round(float(1.0 - mean_p.squeeze().numpy()[0]), 4),
    }
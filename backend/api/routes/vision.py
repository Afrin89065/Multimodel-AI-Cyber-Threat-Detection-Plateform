from fastapi import APIRouter, Request, Depends, UploadFile, File
from typing import Optional
from core.security import get_current_user, TokenData
from utils.metrics import INFERENCE_TOTAL

router = APIRouter(tags=["vision"])

@router.post("/analyse/vision")
async def analyse_vision(
    request: Request,
    file: UploadFile = File(...),
    email_text: Optional[str] = "",
    explain: bool = False,
    current_user: TokenData = Depends(get_current_user)
):
    image_bytes = await file.read()
    result = request.app.state.models["vision"].analyse(image_bytes, email_text or "")
    INFERENCE_TOTAL.labels(module="vision", status="success").inc()
    if explain:
        try:
            from explainability.gradcam_service import GradCAMService
            result["gradcam_explanation"] = GradCAMService(
                request.app.state.models["vision"].clf
            ).generate(image_bytes)
        except Exception as e:
            result["gradcam_explanation"] = {"error": str(e)}
    return result
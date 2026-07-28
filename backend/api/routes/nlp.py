from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel
from typing import Optional
from core.security import get_current_user, TokenData
from utils.metrics import INFERENCE_TOTAL

router = APIRouter(tags=["nlp"])

class NLPRequest(BaseModel):
    text: str
    url: Optional[str] = ""
    use_uncertainty: bool = True

@router.post("/analyse/nlp")
async def analyse_nlp(request: Request, body: NLPRequest, current_user: TokenData = Depends(get_current_user)):
    result = request.app.state.models["nlp"].analyse(body.text, body.url or "", body.use_uncertainty)
    ti = getattr(request.app.state, "threat_intel", None)
    if ti and body.url:
        result["threat_intel"] = ti.check_url(body.url)
    INFERENCE_TOTAL.labels(module="nlp", status="success").inc()
    return result
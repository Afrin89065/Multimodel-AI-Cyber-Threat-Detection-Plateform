from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel
from core.security import get_current_user, TokenData
from utils.metrics import INFERENCE_TOTAL

router = APIRouter(tags=["network"])

class NetworkRequest(BaseModel):
    features: list
    compute_shap: bool = True

@router.post("/analyse/network")
async def analyse_network(request: Request, body: NetworkRequest, current_user: TokenData = Depends(get_current_user)):
    result = request.app.state.models["network"].analyse(body.features, body.compute_shap)
    try:
        import numpy as np
        X = np.array(body.features, dtype=np.float32).reshape(1, -1)
        result["drift_check"] = request.app.state.models["drift"].check_drift(X)
    except Exception:
        pass
    INFERENCE_TOTAL.labels(module="network", status="success").inc()
    return result
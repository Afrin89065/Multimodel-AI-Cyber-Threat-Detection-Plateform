from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.security import get_current_user, TokenData

router = APIRouter(tags=["queue"])

class QueueRequest(BaseModel):
    nlp_result: Optional[dict] = None
    vision_result: Optional[dict] = None
    network_result: Optional[dict] = None
    malware_result: Optional[dict] = None

@router.post("/queue/submit")
async def submit_to_queue(request: Request, body: QueueRequest, current_user: TokenData = Depends(get_current_user)):
    queue = getattr(request.app.state, "queue", None)
    if not queue:
        raise HTTPException(status_code=503, detail="Queue service unavailable")
    job_id = await queue.enqueue(body.dict())
    return {"job_id": job_id, "status": "QUEUED"}

@router.get("/queue/result/{job_id}")
async def get_queue_result(job_id: str, request: Request, current_user: TokenData = Depends(get_current_user)):
    queue = getattr(request.app.state, "queue", None)
    if not queue:
        raise HTTPException(status_code=503, detail="Queue service unavailable")
    return await queue.get_result(job_id)
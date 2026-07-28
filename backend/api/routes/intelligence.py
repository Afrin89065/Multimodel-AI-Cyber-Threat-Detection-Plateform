from fastapi import APIRouter, Request, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.security import get_current_user, TokenData, require_role

router = APIRouter(tags=["intelligence"])

@router.get("/intelligence/graph/campaigns")
async def get_campaigns(request: Request, min_events: int = 3, current_user: TokenData = Depends(get_current_user)):
    kg = getattr(request.app.state, "knowledge_graph", None)
    if not kg or not kg.driver:
        return {"campaigns": [], "graph_available": False}
    return {"campaigns": kg.detect_campaigns(min_events=min_events)}

@router.get("/intelligence/graph/ip/{ip_address}")
async def get_ip_history(ip_address: str, request: Request, hours: int = 24, current_user: TokenData = Depends(get_current_user)):
    kg = getattr(request.app.state, "knowledge_graph", None)
    if not kg or not kg.driver:
        return {"events": [], "graph_available": False}
    return {"ip": ip_address, "events": kg.get_related_events(ip_address, hours=hours)}

@router.get("/intelligence/graph/malware")
async def get_malware_families(request: Request, current_user: TokenData = Depends(get_current_user)):
    kg = getattr(request.app.state, "knowledge_graph", None)
    if not kg or not kg.driver:
        return {"families": [], "graph_available": False}
    return {"families": kg.find_common_malware()}

class STIXRequest(BaseModel):
    fusion_result: dict
    nlp_result: Optional[dict] = None
    malware_result: Optional[dict] = None
    event_id: Optional[str] = None

@router.post("/intelligence/stix")
async def generate_stix(request: Request, body: STIXRequest, current_user: TokenData = Depends(get_current_user)):
    stix_svc = getattr(request.app.state, "stix", None)
    if not stix_svc:
        raise HTTPException(status_code=503, detail="STIX unavailable. pip install stix2")
    return stix_svc.detection_to_stix(body.fusion_result, body.nlp_result, body.malware_result, body.event_id)

class ComplianceRequest(BaseModel):
    event_id: str
    fusion_result: dict
    shap_explanation: Optional[dict] = None
    counterfactual: Optional[dict] = None
    attack_tags: Optional[dict] = None

@router.post("/intelligence/compliance/transparency")
async def transparency_report(request: Request, body: ComplianceRequest, current_user: TokenData = Depends(get_current_user)):
    svc = getattr(request.app.state, "compliance", None)
    if not svc:
        raise HTTPException(status_code=503, detail="Compliance service unavailable")
    return svc.generate_transparency_report(
        event_id=body.event_id, fusion_result=body.fusion_result,
        shap_explanation=body.shap_explanation, counterfactual=body.counterfactual,
        attack_tags=body.attack_tags, analyst_id=current_user.username)

@router.get("/intelligence/compliance/technical-docs")
async def technical_docs(request: Request, current_user: TokenData = Depends(require_role("admin"))):
    svc = getattr(request.app.state, "compliance", None)
    if not svc:
        raise HTTPException(status_code=503, detail="Compliance service unavailable")
    return svc.generate_technical_documentation()

class ATTACKRequest(BaseModel):
    fusion_result: dict
    nlp_result: Optional[dict] = None
    vision_result: Optional[dict] = None
    network_result: Optional[dict] = None
    malware_result: Optional[dict] = None

@router.post("/intelligence/attack/tag")
async def attack_tag(request: Request, body: ATTACKRequest, current_user: TokenData = Depends(get_current_user)):
    mapper = getattr(request.app.state, "attack_mapper", None)
    if not mapper:
        raise HTTPException(status_code=503, detail="ATT&CK mapper unavailable")
    return mapper.tag_detection(body.fusion_result, body.nlp_result, body.vision_result, body.network_result, body.malware_result)
from fastapi import APIRouter, Request, Depends
from pydantic import BaseModel
from typing import Optional
import uuid, datetime
from core.database import get_db
from core.security import get_current_user, TokenData
from core.websocket_manager import ws_manager
from utils.audit_logger import AuditLogger
from utils.metrics import INFERENCE_TOTAL
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

router = APIRouter(tags=["fusion"])

class FullAnalysisRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = ""
    network_features: Optional[list] = None
    nlp_result: Optional[dict] = None
    vision_result: Optional[dict] = None
    network_result: Optional[dict] = None
    malware_result: Optional[dict] = None
    explain: bool = False
    output_stix: bool = False
    enrich_vt: bool = False
    source_ip: Optional[str] = None


@router.post("/analyse/full")
async def full_analysis(
    request: Request, body: FullAnalysisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: TokenData = Depends(get_current_user)
):
    models = request.app.state.models
    request_id = str(uuid.uuid4())
    nlp_result = vision_result = network_result = malware_result = None

    if body.text or body.url:
        nlp_result = body.nlp_result or models["nlp"].analyse(body.text or "", body.url or "")
        INFERENCE_TOTAL.labels(module="nlp", status="success").inc()

    if body.network_features:
        network_result = body.network_result or models["network"].analyse(body.network_features, compute_shap=False)
        INFERENCE_TOTAL.labels(module="network", status="success").inc()
        # v3 FIX: drift checking previously only happened in the standalone
        # /analyse/network route, so DriftAlert on the dashboard never had
        # data to show — network_result (and therefore the persisted
        # ThreatEvent row and the live websocket feed) never included it.
        try:
            import numpy as np
            X = np.array(body.network_features, dtype=np.float32).reshape(1, -1)
            network_result["drift_check"] = models["drift"].check_drift(X)
        except Exception as e:
            logger.warning(f"Drift check failed: {e}")

    if body.url and not body.vision_result:
        svc = getattr(request.app.state, "screenshot_svc", None)
        if svc:
            try:
                screenshot_bytes = await svc.take_screenshot(body.url)
                if screenshot_bytes:
                    vision_result = models["vision"].analyse(screenshot_bytes, email_text=body.text or "")
                    INFERENCE_TOTAL.labels(module="vision", status="success").inc()
            except Exception as e:
                logger.warning(f"Screenshot failed: {e}")

    vision_result = vision_result or body.vision_result
    malware_result = body.malware_result

    fusion_result = models["fusion"].fuse(
        nlp=nlp_result, vision=vision_result,
        network=network_result, malware=malware_result
    )
    INFERENCE_TOTAL.labels(module="fusion", status="success").inc()

    attack_tags = None
    mapper = getattr(request.app.state, "attack_mapper", None)
    if mapper:
        try:
            attack_tags = mapper.tag_detection(fusion_result, nlp_result, vision_result, network_result, malware_result)
            fusion_result["mitre_tags"] = attack_tags.get("mitre_tags", [])
            fusion_result["mitre_tactics"] = attack_tags.get("tactics", [])
        except Exception as e:
            logger.warning(f"ATT&CK tagging failed: {e}")

    kg = getattr(request.app.state, "knowledge_graph", None)
    if kg:
        try:
            kg.ingest_event(fusion_result, nlp_result, network_result, malware_result, body.source_ip, request_id)
        except Exception as e:
            logger.warning(f"KG ingestion failed: {e}")

    shap_values = counterfactual = None
    if body.explain:
        if nlp_result and body.text:
            try:
                from explainability.shap_service import NLPSHAPExplainer
                shap_values = NLPSHAPExplainer(models["nlp"]).explain(body.text, body.url or "")
            except Exception as e:
                logger.warning(f"SHAP failed: {e}")
        try:
            from explainability.dice_service import DiCEService
            counterfactual = DiCEService(models["fusion"]).explain(
                nlp=nlp_result, vision=vision_result, network=network_result, malware=malware_result)
        except Exception as e:
            logger.warning(f"DiCE failed: {e}")

    vt_result = cs_result = stix_bundle = compliance_report = None

    if body.enrich_vt and getattr(request.app.state, "virustotal", None) and body.url:
        try:
            vt_result = await request.app.state.virustotal.check_url(body.url)
        except Exception as e:
            logger.warning(f"VT failed: {e}")

    if body.source_ip and getattr(request.app.state, "crowdstrike", None):
        if request.app.state.crowdstrike.available:
            try:
                cs_result = request.app.state.crowdstrike.enrich_ip(body.source_ip)
            except Exception as e:
                logger.warning(f"CrowdStrike failed: {e}")

    if body.output_stix and getattr(request.app.state, "stix", None):
        try:
            stix_bundle = request.app.state.stix.detection_to_stix(
                fusion_result, nlp_result, malware_result, request_id)
        except Exception as e:
            logger.warning(f"STIX failed: {e}")

    compliance_svc = getattr(request.app.state, "compliance", None)
    if compliance_svc and fusion_result.get("severity") in ("HIGH", "CRITICAL"):
        try:
            compliance_report = compliance_svc.generate_transparency_report(
                event_id=request_id, fusion_result=fusion_result,
                shap_explanation=shap_values, counterfactual=counterfactual,
                attack_tags=attack_tags, analyst_id=current_user.username)
        except Exception as e:
            logger.warning(f"Compliance report failed: {e}")

    try:
        await AuditLogger.log_inference(
            db=db, fusion_result=fusion_result, nlp_result=nlp_result,
            vision_result=vision_result, network_result=network_result,
            malware_result=malware_result, shap_values=shap_values,
            counterfactual=counterfactual, analyst_id=current_user.username,
            source_ip=body.source_ip)
    except Exception as e:
        logger.error(f"Audit log failed: {e}")

    await ws_manager.broadcast({
        "request_id": request_id,
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "severity": fusion_result["severity"],
        "risk_score": fusion_result["risk_score"],
        "threat_class": fusion_result.get("threat_class"),
        "reason": fusion_result["reason"],
        "module_scores": fusion_result["module_scores"],
        "attention_weights": fusion_result.get("attention_weights"),
        "mitre_tags": fusion_result.get("mitre_tags", []),
        "needs_human_review": fusion_result.get("needs_human_review", False),
        # v3 FIX: was computed (see above) but never actually sent anywhere
        "drift_check": (network_result or {}).get("drift_check"),
    }, room="soc")

    notif_svc = getattr(request.app.state, "notifications", None)
    if notif_svc and fusion_result["severity"] in ("CRITICAL", "HIGH"):
        try:
            import asyncio
            asyncio.create_task(notif_svc.notify(fusion_result, request_id))
        except Exception:
            pass

    return {
        "request_id": request_id, "fusion": fusion_result,
        "nlp": nlp_result, "vision": vision_result,
        "network": network_result, "malware": malware_result,
        "attack_tags": attack_tags,
        "explanation": {"shap": shap_values, "counterfactual": counterfactual} if body.explain else None,
        "enrichments": {"virustotal": vt_result, "crowdstrike": cs_result},
        "stix_bundle": stix_bundle, "compliance": compliance_report,
    }


@router.post("/analyse/fusion")
async def fusion_only(request: Request, body: FullAnalysisRequest, current_user: TokenData = Depends(get_current_user)):
    result = request.app.state.models["fusion"].fuse(
        nlp=body.nlp_result or {}, vision=body.vision_result or {},
        network=body.network_result or {}, malware=body.malware_result or {})
    mapper = getattr(request.app.state, "attack_mapper", None)
    if mapper:
        try:
            tags = mapper.tag_detection(result)
            result["mitre_tags"] = tags.get("mitre_tags", [])
            result["mitre_tactics"] = tags.get("tactics", [])
        except Exception:
            pass
    return result
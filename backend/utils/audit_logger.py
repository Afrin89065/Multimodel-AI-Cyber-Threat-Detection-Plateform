import uuid
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession
from models.db_models import ThreatEvent

class AuditLogger:
    @staticmethod
    async def log_inference(
        db: AsyncSession,
        fusion_result: dict,
        nlp_result=None,
        vision_result=None,
        network_result=None,
        malware_result=None,
        shap_values=None,
        counterfactual=None,
        analyst_id=None,
        source_ip=None,
    ) -> str:
        request_id = str(uuid.uuid4())
        scores = fusion_result.get("module_scores", {})
        uncertainties = fusion_result.get("uncertainties", {})
        event = ThreatEvent(
            severity=fusion_result.get("severity", "LOW"),
            risk_score=fusion_result.get("risk_score", 0.0),
            threat_class=fusion_result.get("threat_class", "UNKNOWN"),
            reason=fusion_result.get("reason", ""),
            nlp_score=scores.get("nlp", 0.0),
            cv_score=scores.get("vision", 0.0),
            network_score=scores.get("network", 0.0),
            malware_score=scores.get("malware", 0.0),
            nlp_uncertainty=uncertainties.get("nlp", 0.0),
            cv_uncertainty=uncertainties.get("vision", 0.0),
            network_uncertainty=uncertainties.get("network", 0.0),
            malware_uncertainty=uncertainties.get("malware", 0.0),
            nlp_result=nlp_result,
            vision_result=vision_result,
            network_result=network_result,
            malware_result=malware_result,
            fusion_result=fusion_result,
            shap_values=shap_values,
            counterfactual=counterfactual,
            analyst_id=analyst_id,
            source_ip=source_ip,
            request_id=request_id,
            needs_human_review=fusion_result.get("needs_human_review", False),
        )
        db.add(event)
        await db.flush()
        logger.info(f"Audit: {request_id} | severity={event.severity} | risk={event.risk_score:.2f}")
        return request_id
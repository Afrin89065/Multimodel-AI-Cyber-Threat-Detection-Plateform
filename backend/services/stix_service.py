"""AIDTECT v3.0 — STIX 2.1 Output Service"""
import json
from datetime import datetime
from loguru import logger

try:
    from stix2 import Indicator, Bundle, Relationship, Malware, AttackPattern, Identity
    STIX_AVAILABLE = True
    AIDTECT_IDENTITY = Identity(
        name="AIDTECT v3.0",
        identity_class="system",
        description="AIDTECT Multi-Modal XAI Cyber Defense Platform"
    )
except ImportError:
    STIX_AVAILABLE = False
    logger.warning("stix2 not installed. Run: pip install stix2")


class STIXService:
    def detection_to_stix(self, fusion_result: dict, nlp_result: dict = None,
                           malware_result: dict = None, event_id: str = None) -> dict:
        if not STIX_AVAILABLE:
            return {"error": "stix2 not installed", "stix_version": "2.1"}

        objects = [AIDTECT_IDENTITY]
        threat_class = fusion_result.get("threat_class", "CLEAN")
        risk_score = fusion_result.get("risk_score", 0.0)

        try:
            if nlp_result and nlp_result.get("url") and threat_class in ("PHISHING", "BEC"):
                url = nlp_result["url"]
                indicator = Indicator(
                    name="Phishing URL detected by AIDTECT",
                    description=f"AIDTECT NLP: phishing URL. Score: {nlp_result.get('nlp_score', 0):.2f}",
                    pattern=f"[url:value = '{url}']",
                    pattern_type="stix",
                    valid_from=datetime.utcnow(),
                    confidence=int(risk_score * 100),
                    labels=["malicious-activity", "phishing"],
                    created_by_ref=AIDTECT_IDENTITY.id
                )
                objects.append(indicator)
                attack_pattern = AttackPattern(
                    name="Phishing: Spearphishing Link",
                    external_references=[{
                        "source_name": "mitre-attack",
                        "external_id": "T1566.002",
                        "url": "https://attack.mitre.org/techniques/T1566/002"
                    }]
                )
                objects.append(attack_pattern)
                objects.append(Relationship(
                    relationship_type="indicates",
                    source_ref=indicator.id,
                    target_ref=attack_pattern.id
                ))

            if malware_result and malware_result.get("verdict") == "MALWARE":
                malware_obj = Malware(
                    name=malware_result.get("family", "Unknown"),
                    is_family=True,
                    description=f"AIDTECT malware: {malware_result.get('method', '?')}, score={malware_result.get('malware_score', 0):.2f}",
                    labels=["trojan"]
                )
                objects.append(malware_obj)

            bundle = Bundle(objects=objects)
            bundle_dict = json.loads(bundle.serialize())
            return {
                "stix_version": "2.1",
                "bundle_id": bundle_dict.get("id"),
                "event_id": event_id,
                "threat_class": threat_class,
                "object_count": len(bundle_dict.get("objects", [])),
                "bundle": bundle_dict,
            }
        except Exception as e:
            logger.error(f"STIX generation failed: {e}")
            return {"error": str(e), "stix_version": "2.1"}
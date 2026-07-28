"""AIDTECT v3.0 — MITRE ATT&CK Auto-Tagger"""
from loguru import logger

THREAT_TO_ATTACK = {
    "PHISHING": {
        "tactic": "Initial Access", "tactic_id": "TA0001",
        "technique_id": "T1566", "technique_name": "Phishing",
        "mitre_url": "https://attack.mitre.org/techniques/T1566/"
    },
    "BEC": {
        "tactic": "Collection", "tactic_id": "TA0009",
        "technique_id": "T1114", "technique_name": "Email Collection",
        "mitre_url": "https://attack.mitre.org/techniques/T1114/"
    },
    "MALWARE": {
        "tactic": "Execution", "tactic_id": "TA0002",
        "technique_id": "T1204", "technique_name": "User Execution",
        "mitre_url": "https://attack.mitre.org/techniques/T1204/"
    },
    "NETWORK_ATTACK": {
        "DDoS": {"tactic": "Impact", "tactic_id": "TA0040",
                 "technique_id": "T1498", "technique_name": "Network DoS"},
        "PortScan": {"tactic": "Discovery", "tactic_id": "TA0007",
                     "technique_id": "T1046", "technique_name": "Network Service Discovery"},
        "BruteForce": {"tactic": "Credential Access", "tactic_id": "TA0006",
                       "technique_id": "T1110", "technique_name": "Brute Force"},
    },
    "CLEAN": None
}

SEVERITY_COLORS = {
    "CRITICAL": "#FF0000", "HIGH": "#FF6600",
    "MEDIUM": "#FFAA00", "LOW": "#FFFF00"
}


class ATTACKMapper:
    def tag_detection(self, fusion_result: dict, nlp_result: dict = None,
                      vision_result: dict = None, network_result: dict = None,
                      malware_result: dict = None) -> dict:
        threat_class = fusion_result.get("threat_class", "CLEAN")
        severity = fusion_result.get("severity", "LOW")
        risk_score = fusion_result.get("risk_score", 0.0)

        if threat_class == "CLEAN":
            return {"techniques": [], "tactics": [], "mitre_tags": [],
                    "navigator_layer": {"name": "AIDTECT", "domain": "enterprise-attack",
                                        "techniques": []}}

        techniques = []
        mapping = THREAT_TO_ATTACK.get(threat_class)

        if threat_class == "NETWORK_ATTACK":
            attack_class = (network_result or {}).get("attack_class", "")
            mapping = THREAT_TO_ATTACK["NETWORK_ATTACK"].get(attack_class, {})

        if mapping:
            techniques.append({
                "technique_id": mapping.get("technique_id", ""),
                "technique_name": mapping.get("technique_name", ""),
                "tactic": mapping.get("tactic", ""),
                "tactic_id": mapping.get("tactic_id", ""),
                "confidence": risk_score,
                "mitre_url": mapping.get("mitre_url", ""),
            })

        color = SEVERITY_COLORS.get(severity, "#888888")
        navigator_layer = {
            "name": "AIDTECT Detection",
            "versions": {"attack": "14", "navigator": "4.9"},
            "domain": "enterprise-attack",
            "description": f"AIDTECT v3.0 — severity: {severity}",
            "techniques": [
                {"techniqueID": t["technique_id"],
                 "score": int(t["confidence"] * 100),
                 "color": color, "enabled": True}
                for t in techniques
            ]
        }

        return {
            "techniques": techniques,
            "tactics": list({t["tactic"] for t in techniques}),
            "tactic_ids": list({t.get("tactic_id", "") for t in techniques}),
            "mitre_tags": [t["technique_id"] for t in techniques],
            "navigator_layer": navigator_layer,
            "threat_class": threat_class,
        }
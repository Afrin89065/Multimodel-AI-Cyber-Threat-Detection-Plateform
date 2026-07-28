"""AIDTECT v3.0 — CrowdStrike Falcon integration. Set CROWDSTRIKE_* in .env"""
import os
from loguru import logger

try:
    from falconpy import APIHarnessV2, Hosts, Intel
    CROWDSTRIKE_SDK = True
except ImportError:
    CROWDSTRIKE_SDK = False
    logger.warning("crowdstrike-falconpy not installed. Run: pip install crowdstrike-falconpy")


class CrowdStrikeConnector:
    def __init__(self):
        self.available = False
        if not CROWDSTRIKE_SDK:
            return
        client_id = os.getenv("CROWDSTRIKE_CLIENT_ID", "")
        client_secret = os.getenv("CROWDSTRIKE_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            logger.info("CrowdStrike credentials not set — add to .env to enable")
            return
        try:
            self.falcon = APIHarnessV2(client_id=client_id, client_secret=client_secret)
            self.available = True
            logger.info("CrowdStrike Falcon connector ready")
        except Exception as e:
            logger.warning(f"CrowdStrike init failed: {e}")

    def enrich_ip(self, ip_address: str) -> dict:
        if not self.available:
            return {"enrichment_available": False}
        try:
            hosts = Hosts(auth_object=self.falcon)
            response = hosts.query_devices_by_filter(filter=f"external_ip:'{ip_address}'")
            if response["status_code"] == 200:
                device_ids = response["body"].get("resources", [])
                return {
                    "enrichment_available": True, "source": "CrowdStrike Falcon",
                    "ip": ip_address, "managed_devices": len(device_ids),
                    "is_known_managed": len(device_ids) > 0,
                }
        except Exception as e:
            logger.warning(f"CrowdStrike IP enrichment failed: {e}")
        return {"enrichment_available": False}

    def enrich_hash(self, sha256: str) -> dict:
        if not self.available:
            return {"enrichment_available": False}
        try:
            intel = Intel(auth_object=self.falcon)
            response = intel.query_indicators(filter=f"type:'hash_sha256'+value:'{sha256}'")
            if response["status_code"] == 200:
                results = response["body"].get("resources", [])
                return {
                    "enrichment_available": True, "source": "CrowdStrike Intel",
                    "sha256": sha256, "known_malicious": len(results) > 0,
                }
        except Exception as e:
            logger.warning(f"CrowdStrike hash enrichment failed: {e}")
        return {"enrichment_available": False}
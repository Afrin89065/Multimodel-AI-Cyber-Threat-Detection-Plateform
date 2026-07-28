"""AIDTECT v3.0 — VirusTotal enrichment. Set VIRUSTOTAL_API_KEY in .env"""
import hashlib
import os
from loguru import logger

try:
    import vt
    VT_AVAILABLE = True
except ImportError:
    VT_AVAILABLE = False
    logger.warning("vt-py not installed. Run: pip install vt-py")


class VirusTotalService:
    def __init__(self):
        self.api_key = os.getenv("VIRUSTOTAL_API_KEY", "")
        self.available = bool(self.api_key) and VT_AVAILABLE
        if self.available:
            logger.info("VirusTotal service ready")

    async def check_file_hash(self, file_bytes: bytes) -> dict:
        if not self.available:
            return {"vt_available": False}
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        async with vt.Client(self.api_key) as client:
            try:
                file_obj = await client.get_object_async(f"/files/{sha256}")
                stats = file_obj.last_analysis_stats
                total = sum(stats.values())
                malicious = stats.get("malicious", 0)
                return {
                    "vt_available": True, "sha256": sha256,
                    "vt_malicious": malicious, "vt_total_engines": total,
                    "vt_detection_rate": round(malicious / total, 4) if total else 0,
                    "vt_verdict": "MALWARE" if malicious > 5 else "SUSPICIOUS" if malicious > 2 else "CLEAN",
                    "vt_url": f"https://www.virustotal.com/gui/file/{sha256}",
                }
            except Exception as e:
                return {"vt_available": False, "error": str(e)}

    async def check_url(self, url: str) -> dict:
        if not self.available:
            return {"vt_available": False}
        import base64
        async with vt.Client(self.api_key) as client:
            try:
                url_id = base64.urlsafe_b64encode(url.encode()).decode().rstrip("=")
                url_obj = await client.get_object_async(f"/urls/{url_id}")
                stats = url_obj.last_analysis_stats
                total = sum(stats.values())
                malicious = stats.get("malicious", 0)
                return {
                    "vt_available": True, "url": url,
                    "vt_malicious": malicious, "vt_total": total,
                    "vt_verdict": "PHISHING" if malicious > 3 else "CLEAN",
                }
            except Exception as e:
                return {"vt_available": False, "error": str(e)}
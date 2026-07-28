"""
Live threat intelligence feeds from URLhaus and MalwareBazaar.
PATH: backend/services/threat_intel_service.py
"""
import os
import json
import hashlib
from pathlib import Path
from loguru import logger

try:
    import httpx
    HTTPX_OK = True
except ImportError:
    HTTPX_OK = False


class ThreatIntelService:
    STORE = Path("models_store/threat_intel")
    URL_FILE = STORE / "malicious_urls.json"
    HASH_FILE = STORE / "malware_hashes.json"

    def __init__(self):
        self.STORE.mkdir(parents=True, exist_ok=True)
        self.malicious_urls: set = set()
        self.malware_hashes: set = set()
        self._load()

    def _load(self):
        if self.URL_FILE.exists():
            try:
                self.malicious_urls = set(json.loads(self.URL_FILE.read_text()))
            except Exception:
                self.malicious_urls = set()
        if self.HASH_FILE.exists():
            try:
                self.malware_hashes = set(json.loads(self.HASH_FILE.read_text()))
            except Exception:
                self.malware_hashes = set()
        logger.info(f"Threat intel loaded: {len(self.malicious_urls)} URLs, {len(self.malware_hashes)} hashes")

    def check_url(self, url: str) -> dict:
        url_clean = url.lower().strip().rstrip("/")
        is_known = any(bad in url_clean for bad in self.malicious_urls)
        return {"is_known_malicious": is_known, "url": url, "source": "URLhaus" if is_known else None}

    def check_hash(self, sha256: str) -> dict:
        is_known = sha256.lower() in self.malware_hashes
        return {"is_known_malware": is_known, "sha256": sha256, "source": "MalwareBazaar" if is_known else None}

    async def update(self):
        if not HTTPX_OK:
            logger.warning("httpx not installed — cannot update threat intel")
            return
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # URLhaus recent URLs
                resp = await client.post("https://urlhaus-api.abuse.ch/v1/urls/recent/", data={"limit": 10000})
                if resp.status_code == 200:
                    data = resp.json()
                    urls = {u["url"].lower() for u in data.get("urls", []) if u.get("url")}
                    self.malicious_urls.update(urls)
                    self.URL_FILE.write_text(json.dumps(list(self.malicious_urls)))
                    logger.info(f"URLhaus updated: {len(self.malicious_urls)} total URLs")

                # MalwareBazaar recent hashes
                resp2 = await client.post("https://mb-api.abuse.ch/api/v1/", data={"query": "get_recent", "selector": "100"})
                if resp2.status_code == 200:
                    data2 = resp2.json()
                    hashes = {s["sha256_hash"].lower() for s in data2.get("data", []) if s.get("sha256_hash")}
                    self.malware_hashes.update(hashes)
                    self.HASH_FILE.write_text(json.dumps(list(self.malware_hashes)))
                    logger.info(f"MalwareBazaar updated: {len(self.malware_hashes)} total hashes")
        except Exception as e:
            logger.error(f"Threat intel update failed: {e}")
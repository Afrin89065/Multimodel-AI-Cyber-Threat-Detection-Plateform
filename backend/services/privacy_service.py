"""
PII anonymisation for audit logs (EU AI Act Article 10 compliance).
PATH: backend/services/privacy_service.py
"""
import re
from loguru import logger


class PrivacyService:
    EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
    PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")

    def __init__(self, anonymise_logs: bool = False):
        self.anonymise = anonymise_logs
        if anonymise_logs:
            logger.info("Privacy service: PII anonymisation ENABLED")

    def anonymise_text(self, text: str) -> str:
        if not self.anonymise or not text:
            return text
        text = self.EMAIL_RE.sub("[EMAIL]", text)
        text = self.IP_RE.sub("[IP]", text)
        text = self.PHONE_RE.sub("[PHONE]", text)
        return text

    def anonymise_dict(self, data: dict) -> dict:
        if not self.anonymise or not data:
            return data
        result = {}
        for k, v in data.items():
            if isinstance(v, str):
                result[k] = self.anonymise_text(v)
            elif isinstance(v, dict):
                result[k] = self.anonymise_dict(v)
            else:
                result[k] = v
        return result
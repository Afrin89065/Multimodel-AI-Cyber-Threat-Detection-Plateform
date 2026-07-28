"""
Sends email/Slack alerts for HIGH and CRITICAL detections.
PATH: backend/services/notification_service.py
"""
import os
import asyncio
from loguru import logger


class NotificationService:
    def __init__(self):
        self.smtp_email = os.getenv("SMTP_EMAIL", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.alert_email = os.getenv("ALERT_EMAIL", "")
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "")
        self.enabled = bool(self.smtp_email or self.slack_webhook)
        if self.enabled:
            logger.info("Notification service ready")

    async def notify(self, fusion_result: dict, request_id: str):
        if not self.enabled:
            return
        severity = fusion_result.get("severity", "LOW")
        risk = fusion_result.get("risk_score", 0)
        reason = fusion_result.get("reason", "")
        message = f"[AIDTECT] {severity} threat detected | Risk: {risk:.0%} | {reason} | ID: {request_id}"

        if self.slack_webhook:
            await self._send_slack(message, severity)

        if self.smtp_email and self.alert_email:
            await self._send_email(message, severity)

    async def _send_slack(self, message: str, severity: str):
        try:
            import httpx
            colors = {"CRITICAL": "#FF0000", "HIGH": "#FF6600", "MEDIUM": "#FFAA00"}
            payload = {"attachments": [{"color": colors.get(severity, "#888"), "text": message}]}
            async with httpx.AsyncClient() as client:
                await client.post(self.slack_webhook, json=payload, timeout=5)
        except Exception as e:
            logger.warning(f"Slack notification failed: {e}")

    async def _send_email(self, message: str, severity: str):
        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(message)
            msg["Subject"] = f"[AIDTECT {severity}] Threat Alert"
            msg["From"] = self.smtp_email
            msg["To"] = self.alert_email
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
                s.login(self.smtp_email, self.smtp_password)
                s.send_message(msg)
        except Exception as e:
            logger.warning(f"Email notification failed: {e}")
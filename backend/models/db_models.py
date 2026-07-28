from sqlalchemy import Column, String, Float, DateTime, JSON, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from core.database import Base
import uuid

class ThreatEvent(Base):
    __tablename__ = "threat_events"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    severity = Column(String(10), index=True)
    risk_score = Column(Float)
    threat_class = Column(String(50))
    reason = Column(Text)
    nlp_score = Column(Float)
    cv_score = Column(Float)
    network_score = Column(Float)
    malware_score = Column(Float)
    nlp_uncertainty = Column(Float)
    cv_uncertainty = Column(Float)
    network_uncertainty = Column(Float)
    malware_uncertainty = Column(Float)
    nlp_result = Column(JSON)
    vision_result = Column(JSON)
    network_result = Column(JSON)
    malware_result = Column(JSON)
    fusion_result = Column(JSON)
    shap_values = Column(JSON)
    counterfactual = Column(JSON)
    analyst_id = Column(String(100))
    analyst_verdict = Column(String(20))
    analyst_notes = Column(Text)
    is_false_positive = Column(Boolean, default=False)
    needs_human_review = Column(Boolean, default=False)
    source_ip = Column(String(50))
    request_id = Column(String(100), unique=True)

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, index=True)
    email = Column(String(200), unique=True)
    hashed_password = Column(String(200))
    role = Column(String(20), default="analyst")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
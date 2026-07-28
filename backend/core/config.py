from pydantic_settings import BaseSettings
from pathlib import Path
from functools import lru_cache
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────
    APP_NAME: str = "AIDTECT"
    APP_VERSION: str = "3.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # ── Security ─────────────────────────────────────────────────
    SECRET_KEY: str = "CHANGE-THIS-TO-256-BIT-RANDOM-STRING"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]

    # ── Database ─────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://aidtect:aidtect@localhost:5432/aidtect_db"
    DATABASE_URL_SYNC: str = "postgresql://aidtect:aidtect@localhost:5432/aidtect_db"
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # ── Redis ────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL: int = 3600
    RATE_LIMIT_PER_MINUTE: int = 60

    # ── Model Paths ──────────────────────────────────────────────
    NLP_MODEL_PATH: str = str(BASE_DIR / "models_store/nlp/distilbert_v2.pt")
    VISION_CLF_PATH: str = str(BASE_DIR / "models_store/vision/mobilenetv3_v2.pt")
    YOLO_PATH: str = str(BASE_DIR / "models_store/vision/yolov8n_logos_v2.pt")
    PHASH_DB_PATH: str = str(BASE_DIR / "models_store/vision/phash_db.pkl")
    NETWORK_XGB_PATH: str = str(BASE_DIR / "models_store/network/xgb_nids_v2.pkl")
    NETWORK_ISO_PATH: str = str(BASE_DIR / "models_store/network/isolation_forest_v2.pkl")
    NETWORK_SCALER_PATH: str = str(BASE_DIR / "models_store/network/scaler_v2.pkl")
    NETWORK_FEATURES_PATH: str = str(BASE_DIR / "models_store/network/feature_cols.txt")
    MALWARE_CNN_PATH: str = str(BASE_DIR / "models_store/malware/resnet18_v2.pt")
    MALWARE_XGB_PATH: str = str(BASE_DIR / "models_store/malware/xgb_pe_v2.pkl")
    YARA_PATH: str = str(BASE_DIR / "models_store/malware/rules.yar")
    FUSION_PATH: str = str(BASE_DIR / "models_store/fusion/mlp_fusion_v2.pt")
    FUSION_CALIBRATOR_PATH: str = str(BASE_DIR / "models_store/fusion/calibrator_v2.pkl")

    # ── Drift Detection ──────────────────────────────────────────
    DRIFT_REFERENCE_PATH: str = str(BASE_DIR / "models_store/network/drift_reference.pkl")
    DRIFT_PSI_THRESHOLD: float = 0.2
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "aidtect123"
    CROWDSTRIKE_CLIENT_ID: str = ""
    CROWDSTRIKE_CLIENT_SECRET: str = ""
    VIRUSTOTAL_API_KEY: str = ""
    ANONYMISE_LOGS: bool = False

    # ── Thresholds ───────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = 0.60
    HIGH_ALERT_THRESHOLD: float = 0.85
    UNCERTAINTY_THRESHOLD: float = 0.15

    # ── MLflow ───────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_PREFIX: str = "AIDTECT_v3"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
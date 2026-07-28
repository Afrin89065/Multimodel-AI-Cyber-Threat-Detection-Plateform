import sys, os, importlib
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_client import make_asgi_app
from loguru import logger
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import settings
from core.database import init_db
from core.redis_client import get_redis, close_redis
from api.routes import auth, nlp, vision, network, malware, fusion, explain, dashboard, health
from api.middleware.rate_limiter import RateLimitMiddleware

logger.remove()
logger.add(sys.stdout, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logs_dir = PROJECT_ROOT / "logs"
logs_dir.mkdir(exist_ok=True)
logger.add(str(logs_dir / "aidtect.log"), rotation="100 MB",
           retention="30 days", level="DEBUG", compression="gz")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 55)
    logger.info("AIDTECT v3.0 starting up...")
    logger.info("=" * 55)

    await init_db()
    redis = await get_redis()
    app.state.redis = redis
    logger.info("✅ PostgreSQL + Redis connected")

    logger.info("Loading AI models — please wait 30-60 seconds...")
    from services.nlp_service import NLPService
    from services.vision_service import VisionService
    from services.network_service import NetworkService
    from services.malware_service import MalwareService
    from services.fusion_service import FusionService
    from services.drift_service import DriftService
    from services.graceful_degradation import wrap_all_models

    raw_models = {
        "nlp":     NLPService(settings.NLP_MODEL_PATH),
        "vision":  VisionService(settings.VISION_CLF_PATH, settings.YOLO_PATH, settings.PHASH_DB_PATH),
        "network": NetworkService(settings.NETWORK_XGB_PATH, settings.NETWORK_ISO_PATH,
                                  settings.NETWORK_SCALER_PATH, settings.NETWORK_FEATURES_PATH),
        "malware": MalwareService(settings.MALWARE_CNN_PATH, settings.MALWARE_XGB_PATH, settings.YARA_PATH),
        "fusion":  FusionService(settings.FUSION_PATH, settings.FUSION_CALIBRATOR_PATH),
        "drift":   DriftService(settings.DRIFT_REFERENCE_PATH),
    }
    app.state.models = wrap_all_models(raw_models)
    logger.info("✅ AI models loaded (SecureBERT + CLIP + TabTransformer + MalBERT + AttentionFusion)")

    # ATT&CK mapper
    try:
        from services.attack_mapper import ATTACKMapper
        app.state.attack_mapper = ATTACKMapper()
        logger.info("✅ MITRE ATT&CK mapper ready")
    except Exception as e:
        app.state.attack_mapper = None
        logger.warning(f"ATT&CK mapper unavailable: {e}")

    # Neo4j Knowledge Graph
    try:
        from services.knowledge_graph import ThreatKnowledgeGraph
        app.state.knowledge_graph = ThreatKnowledgeGraph(
            uri=settings.NEO4J_URI, user=settings.NEO4J_USER, password=settings.NEO4J_PASSWORD)
        logger.info("✅ Knowledge graph (Neo4j) ready")
    except Exception as e:
        app.state.knowledge_graph = None
        logger.warning(f"Knowledge graph unavailable: {e}")

    # EU AI Act Compliance
    try:
        from services.eu_ai_act_compliance import EUAIActComplianceService
        app.state.compliance = EUAIActComplianceService()
        logger.info("✅ EU AI Act compliance ready")
    except Exception as e:
        app.state.compliance = None

    # STIX 2.1
    try:
        from services.stix_service import STIXService
        app.state.stix = STIXService()
        logger.info("✅ STIX 2.1 ready")
    except Exception as e:
        app.state.stix = None

    # VirusTotal
    try:
        from services.virustotal_service import VirusTotalService
        app.state.virustotal = VirusTotalService()
        logger.info(f"✅ VirusTotal: {'ready' if app.state.virustotal.available else 'no API key'}")
    except Exception as e:
        app.state.virustotal = None

    # CrowdStrike
    try:
        from integrations.crowdstrike_connector import CrowdStrikeConnector
        app.state.crowdstrike = CrowdStrikeConnector()
        logger.info(f"✅ CrowdStrike: {'ready' if app.state.crowdstrike.available else 'no credentials'}")
    except Exception as e:
        app.state.crowdstrike = None

    # Threat intel
    try:
        from services.threat_intel_service import ThreatIntelService
        ti = ThreatIntelService()
        app.state.threat_intel = ti
        logger.info(f"✅ Threat intel: {len(ti.malicious_urls)} URLs, {len(ti.malware_hashes)} hashes")
    except Exception as e:
        app.state.threat_intel = None

    # Screenshot
    try:
        from services.screenshot_service import ScreenshotService
        app.state.screenshot_svc = ScreenshotService()
        logger.info("✅ Screenshot service ready (Playwright)")
    except Exception as e:
        app.state.screenshot_svc = None

    # Privacy + notifications + queue
    from services.privacy_service import PrivacyService
    from services.notification_service import NotificationService
    from services.batch_queue_service import BatchQueueService

    app.state.privacy = PrivacyService(anonymise_logs=settings.ANONYMISE_LOGS)
    app.state.notifications = NotificationService()
    queue = BatchQueueService(redis)
    app.state.queue = queue
    import asyncio
    app.state.queue_task = asyncio.create_task(queue.worker_loop(app.state.models))

    logger.info("=" * 55)
    logger.info("AIDTECT v3.0 fully operational ✅")
    logger.info("Dashboard : http://localhost:3000")
    logger.info("API Docs  : http://localhost:8000/docs")
    logger.info("MLflow    : http://localhost:5000")
    logger.info("Neo4j     : http://localhost:7474")
    logger.info("=" * 55)
    yield

    logger.info("Shutting down...")
    if hasattr(app.state, "queue_task"):
        app.state.queue_task.cancel()
    if getattr(app.state, "screenshot_svc", None):
        await app.state.screenshot_svc.close()
    if getattr(app.state, "knowledge_graph", None):
        app.state.knowledge_graph.close()
    await close_redis()


app = FastAPI(title="AIDTECT v3.0 API", version="3.0.0", lifespan=lifespan, docs_url="/docs")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(CORSMiddleware, allow_origins=settings.ALLOWED_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(RateLimitMiddleware)

PREFIX = "/api/v1"
for mod in [auth, nlp, vision, network, malware, fusion, explain, dashboard, health]:
    app.include_router(mod.router, prefix=PREFIX)

for route_name in ["queue", "intelligence"]:
    try:
        m = importlib.import_module(f"api.routes.{route_name}")
        app.include_router(m.router, prefix=PREFIX)
    except ImportError:
        pass

app.mount("/metrics", make_asgi_app())
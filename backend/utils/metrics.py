from prometheus_client import Counter, Histogram, Gauge
import time

INFERENCE_TOTAL = Counter(
    "aidtect_inferences_total", "Total inferences", ["module", "status"]
)
THREAT_DETECTIONS = Counter(
    "aidtect_threats_detected_total", "Threats detected", ["severity", "threat_class"]
)
FALSE_POSITIVES = Counter(
    "aidtect_false_positives_total", "False positives", ["module"]
)
INFERENCE_LATENCY = Histogram(
    "aidtect_inference_latency_seconds", "Inference latency", ["module"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
MODEL_CONFIDENCE = Gauge(
    "aidtect_model_confidence", "Model confidence", ["module"]
)
DRIFT_SCORE = Gauge(
    "aidtect_drift_psi_score", "PSI drift score", ["module"]
)

class timer:
    """Context manager to time a block and record to Prometheus."""
    def __init__(self, module: str):
        self.module = module
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        INFERENCE_LATENCY.labels(module=self.module).observe(
            time.time() - self.start
        )
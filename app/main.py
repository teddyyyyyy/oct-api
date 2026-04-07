import os
import time
from fastapi import FastAPI, UploadFile, File
from loguru import logger

from app.schemas import PredictResponse
from app.inference.model import OCTClassifier
from llm.report import generate_report
from llm.monitoring.metrics import REQUESTS, LATENCY, PRED_CLASS, setup_metrics
from llm.monitoring.drift import DriftMonitor

# --- config (traditional + simple) ---
WEIGHTS_PATH = os.getenv("OCT_WEIGHTS_PATH", "weights/best_convnext_model_clean.pth")
ARCH = os.getenv("OCT_ARCH", "convnext_tiny")
DEVICE = os.getenv("OCT_DEVICE", "cpu")
SKIP_MODEL_LOAD = os.getenv("OCT_SKIP_MODEL_LOAD", "0") == "1"
REF_STATS_PATH = os.getenv("OCT_REF_STATS_PATH", "app/monitoring/ref_stats.npz")

app = FastAPI(title="OCT AI Diagnosis API", version="0.1.0")
setup_metrics(app)

class DummyClassifier:
    """Used for tests/CI when weights are unavailable."""
    def predict(self, img_bytes: bytes):
        return {
            "label": "NORMAL",
            "probs": {"CNV": 0.0, "DME": 0.0, "DRUSEN": 0.0, "NORMAL": 1.0},
        }

@app.on_event("startup")
def startup():
    if SKIP_MODEL_LOAD:
        logger.warning("OCT_SKIP_MODEL_LOAD=1 -> using DummyClassifier (no weights needed)")
        app.state.classifier = DummyClassifier()
    else:
        app.state.classifier = OCTClassifier(
            weights_path=WEIGHTS_PATH,
            arch=ARCH,
            device=DEVICE,
        )

    # drift monitor can also be skipped if you want; here we keep it
    app.state.drift = DriftMonitor(ref_stats_path=REF_STATS_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    start = time.time()
    REQUESTS.inc()

    img_bytes = await file.read()

    pred = app.state.classifier.predict(img_bytes)
    PRED_CLASS.labels(pred["label"]).inc()

    drift_result = app.state.drift.check_and_update(img_bytes)

    report = generate_report(
        label=pred["label"],
        probs=pred["probs"],
        drift=drift_result,
    )

    latency = time.time() - start
    LATENCY.observe(latency)

    logger.info(
        f"predict label={pred['label']} latency={latency:.3f}s drift={drift_result.get('alert')}"
    )

    return {
        "label": pred["label"],
        "probs": pred["probs"],
        "report": report,
        "drift": drift_result,
    }
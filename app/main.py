import os
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles      
from fastapi.responses import FileResponse       
from loguru import logger

from app.schemas import PredictResponse
from app.inference.model import OCTClassifier
from llm.report import generate_report
from llm.report import generate_report_with_fallback
from llm.monitoring.metrics import REQUESTS, LATENCY, PRED_CLASS, setup_metrics
from llm.monitoring.drift import DriftMonitor

# --- config (traditional + simple) ---
OLLAMA_VERSION = os.getenv("OLLAMA_VERSION", "0.20.4")
WEIGHTS_PATH = os.getenv("OCT_WEIGHTS_PATH", "weights/best_convnext_model_clean.pth")
ARCH = os.getenv("OCT_ARCH", "convnext_tiny")
DEVICE = os.getenv("OCT_DEVICE", "cpu")
SKIP_MODEL_LOAD = os.getenv("OCT_SKIP_MODEL_LOAD", "0") == "1"
REF_STATS_PATH = os.getenv("OCT_REF_STATS_PATH", "app/monitoring/ref_stats.npz")



app = FastAPI(title="OCT AI Diagnosis API", version="0.1.0")
setup_metrics(app)


# Mounting static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Homepage routing
@app.get("/")
def serve_index():
    return FileResponse("app/static/index.html")
# ====================================

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
    return {
        "status": "ok",
        "ollama_version": OLLAMA_VERSION,
    }
@app.get("/health/llm")
def health_llm():
    import json
    import urllib.request

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")

    payload = {
        "model": model,
        "prompt": "Reply with exactly: ok",
        "stream": False,
    }

    req = urllib.request.Request(
        url=f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {
            "status": "ok",
            "ollama_model": model,
            "ollama_response": data.get("response", "").strip(),
        }
    except Exception as e:
        return {
            "status": "error",
            "ollama_model": model,
            "detail": str(e),
        }

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    REQUESTS.inc()

    # 1. read file
    img_bytes = await file.read()
    t1 = time.perf_counter()

    # 2. model inference
    pred = app.state.classifier.predict(img_bytes)
    PRED_CLASS.labels(pred["label"]).inc()
    t2 = time.perf_counter()

    # 3. drift check
    drift_result = app.state.drift.check_and_update(img_bytes)
    t3 = time.perf_counter()

    # 4. report generation
    report, report_source = generate_report_with_fallback(
        pred["label"],
        pred["probs"],
        drift_result,
    )
    t4 = time.perf_counter()

    # total latency
    total_latency = t4 - t0
    file_read_time = t1 - t0
    inference_time = t2 - t1
    drift_time = t3 - t2
    report_time = t4 - t3

    LATENCY.observe(total_latency)

    logger.info(
        f"predict label={pred['label']} "
        f"total={total_latency:.3f}s "
        f"read={file_read_time:.3f}s "
        f"infer={inference_time:.3f}s "
        f"drift={drift_time:.3f}s "
        f"report={report_time:.3f}s "
        f"drift_alert={drift_result.get('alert')}"
    )

    return {
        "label": pred["label"],
        "probs": pred["probs"],
        "report": report,
        "report_source": report_source,
        "drift": drift_result,
        "ollama_version": OLLAMA_VERSION,
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3"),
        "timing": {
            "total_latency_seconds": round(total_latency, 4),
            "file_read_seconds": round(file_read_time, 4),
            "inference_seconds": round(inference_time, 4),
            "drift_seconds": round(drift_time, 4),
            "report_seconds": round(report_time, 4),
        },
    }
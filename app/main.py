import os
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles      # ← 添加這行
from fastapi.responses import FileResponse       # ← 添加這行
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

# ======== 在這裡添加以下代碼 ========
# 掛載靜態文件
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 首頁路由
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
    start = time.time()
    REQUESTS.inc()

    img_bytes = await file.read()

    pred = app.state.classifier.predict(img_bytes)
    PRED_CLASS.labels(pred["label"]).inc()

    drift_result = app.state.drift.check_and_update(img_bytes)

    report, report_source = generate_report_with_fallback(
        pred["label"],
        pred["probs"],
        drift_result,
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
        "report_source": report_source,
        "drift": drift_result,
        "ollama_version": OLLAMA_VERSION,
        "ollama_model": os.getenv("OLLAMA_MODEL", "llama3"),
    }
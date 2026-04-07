from prometheus_client import Counter, Histogram, make_asgi_app
from fastapi import FastAPI

REQUESTS = Counter("oct_requests_total", "Total requests")
PRED_CLASS = Counter("oct_pred_class_total", "Predicted class counts", ["label"])
LATENCY = Histogram("oct_latency_seconds", "Latency seconds")

def setup_metrics(app: FastAPI):
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
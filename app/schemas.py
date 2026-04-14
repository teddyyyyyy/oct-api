from pydantic import BaseModel
from typing import Dict, Optional


class TimingResponse(BaseModel):
    total_latency_seconds: float
    file_read_seconds: float
    inference_seconds: float
    drift_seconds: float
    report_seconds: float


class PredictResponse(BaseModel):
    label: str
    probs: Dict[str, float]
    report: str
    report_source: str
    drift: dict
    ollama_version: str
    ollama_model: str
    timing: TimingResponse
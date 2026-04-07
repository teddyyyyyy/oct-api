from pydantic import BaseModel
from typing import Dict, Any

class PredictResponse(BaseModel):
    label: str
    probs: Dict[str, float]   # {"CNV":0.1,"DME":0.2,"DRUSEN":0.6,"NORMAL":0.1}
    report: str
    drift: Dict[str, Any]
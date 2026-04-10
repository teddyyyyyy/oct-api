# OCT Classification API

An end-to-end Medical AI system for Optical Coherence Tomography (OCT) image classification, integrating deep learning inference, LLM-powered report generation, and real-time monitoring in a production-ready pipeline.

This project demonstrates how to design and deploy a robust machine learning system with drift awareness, API serving, and fallback-safe inference reporting.

---

## 🚀 Features

- 🔬 **Deep Learning Inference**  
  OCT image classification using ConvNeXt architecture (PyTorch)

- 🌐 **REST API (FastAPI)**  
  Production-ready inference API with Swagger documentation

- 🤖 **LLM Report Generation**  
  Automated medical-style report generation using LLM (Ollama)

- ⚠️ **Fallback Mechanism (Critical Feature)**  
  Ensures system reliability when LLM is unavailable (rule-based fallback)

- 📊 **Prometheus Monitoring**  
  Real-time tracking of inference requests, latency, and system health

- 🔍 **Data Drift Detection**  
  Monitors input data distribution shifts for deployment safety

- 🎨 **Web UI Interface**  
  Interactive frontend for image upload and prediction visualization

---

## 🧠 System Architecture

This system is designed as a modular ML pipeline:

- **Model Layer**  
  ConvNeXt-based classifier for OCT image prediction

- **API Layer**  
  FastAPI service for handling inference requests

- **LLM Layer**  
  Report generation module with fallback-safe logic

- **Monitoring Layer**  
  Prometheus metrics + drift detection for observability

- **Frontend Layer**  
  Lightweight web interface for user interaction

---

## 📊 Model Performance

- Internal Validation (OCT2017):  
  **AUROC > 0.99**

- External Validation (Leakage-Controlled Dataset):  
  **Macro-F1 ≈ 0.93**

- Demonstrates realistic generalization under distribution shift

---

## ⚙️ Tech Stack

- **Deep Learning**: PyTorch, timm (ConvNeXt)
- **Backend**: FastAPI, Uvicorn
- **Monitoring**: Prometheus
- **LLM**: Ollama (with fallback system)
- **Deployment**: Docker, Docker Compose
- **Frontend**: HTML / JS

---

## 🏗️ Project Structure
```
oct_api/
├── app/                    # FastAPI application
│   ├── main.py            # API entry point
│   ├── config.py          # Configuration
│   ├── schemas.py         # Data models
│   ├── inference/         # Model inference
│   │   ├── model.py
│   │   └── preprocess.py
│   └── static/            # Web UI
├── llm/                   # LLM + monitoring
│   ├── report.py          # Report generation
│   └── monitoring/
│       ├── metrics.py     # Performance metrics
│       └── drift.py       # Data drift detection
├── tests/                 # Test suite
├── weights/               # Model weights
├── Dockerfile
├── docker-compose.yml
└── README.md
```
---

## ⚡ Quick Start

### 🔹 Local Setup

```bash
conda activate oct_api
cd ~/teddy/oct_api

OCT_WEIGHTS_PATH=weights/best_convnext_model_clean.pth \
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Access:
- Web UI: http://localhost:8000  
- API Docs: http://localhost:8000/docs  
- Metrics: http://localhost:8000/metrics  

### 🔹 Docker Setup
```bash
docker-compose up --build
```

## 📡 API Usage
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg"
```
### Response
```bash
{
  "prediction": "DME",
  "confidence": 0.987,
  "report": "Generated medical summary..."
}
```

## 🧪 Testing
```bash
pytest -v
```
## 🔧 Configuration
```bash
OCT_WEIGHTS_PATH=weights/best_convnext_model_clean.pth
OCT_ARCH=convnext_tiny
OCT_DEVICE=cpu
```

## ⚠️ Troubleshooting

| Issue | Solution |
|------|--------|
| Model not loading | Check `OCT_WEIGHTS_PATH` |
| CUDA OOM | Use CPU or reduce batch size |
| API not responding | Check port 8000 |
| Slow inference | Enable GPU |

## 📌 Highlights (For Recruiters)
	•	Designed end-to-end ML system (not just model training)
	•	Implemented LLM fallback mechanism for robustness
	•	Integrated drift detection for real-world deployment safety
	•	Built production-ready API with monitoring and Docker

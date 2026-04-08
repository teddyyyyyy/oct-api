# OCT Classification API

An automated Optical Coherence Tomography (OCT) image classification system using ConvNeXt deep learning model. Provides REST API and Web UI with real-time inference, performance monitoring, and data drift detection.

## Features

- 🔬 **Deep Learning Inference** - OCT image classification using ConvNeXt model
- 🌐 **REST API** - FastAPI framework supporting image upload and predictions
- 📊 **Prometheus Monitoring** - Real-time performance metrics and inference statistics
- 🔍 **Drift Detection** - Monitor input data distribution changes
- 📈 **Report Generation** - Automated inference result reports
- 🎨 **Web UI** - Interactive frontend interface

## System Requirements

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- 4GB+ RAM (8GB+ recommended)
- 2GB+ available disk space

## Quick Start

### Method 1: Local Setup (Recommended)

**Prerequisites**: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed

```bash
# 1. Activate conda environment
conda activate oct_api

# 2. Navigate to project directory
cd ~/teddy/oct_api

# 3. Start API service
OCT_WEIGHTS_PATH=weights/best_convnext_model_clean.pth \
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

After startup, access:

- Web UI: http://localhost:8000
- API Documentation: http://localhost:8000/docs (Swagger UI)
- Metrics: http://localhost:8000/metrics (Prometheus)

### Method 2: Docker Setup
```bash
docker-compose up
```

## Configuration
```bash
# Model weights path (required)
OCT_WEIGHTS_PATH=weights/best_convnext_model_clean.pth

# Model architecture (optional, default: convnext_tiny)
OCT_ARCH=convnext_tiny

# Compute device (optional, default: cpu, options: cuda)
OCT_DEVICE=cpu
```

## API Usage
Image Classification
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```
Response Example:
```json
{
  "prediction": "healthy",
  "confidence": 0.95,
  "timestamp": "2026-04-08T10:30:45Z"
}


## Testing 
Run All Tests
```bash
conda run -n oct_api python -m -pytest -q
```

## Run Specific Tests
```bash
# Unit tests
conda run -n oct_api python -m pytest tests/test_report.py -v

# Integration tests
conda run -n oct_api python -m pytest tests/test_integration.py -v

# Drift detection tests
conda run -n oct_api python -m pytest tests/test_drift.py -v
```

## Project Structure
```bash
oct_api/
├── app/                    # FastAPI application
│   ├── main.py            # Application entry point, API routes
│   ├── config.py          # Configuration management
│   ├── schemas.py         # Request/response data models
│   ├── inference/         # Inference module
│   │   ├── model.py      # Model loading and inference
│   │   └── preprocess.py # Image preprocessing
│   └── static/            # Frontend assets
│       ├── index.html    # Web UI
│       ├── script.js     # Interactive logic
│       └── style.css     # Styling
├── llm/                   # Report and monitoring module
│   ├── report.py         # Report generation
│   └── monitoring/       # Monitoring and drift detection
│       ├── metrics.py   # Performance metrics collection
│       └── drift.py     # Data drift detection
├── scripts/              # Utility scripts
│   └── generate_ref_stats.py  # Generate reference statistics
├── tests/               # Test cases
│   ├── conftest.py     # Pytest configuration
│   ├── test_integration.py
│   ├── test_drift.py
│   └── test_report.py
├── weights/             # Pre-trained model weights
│   └── best_convnext_model_clean.pth
├── requirements.txt     # Python dependencies
├── docker-compose.yml  # Docker configuration
├── Dockerfile          # Docker image definition
└── README.md           # This file
```

## Dependency Installation
If the conda env hasn't been created yet:
```bash
# Create new environment
conda create -n oct_api python=3.10

# Activate environment
conda activate oct_api

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Or install CUDA version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Development

Code Style
Follow PEP 8 standards. Recommended tools: black and pylint.

Run Development Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Troubleshooting
| Issue                     | Solution                                                                 |
|--------------------------|--------------------------------------------------------------------------|
| Model weights not found  | Check `OCT_WEIGHTS_PATH` environment variable is set correctly          |
| CUDA out of memory       | Set `OCT_DEVICE` to `cpu` or reduce batch size                          |
| Slow inference           | Ensure GPU is used (if available), check system resources               |
| API connection failed    | Verify service is running, check firewall and port 8000                 |


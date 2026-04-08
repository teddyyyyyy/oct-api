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


FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY app /app/app
COPY llm /app/llm
COPY weights /app/weights
COPY scripts /app/scripts

EXPOSE 8000

ENV OCT_WEIGHTS_PATH=/app/weights/best_convnext_model_clean.pth \
    OCT_ARCH=convnext_tiny \
    OCT_DEVICE=cpu

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
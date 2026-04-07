## 本地（WSL/conda）啟動
conda activate oct_api
cd ~/teddy/oct_api
OCT_WEIGHTS_PATH=weights/best_convnext_model_clean.pth \
uvicorn app.main:app --host 0.0.0.0 --port 8000

## 測試
conda run -n oct_api python -m pytest -q

## 推論測試
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/home/ser/teddy/oct_api/test_image.jpg"

## Metrics 
http://localhost:8000/metrics

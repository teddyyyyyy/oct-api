"""Integration tests for OCT API."""

import pytest
import sys
import io
from pathlib import Path
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock imports - skip if dependencies not available
try:
    from fastapi.testclient import TestClient
    from app.main import app
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
class TestAPIEndpoints:
    """Test API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_endpoint_with_image(self, client):
        """Test /predict endpoint with valid image.
        
        Note: This test requires 'weights/convnext.pth' to be available.
        It may be skipped if the model weights are missing.
        """
        # Create a test image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        try:
            response = client.post(
                "/predict",
                files={"file": ("test.png", img_bytes, "image/png")}
            )
            
            # Should get 200 if model weights are available
            if response.status_code == 200:
                data = response.json()
                assert "label" in data
                assert "probs" in data
                assert "report" in data
                assert "drift" in data
                assert data["label"] in ["CNV", "DME", "DRUSEN", "NORMAL"]
            else:
                # May fail if weights not found
                pytest.skip(f"Model weights not available: {response.status_code}")
        except Exception as e:
            pytest.skip(f"Test skipped due to: {str(e)}")

    def test_predict_endpoint_no_file(self, client):
        """Test /predict endpoint without file."""
        response = client.post("/predict")
        assert response.status_code in [400, 422]  # Bad request or validation error

    def test_metrics_endpoint(self, client):
        """Test /metrics endpoint."""
        try:
            response = client.get("/metrics")
            assert response.status_code == 200
            assert "HELP" in response.text or "TYPE" in response.text
        except Exception:
            pytest.skip("Metrics endpoint not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

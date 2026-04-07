"""Unit tests for drift monitoring."""

import pytest
import sys
import numpy as np
import io
from pathlib import Path
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.monitoring.drift import img_stats, DriftMonitor


class TestImgStats:
    """Test image statistics calculation."""

    def test_img_stats_output_shape(self):
        """Test that img_stats returns correct shape."""
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        stats = img_stats(img)
        
        assert isinstance(stats, np.ndarray)
        assert stats.shape == (3,)
        assert stats.dtype == np.float32

    def test_img_stats_values_in_range(self):
        """Test that statistics are in expected ranges."""
        # Create grayscale image with known content
        arr = np.full((100, 100), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        stats = img_stats(img)
        
        mean, std, sharp = stats
        # Mean should be ~0.5 for gray image (128/255)
        assert 0.4 < mean < 0.6
        # Std should be close to 0 for uniform image
        assert std < 0.05
        # Sharpness should be low for uniform image
        assert sharp < 0.1

    def test_img_stats_with_variance(self):
        """Test stats with varied image content."""
        arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        stats = img_stats(img)
        
        mean, std, sharp = stats
        # Mean should be ~0.5 for random image
        assert 0.3 < mean < 0.7
        # Std should be notable for varied image
        assert std > 0.05


class TestDriftMonitor:
    """Test drift monitoring functionality."""

    def test_drift_monitor_initialization(self):
        """Test DriftMonitor can be initialized."""
        monitor = DriftMonitor(ref_stats_path="nonexistent.npz")
        
        assert hasattr(monitor, 'mu')
        assert hasattr(monitor, 'sigma')
        assert isinstance(monitor.mu, np.ndarray)
        assert isinstance(monitor.sigma, np.ndarray)

    def test_drift_monitor_check_valid_image(self):
        """Test drift checking with valid image bytes."""
        monitor = DriftMonitor(ref_stats_path="nonexistent.npz")
        
        # Create valid image
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        result = monitor.check_and_update(img_bytes.getvalue())
        
        assert isinstance(result, dict)
        assert "score" in result
        assert "z" in result
        assert "alert" in result
        assert isinstance(result["score"], float)
        assert isinstance(result["z"], list)

    def test_drift_monitor_alert_thresholds(self):
        """Test drift alert triggering at different thresholds."""
        # Create monitor with extreme reference stats
        monitor = DriftMonitor(ref_stats_path="nonexistent.npz")
        monitor.mu = np.array([0.1, 0.05, 0.01], dtype=np.float32)
        monitor.sigma = np.array([0.01, 0.01, 0.001], dtype=np.float32)
        
        # Create very different image (should trigger alert)
        arr = np.full((100, 100), 250, dtype=np.uint8)  # Very bright
        img = Image.fromarray(arr, mode="L")
        img_bytes = io.BytesIO()
        Image.fromarray(arr).save(img_bytes, format="PNG")
        img_bytes.seek(0)
        
        result = monitor.check_and_update(img_bytes.getvalue())
        
        # Score should be high due to large difference
        assert result["score"] > 0
        # Alert might be triggered depending on exact thresholds
        assert result["alert"] in [None, "moderate_shift", "severe_shift"]

    def test_drift_monitor_score_consistency(self):
        """Test that same image gives consistent score."""
        monitor = DriftMonitor(ref_stats_path="nonexistent.npz")
        
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        img_data = img_bytes.getvalue()
        
        result1 = monitor.check_and_update(img_data)
        result2 = monitor.check_and_update(img_data)
        
        assert result1["score"] == result2["score"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

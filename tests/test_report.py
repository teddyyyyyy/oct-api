git branch"""Unit tests for report generation."""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.report import generate_report


class TestGenerateReport:
    """Test report generation with various inputs."""

    def test_valid_report_generation(self):
        """Test basic report generation with valid inputs."""
        label = "CNV"
        probs = {"CNV": 0.8, "DME": 0.1, "DRUSEN": 0.05, "NORMAL": 0.05}
        drift = {"alert": None, "score": 2.5}
        
        report = generate_report(label, probs, drift)
        
        assert isinstance(report, str)
        assert "CNV" in report
        assert "Confidence ranking:" in report
        assert "OCT AI Preliminary Report" in report

    def test_report_with_drift_alert(self):
        """Test report generation with drift alert."""
        label = "DME"
        probs = {"DME": 0.7, "CNV": 0.15, "DRUSEN": 0.1, "NORMAL": 0.05}
        drift = {"alert": "severe_shift", "score": 5.5}
        
        report = generate_report(label, probs, drift)
        
        assert "severe_shift" in report
        assert "Data shift alert" in report
        assert "5.50" in report

    def test_report_with_empty_probs(self):
        """Test report handles empty probability dict."""
        label = "NORMAL"
        probs = {}
        drift = {}
        
        report = generate_report(label, probs, drift)
        
        assert isinstance(report, str)
        assert "ERROR" in report or "Invalid" in report

    def test_report_with_none_probs(self):
        """Test report handles None probability."""
        label = "NORMAL"
        probs = None
        drift = {}
        
        report = generate_report(label, probs, drift)
        
        assert isinstance(report, str)
        assert "ERROR" in report or "Invalid" in report

    def test_report_with_single_class(self):
        """Test report with only one class probability."""
        label = "NORMAL"
        probs = {"NORMAL": 1.0}
        drift = {"alert": None}
        
        report = generate_report(label, probs, drift)
        
        assert isinstance(report, str)
        assert "NORMAL" in report
        assert "1.000" in report

    def test_report_probability_formatting(self):
        """Test probability values are correctly formatted."""
        label = "CNV"
        probs = {"CNV": 0.345678, "DME": 0.234567, "DRUSEN": 0.296299, "NORMAL": 0.123456}
        drift = {}
        
        report = generate_report(label, probs, drift)
        
        assert "0.346" in report  # Should be 3 decimals (top 1)
        assert "0.235" in report  # top 2
        assert "0.296" in report  # top 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

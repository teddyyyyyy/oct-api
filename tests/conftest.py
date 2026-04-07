"""Test configuration and fixtures."""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing."""
    from PIL import Image
    import io
    
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes.seek(0)
    return img_bytes.getvalue()

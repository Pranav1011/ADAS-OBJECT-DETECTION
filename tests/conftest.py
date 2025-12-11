"""Pytest configuration and fixtures."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def data_dir(project_root):
    """Get data directory."""
    return project_root / "data"


@pytest.fixture
def weights_dir(project_root):
    """Get weights directory."""
    return project_root / "weights"

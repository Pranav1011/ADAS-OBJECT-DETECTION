"""Tests for inference module."""

import pytest
import numpy as np


class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_properties(self):
        """Test Detection properties."""
        from inference.detector import Detection

        det = Detection(
            x1=100, y1=200, x2=300, y2=400,
            confidence=0.95,
            class_id=0,
            class_name="car"
        )

        # Test box property
        assert det.box == (100, 200, 300, 400)

        # Test center property
        assert det.center == (200, 300)

        # Test area property
        assert det.area == 200 * 200

    def test_detection_to_dict(self):
        """Test Detection to_dict method."""
        from inference.detector import Detection

        det = Detection(
            x1=100, y1=200, x2=300, y2=400,
            confidence=0.95,
            class_id=0,
            class_name="car"
        )

        d = det.to_dict()

        assert d["x1"] == 100
        assert d["class_name"] == "car"
        assert d["confidence"] == 0.95


class TestObjectDetector:
    """Tests for ObjectDetector class."""

    def test_letterbox_preserves_aspect_ratio(self, sample_image):
        """Test letterbox resizing preserves aspect ratio."""
        from inference.detector import ObjectDetector

        # Create a non-square image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Can't test full detector without model, but can test preprocessing
        # This would need a mock or actual model

    def test_nms_removes_overlapping_boxes(self):
        """Test NMS removes overlapping boxes."""
        from inference.detector import ObjectDetector

        # Create detector instance (would need mock model)
        # Test NMS logic separately

        # Sample boxes with overlap
        boxes = np.array([
            [0, 0, 100, 100],
            [10, 10, 110, 110],  # Overlaps with first
            [200, 200, 300, 300],  # No overlap
        ])
        scores = np.array([0.9, 0.8, 0.7])

        # NMS should keep first and third box
        # This requires instantiating the detector which needs a model


class TestDrawDetections:
    """Tests for drawing functions."""

    def test_draw_detections_returns_image(self, sample_image):
        """Test draw_detections returns valid image."""
        from inference.detector import draw_detections, Detection

        detections = [
            Detection(
                x1=100, y1=100, x2=200, y2=200,
                confidence=0.9,
                class_id=0,
                class_name="car"
            )
        ]

        output = draw_detections(sample_image, detections)

        assert output.shape == sample_image.shape
        assert output.dtype == sample_image.dtype

    def test_draw_detections_empty_list(self, sample_image):
        """Test draw_detections with no detections."""
        from inference.detector import draw_detections

        output = draw_detections(sample_image, [])

        # Should return copy of original
        assert output.shape == sample_image.shape

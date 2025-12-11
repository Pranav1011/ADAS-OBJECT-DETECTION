"""Tests for data pipeline."""

import pytest
import numpy as np
from pathlib import Path


class TestBDD100KToYOLO:
    """Tests for annotation conversion."""

    def test_convert_box_to_yolo(self):
        """Test bounding box conversion."""
        from data_pipeline.convert_annotations import BDD100KToYOLO

        converter = BDD100KToYOLO(
            bdd_root="data/raw",
            output_dir="data/processed"
        )

        # Test box: x1=100, y1=200, x2=300, y2=400
        # Image size: 1280x720
        box2d = {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
        cx, cy, w, h = converter.convert_box_to_yolo(box2d)

        # Expected: center=(200, 300), size=(200, 200)
        # Normalized: cx=200/1280, cy=300/720, w=200/1280, h=200/720
        assert abs(cx - 200/1280) < 0.001
        assert abs(cy - 300/720) < 0.001
        assert abs(w - 200/1280) < 0.001
        assert abs(h - 200/720) < 0.001

    def test_class_mapping(self):
        """Test class mapping is correct."""
        from data_pipeline.convert_annotations import BDD100KToYOLO

        expected_classes = [
            "car", "truck", "bus", "pedestrian",
            "cyclist", "motorcycle", "traffic_light", "traffic_sign"
        ]

        assert BDD100KToYOLO.CLASS_NAMES == expected_classes
        assert len(BDD100KToYOLO.CLASS_MAPPING) > 0


class TestDataAugmentation:
    """Tests for data augmentation."""

    def test_augmentation_creates_valid_output(self, sample_image):
        """Test augmentation produces valid output."""
        from data_pipeline.data_augmentation import DataAugmentation

        aug = DataAugmentation(mode="train")

        # Sample bboxes in YOLO format
        bboxes = [[0.5, 0.5, 0.2, 0.2]]
        labels = [0]

        aug_img, aug_boxes, aug_labels = aug.augment(
            sample_image, bboxes, labels
        )

        # Check output is valid
        assert aug_img is not None
        assert isinstance(aug_boxes, (list, tuple))
        assert isinstance(aug_labels, (list, tuple))

    def test_val_augmentation_preserves_boxes(self, sample_image):
        """Test validation augmentation preserves bounding boxes."""
        from data_pipeline.data_augmentation import DataAugmentation

        aug = DataAugmentation(mode="val")

        bboxes = [[0.5, 0.5, 0.2, 0.2]]
        labels = [0]

        _, aug_boxes, aug_labels = aug.augment(
            sample_image, bboxes, labels
        )

        # Boxes should be preserved (just resized)
        assert len(aug_boxes) == len(bboxes)
        assert len(aug_labels) == len(labels)

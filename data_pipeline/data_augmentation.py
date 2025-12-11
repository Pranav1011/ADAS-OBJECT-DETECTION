"""
Data augmentation pipeline for ADAS object detection.

Uses Albumentations for robust augmentation including:
- Geometric transforms
- Color transforms
- Weather simulation (rain, fog, sun flare)
- Occlusion simulation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DataAugmentation:
    """
    Augmentation pipeline for autonomous driving images.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        mode: str = "train"  # "train", "val", or "test"
    ):
        """
        Initialize augmentation pipeline.

        Args:
            config: Optional configuration dict
            mode: Dataset mode - "train" for full augmentation,
                  "val"/"test" for resize only
        """
        self.config = config or {}
        self.mode = mode

        if mode == "train":
            self.transform = self._get_train_transform()
        else:
            self.transform = self._get_val_transform()

    def _get_train_transform(self) -> A.Compose:
        """
        Get training augmentation pipeline.

        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            # Resize to target size
            A.Resize(
                height=self.config.get("img_size", 640),
                width=self.config.get("img_size", 640)
            ),

            # Geometric transforms
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=0,  # No rotation for driving scenes
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),

            # Color transforms
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=1.0
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
            ], p=0.5),

            # Blur and noise
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=(3, 5), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.2),

            A.GaussNoise(var_limit=(10, 50), p=0.2),

            # Weather simulation (important for ADAS robustness)
            A.OneOf([
                A.RandomRain(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=20,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=5,
                    brightness_coefficient=0.7,
                    rain_type="drizzle",
                    p=1.0
                ),
                A.RandomFog(
                    fog_coef_lower=0.1,
                    fog_coef_upper=0.3,
                    alpha_coef=0.1,
                    p=1.0
                ),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),  # Upper half of image
                    angle_lower=0.5,
                    src_radius=100,
                    num_flare_circles_lower=1,
                    num_flare_circles_upper=2,
                    p=1.0
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),  # Lower half
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=5,
                    p=1.0
                ),
            ], p=0.3),

            # Occlusion simulation (random erasing)
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=0,
                p=0.2
            ),

            # Normalize and convert
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),

        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_area=100,  # Remove boxes that become too small
            min_visibility=0.3  # Remove heavily cropped boxes
        ))

    def _get_val_transform(self) -> A.Compose:
        """
        Get validation/test augmentation pipeline (resize only).

        Returns:
            Albumentations Compose object
        """
        return A.Compose([
            A.Resize(
                height=self.config.get("img_size", 640),
                width=self.config.get("img_size", 640)
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))

    def augment(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply augmentation to image and bounding boxes.

        Args:
            image: Input image (HWC, BGR or RGB)
            bboxes: List of [cx, cy, w, h] in YOLO format (0-1)
            labels: List of class indices

        Returns:
            Tuple of (augmented_image, augmented_bboxes, augmented_labels)
        """
        # Albumentations expects RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels
        )

        return (
            transformed['image'],
            transformed['bboxes'],
            transformed['class_labels']
        )

    def visualize_augmentation(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        labels: List[int],
        class_names: List[str],
        num_samples: int = 4
    ) -> np.ndarray:
        """
        Visualize multiple augmentation samples.

        Args:
            image: Original image
            bboxes: Original bounding boxes
            labels: Original class labels
            class_names: List of class names
            num_samples: Number of augmented samples to show

        Returns:
            Grid image showing original + augmented samples
        """
        samples = []

        # Original image
        orig_vis = self._draw_boxes(
            image.copy(), bboxes, labels, class_names, "Original"
        )
        samples.append(orig_vis)

        # Augmented samples
        for i in range(num_samples - 1):
            aug_img, aug_boxes, aug_labels = self.augment(
                image.copy(), bboxes.copy(), labels.copy()
            )

            # Denormalize if normalized
            if aug_img.max() <= 1.0:
                aug_img = (aug_img * 255).astype(np.uint8)

            aug_vis = self._draw_boxes(
                aug_img, aug_boxes, aug_labels, class_names, f"Aug {i+1}"
            )
            samples.append(aug_vis)

        # Create grid
        grid = self._create_grid(samples, cols=2)
        return grid

    def _draw_boxes(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        labels: List[int],
        class_names: List[str],
        title: str = ""
    ) -> np.ndarray:
        """Draw bounding boxes on image."""
        h, w = image.shape[:2]
        vis = image.copy()

        # Colors for each class
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 128, 0), (128, 0, 128)
        ]

        for bbox, label in zip(bboxes, labels):
            cx, cy, bw, bh = bbox

            # Convert to pixel coordinates
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            color = colors[label % len(colors)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label text
            text = class_names[label] if label < len(class_names) else str(label)
            cv2.putText(
                vis, text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

        # Title
        if title:
            cv2.putText(
                vis, title, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
            )

        return vis

    def _create_grid(
        self,
        images: List[np.ndarray],
        cols: int = 2
    ) -> np.ndarray:
        """Create a grid of images."""
        rows = (len(images) + cols - 1) // cols

        # Resize all to same size
        h, w = images[0].shape[:2]
        resized = [cv2.resize(img, (w, h)) for img in images]

        # Pad to fill grid
        while len(resized) < rows * cols:
            resized.append(np.zeros_like(resized[0]))

        # Create grid
        grid_rows = []
        for i in range(rows):
            row_imgs = resized[i * cols:(i + 1) * cols]
            grid_rows.append(np.hstack(row_imgs))

        grid = np.vstack(grid_rows)
        return grid


def main():
    """Demo augmentation on a sample image."""
    import argparse

    parser = argparse.ArgumentParser(description="Augmentation Demo")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Path to YOLO format label file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="augmentation_demo.jpg",
        help="Output visualization path"
    )

    args = parser.parse_args()

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return

    # Load labels if provided
    bboxes = []
    labels = []
    if args.label and Path(args.label).exists():
        with open(args.label, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:5]])

    # Create augmentation pipeline
    aug = DataAugmentation(mode="train")

    # Class names
    class_names = [
        "car", "truck", "bus", "pedestrian",
        "cyclist", "motorcycle", "traffic_light", "traffic_sign"
    ]

    # Visualize
    vis = aug.visualize_augmentation(
        image, bboxes, labels, class_names, num_samples=4
    )

    cv2.imwrite(args.output, vis)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()

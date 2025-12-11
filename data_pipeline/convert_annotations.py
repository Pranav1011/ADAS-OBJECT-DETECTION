"""
Convert BDD100K annotations to YOLO format.

BDD100K format:
{
    "name": "image.jpg",
    "labels": [
        {
            "category": "car",
            "box2d": {"x1": 100, "y1": 200, "x2": 300, "y2": 400}
        }
    ]
}

YOLO format (per line):
class_id center_x center_y width height (all normalized 0-1)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import shutil
from tqdm import tqdm
import yaml


class BDD100KToYOLO:
    """
    Convert BDD100K annotations to YOLO format.
    """

    # BDD100K class to YOLO class index mapping
    CLASS_MAPPING = {
        "car": 0,
        "truck": 1,
        "bus": 2,
        "person": 3,          # Maps to 'pedestrian'
        "rider": 4,           # Maps to 'cyclist'
        "bike": 5,            # Maps to 'motorcycle'
        "motor": 5,           # Also maps to 'motorcycle'
        "traffic light": 6,
        "traffic sign": 7
    }

    # YOLO class names (for dataset.yaml)
    CLASS_NAMES = [
        "car",
        "truck",
        "bus",
        "pedestrian",
        "cyclist",
        "motorcycle",
        "traffic_light",
        "traffic_sign"
    ]

    # BDD100K image dimensions
    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 720

    def __init__(
        self,
        bdd_root: str,
        output_dir: str,
        min_box_area: int = 400,  # Minimum 20x20 pixels
        min_visibility: float = 0.0  # Minimum visibility ratio
    ):
        """
        Initialize converter.

        Args:
            bdd_root: Root directory of extracted BDD100K dataset
            output_dir: Output directory for YOLO format data
            min_box_area: Minimum bounding box area in pixels
            min_visibility: Minimum visibility ratio (0-1)
        """
        self.bdd_root = Path(bdd_root)
        self.output_dir = Path(output_dir)
        self.min_box_area = min_box_area
        self.min_visibility = min_visibility

        # Statistics
        self.stats = {
            "total_images": 0,
            "images_with_labels": 0,
            "total_boxes": 0,
            "filtered_boxes": 0,
            "class_counts": defaultdict(int)
        }

    def convert_box_to_yolo(
        self,
        box2d: Dict,
        img_width: int = IMAGE_WIDTH,
        img_height: int = IMAGE_HEIGHT
    ) -> Tuple[float, float, float, float]:
        """
        Convert BDD100K box format to YOLO format.

        Args:
            box2d: Dict with x1, y1, x2, y2 keys
            img_width: Image width for normalization
            img_height: Image height for normalization

        Returns:
            Tuple of (center_x, center_y, width, height) normalized to 0-1
        """
        x1, y1 = box2d["x1"], box2d["y1"]
        x2, y2 = box2d["x2"], box2d["y2"]

        # Calculate center and dimensions
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        # Normalize to 0-1
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height

        # Clip to valid range
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        return center_x, center_y, width, height

    def should_include_label(self, label: Dict) -> bool:
        """
        Check if a label should be included based on filters.

        Args:
            label: BDD100K label dict

        Returns:
            True if label should be included
        """
        # Check if category is in our target classes
        category = label.get("category", "").lower()
        if category not in self.CLASS_MAPPING:
            return False

        # Check if box2d exists
        if "box2d" not in label:
            return False

        box = label["box2d"]

        # Check minimum area
        width = box["x2"] - box["x1"]
        height = box["y2"] - box["y1"]
        area = width * height

        if area < self.min_box_area:
            self.stats["filtered_boxes"] += 1
            return False

        # Check visibility/occlusion if available
        attributes = label.get("attributes", {})
        if "occluded" in attributes and attributes["occluded"]:
            # Could filter heavily occluded objects
            pass

        return True

    def convert_annotation(
        self,
        bdd_annotation: Dict
    ) -> List[str]:
        """
        Convert single BDD100K annotation to YOLO format lines.

        Args:
            bdd_annotation: BDD100K annotation dict for one image

        Returns:
            List of YOLO format strings (one per object)
        """
        yolo_lines = []

        labels = bdd_annotation.get("labels", [])

        for label in labels:
            if not self.should_include_label(label):
                continue

            category = label["category"].lower()
            class_id = self.CLASS_MAPPING[category]

            box = label["box2d"]
            cx, cy, w, h = self.convert_box_to_yolo(box)

            # Format: class_id center_x center_y width height
            yolo_line = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            yolo_lines.append(yolo_line)

            self.stats["total_boxes"] += 1
            self.stats["class_counts"][self.CLASS_NAMES[class_id]] += 1

        return yolo_lines

    def convert_split(
        self,
        split: str,
        max_images: Optional[int] = None
    ) -> int:
        """
        Convert all annotations for a dataset split.

        Args:
            split: 'train' or 'val'
            max_images: Maximum number of images to process (for testing)

        Returns:
            Number of images processed
        """
        # Find annotation file
        # Try different possible paths
        possible_paths = [
            self.bdd_root / "labels" / "det_20" / f"det_{split}.json",
            self.bdd_root / "bdd100k" / "labels" / "det_20" / f"det_{split}.json",
            self.bdd_root / "labels" / f"bdd100k_labels_images_{split}.json",
        ]

        annotation_file = None
        for path in possible_paths:
            if path.exists():
                annotation_file = path
                break

        if annotation_file is None:
            print(f"Warning: Could not find annotation file for {split}")
            print(f"Searched paths: {possible_paths}")
            return 0

        print(f"Loading annotations from {annotation_file}")

        # Load annotations
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)

        # Find image directory
        possible_img_dirs = [
            self.bdd_root / "images" / "100k" / split,
            self.bdd_root / "images" / "10k" / split,
            self.bdd_root / "bdd100k" / "images" / "100k" / split,
            self.bdd_root / "bdd100k" / "images" / "10k" / split,
        ]

        img_dir = None
        for path in possible_img_dirs:
            if path.exists():
                img_dir = path
                break

        if img_dir is None:
            print(f"Warning: Could not find image directory for {split}")
            return 0

        print(f"Using images from {img_dir}")

        # Create output directories
        out_img_dir = self.output_dir / "images" / split
        out_label_dir = self.output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        # Process annotations
        if max_images:
            annotations = annotations[:max_images]

        processed = 0
        for ann in tqdm(annotations, desc=f"Converting {split}"):
            img_name = ann["name"]
            img_path = img_dir / img_name

            if not img_path.exists():
                continue

            # Convert annotation
            yolo_lines = self.convert_annotation(ann)

            self.stats["total_images"] += 1

            if yolo_lines:
                self.stats["images_with_labels"] += 1

                # Copy image
                shutil.copy(img_path, out_img_dir / img_name)

                # Write label file
                label_name = img_name.replace(".jpg", ".txt")
                label_path = out_label_dir / label_name

                with open(label_path, 'w') as f:
                    f.write("\n".join(yolo_lines))

                processed += 1

        return processed

    def convert_all(
        self,
        max_train: Optional[int] = None,
        max_val: Optional[int] = None
    ):
        """
        Convert all splits and generate dataset.yaml.

        Args:
            max_train: Maximum training images (for testing)
            max_val: Maximum validation images (for testing)
        """
        print("Converting BDD100K to YOLO format...")
        print(f"Output directory: {self.output_dir}")
        print()

        # Convert splits
        train_count = self.convert_split("train", max_images=max_train)
        val_count = self.convert_split("val", max_images=max_val)

        # Generate dataset.yaml
        self.generate_dataset_yaml()

        # Print statistics
        self.print_stats()

        return train_count, val_count

    def generate_dataset_yaml(self):
        """Generate YOLOv8 dataset configuration file."""
        config = {
            "path": str(self.output_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(self.CLASS_NAMES)},
            "nc": len(self.CLASS_NAMES)
        }

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"Generated dataset config: {yaml_path}")

    def print_stats(self):
        """Print conversion statistics."""
        print()
        print("=" * 50)
        print("Conversion Statistics")
        print("=" * 50)
        print(f"Total images processed: {self.stats['total_images']}")
        print(f"Images with labels: {self.stats['images_with_labels']}")
        print(f"Total bounding boxes: {self.stats['total_boxes']}")
        print(f"Filtered boxes (too small): {self.stats['filtered_boxes']}")
        print()
        print("Class distribution:")
        for cls_name, count in sorted(
            self.stats['class_counts'].items(),
            key=lambda x: -x[1]
        ):
            print(f"  {cls_name}: {count:,}")

    def analyze_class_distribution(self) -> Dict:
        """
        Analyze class distribution and calculate class weights.

        Returns:
            Dict with class statistics and recommended weights
        """
        total = sum(self.stats['class_counts'].values())
        if total == 0:
            return {}

        distribution = {}
        for cls_name in self.CLASS_NAMES:
            count = self.stats['class_counts'].get(cls_name, 0)
            distribution[cls_name] = {
                "count": count,
                "percentage": count / total * 100,
                "weight": total / (len(self.CLASS_NAMES) * count) if count > 0 else 0
            }

        return distribution


def main():
    """Main function for annotation conversion."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert BDD100K to YOLO format")
    parser.add_argument(
        "--bdd-root",
        type=str,
        default="data/raw",
        help="Root directory of BDD100K dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for YOLO format data"
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=400,
        help="Minimum bounding box area in pixels"
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Maximum training images (for testing)"
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Maximum validation images (for testing)"
    )

    args = parser.parse_args()

    converter = BDD100KToYOLO(
        bdd_root=args.bdd_root,
        output_dir=args.output_dir,
        min_box_area=args.min_area
    )

    converter.convert_all(
        max_train=args.max_train,
        max_val=args.max_val
    )


if __name__ == "__main__":
    main()

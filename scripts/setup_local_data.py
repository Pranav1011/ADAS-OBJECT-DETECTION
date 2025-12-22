#!/usr/bin/env python3
"""
Local BDD100K Data Setup Script

Converts BDD100K dataset to YOLO format for local training on Mac.
Handles both:
1. Per-image JSON labels (100K format)
2. Consolidated JSON labels (10K format)

Usage:
    python scripts/setup_local_data.py \
        --images-dir /path/to/bdd100k/images/100k \
        --labels-dir /path/to/bdd100k/labels/det_20 \
        --output-dir data/processed \
        --num-train 8000 \
        --num-val 2000
"""

import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from tqdm import tqdm
import argparse


# BDD100K to YOLO class mapping (ADAS-relevant classes)
CLASS_MAPPING = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "pedestrian": 3,
    "rider": 4,
    "bicycle": 5,
    "motorcycle": 5,  # Merge with bicycle
    "traffic light": 6,
    "traffic sign": 7,
    # Skip: train, other person, trailer, other vehicle
}

CLASS_NAMES = ["car", "truck", "bus", "pedestrian", "rider", "bicycle", "traffic_light", "traffic_sign"]


def find_images_and_labels(images_dir: Path, labels_dir: Path) -> Dict[str, Dict]:
    """
    Find all images and their corresponding labels.
    Handles both per-image JSON and consolidated JSON formats.
    """
    print("Scanning directories...")

    # Find images
    train_images = list((images_dir / "train").glob("*.jpg"))
    val_images = list((images_dir / "val").glob("*.jpg"))

    print(f"Found {len(train_images)} train images, {len(val_images)} val images")

    # Try to find labels - check for per-image JSONs first
    train_labels_dir = labels_dir / "det_train" / "det_train"
    if not train_labels_dir.exists():
        train_labels_dir = labels_dir / "det_train"

    val_labels_dir = labels_dir / "det_val" / "det_val"
    if not val_labels_dir.exists():
        val_labels_dir = labels_dir / "det_val"

    # Check if using per-image format
    per_image_format = False
    if train_labels_dir.exists():
        json_files = list(train_labels_dir.glob("*.json"))
        if json_files:
            per_image_format = True
            print(f"Detected per-image JSON format ({len(json_files)} label files)")

    # Build mapping
    data = {"train": {}, "val": {}}

    for split, images, labels_path in [
        ("train", train_images, train_labels_dir),
        ("val", val_images, val_labels_dir)
    ]:
        print(f"Processing {split} split...")

        for img_path in tqdm(images, desc=f"Matching {split}"):
            img_name = img_path.stem

            if per_image_format:
                label_file = labels_path / f"{img_name}.json"
                if label_file.exists():
                    data[split][img_name] = {
                        "image": img_path,
                        "label": label_file,
                        "format": "per_image"
                    }
            else:
                # For consolidated format, we'll load it later
                data[split][img_name] = {
                    "image": img_path,
                    "label": None,
                    "format": "consolidated"
                }

    # Load consolidated labels if needed
    if not per_image_format:
        for split, labels_file_name in [("train", "det_train.json"), ("val", "det_val.json")]:
            labels_file = labels_dir / labels_file_name
            if labels_file.exists():
                print(f"Loading consolidated {split} labels...")
                with open(labels_file) as f:
                    all_labels = json.load(f)

                labels_by_name = {item["name"].replace(".jpg", ""): item for item in all_labels}

                for img_name in data[split]:
                    if img_name in labels_by_name:
                        data[split][img_name]["labels_data"] = labels_by_name[img_name]

    print(f"Matched: {len(data['train'])} train, {len(data['val'])} val")
    return data


def convert_bbox_to_yolo(box: Dict, img_width: int, img_height: int) -> Optional[str]:
    """Convert BDD100K bbox to YOLO format."""
    category = box.get("category", "")

    if category not in CLASS_MAPPING:
        return None

    class_id = CLASS_MAPPING[category]

    # Get coordinates
    x1 = box["box2d"]["x1"]
    y1 = box["box2d"]["y1"]
    x2 = box["box2d"]["x2"]
    y2 = box["box2d"]["y2"]

    # Convert to YOLO format (normalized center x, y, width, height)
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    # Clamp values
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0, min(1, w))
    h = max(0, min(1, h))

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process_label(label_info: Dict, img_width: int = 1280, img_height: int = 720) -> List[str]:
    """Process a single label (per-image or consolidated)."""
    labels = []

    if label_info["format"] == "per_image":
        with open(label_info["label"]) as f:
            data = json.load(f)
        objects = data.get("frames", [{}])[0].get("objects", [])
    else:
        objects = label_info.get("labels_data", {}).get("labels", [])

    for obj in objects:
        if "box2d" in obj:
            yolo_line = convert_bbox_to_yolo(obj, img_width, img_height)
            if yolo_line:
                labels.append(yolo_line)

    return labels


def setup_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    num_train: int = 8000,
    num_val: int = 2000,
    seed: int = 42
):
    """Set up complete YOLO dataset from BDD100K."""
    random.seed(seed)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    # Find data
    data = find_images_and_labels(images_dir, labels_dir)

    # Sample data
    train_items = list(data["train"].items())
    val_items = list(data["val"].items())

    random.shuffle(train_items)
    random.shuffle(val_items)

    train_items = train_items[:num_train]
    val_items = val_items[:num_val]

    print(f"\nUsing {len(train_items)} train, {len(val_items)} val images")

    # Create output directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process and copy
    stats = defaultdict(int)

    for split, items in [("train", train_items), ("val", val_items)]:
        print(f"\nProcessing {split}...")

        for img_name, info in tqdm(items, desc=f"Converting {split}"):
            # Copy image
            src_img = info["image"]
            dst_img = output_dir / "images" / split / f"{img_name}.jpg"
            shutil.copy2(src_img, dst_img)

            # Convert label
            labels = process_label(info)
            label_file = output_dir / "labels" / split / f"{img_name}.txt"

            with open(label_file, "w") as f:
                f.write("\n".join(labels))

            stats[f"{split}_images"] += 1
            stats[f"{split}_labels"] += len(labels)

    # Create dataset.yaml
    yaml_content = f"""# BDD100K Object Detection Dataset
# Auto-generated for ADAS training

path: {output_dir.absolute()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}

# Dataset info
# Train images: {stats['train_images']}
# Val images: {stats['val_images']}
# Total objects: {stats['train_labels'] + stats['val_labels']}
"""

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n{'='*50}")
    print("Dataset setup complete!")
    print(f"{'='*50}")
    print(f"Train images: {stats['train_images']} ({stats['train_labels']} objects)")
    print(f"Val images:   {stats['val_images']} ({stats['val_labels']} objects)")
    print(f"Dataset YAML: {yaml_path}")
    print(f"\nTo train:")
    print(f"  make train")
    print(f"  # or")
    print(f"  python -m training.train --data {yaml_path} --model m --epochs 50 --device mps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup BDD100K data for local training")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Path to BDD100K images (e.g., bdd100k/images/100k)")
    parser.add_argument("--labels-dir", type=str, required=True,
                        help="Path to BDD100K labels (e.g., bdd100k/labels/det_20)")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--num-train", type=int, default=8000,
                        help="Number of training images to use")
    parser.add_argument("--num-val", type=int, default=2000,
                        help="Number of validation images to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    setup_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed
    )

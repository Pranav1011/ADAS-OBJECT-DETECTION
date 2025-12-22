#!/usr/bin/env python3
"""
Local BDD100K Data Setup Script

Converts BDD100K dataset to YOLO format for local training on Mac.
Handles the BDD100K structure where images and labels are in the same folder.

Usage:
    python scripts/setup_local_data.py \
        --data-dir data/raw/100k \
        --output-dir data/processed \
        --num-train 8000 \
        --num-val 2000
"""

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


def find_image_label_pairs(data_dir: Path) -> Dict[str, Dict]:
    """
    Find all image-label pairs in the BDD100K directory.
    Expects structure: data_dir/train/*.jpg + *.json, data_dir/val/*.jpg + *.json
    """
    print("Scanning for image-label pairs...")

    data = {"train": {}, "val": {}}

    for split in ["train", "val"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist")
            continue

        # Find all jpg files
        images = list(split_dir.glob("*.jpg"))
        print(f"Found {len(images)} images in {split}")

        # Match with JSON labels
        matched = 0
        for img_path in tqdm(images, desc=f"Scanning {split}"):
            json_path = img_path.with_suffix(".json")
            if json_path.exists():
                data[split][img_path.stem] = {
                    "image": img_path,
                    "label": json_path,
                }
                matched += 1

        print(f"Matched {matched} image-label pairs in {split}")

    return data


def convert_bbox_to_yolo(obj: Dict, img_width: int = 1280, img_height: int = 720) -> Optional[str]:
    """Convert BDD100K object to YOLO format."""
    category = obj.get("category", "")

    if category not in CLASS_MAPPING:
        return None

    class_id = CLASS_MAPPING[category]

    # Get bounding box
    box = obj.get("box2d")
    if not box:
        return None

    x1 = box["x1"]
    y1 = box["y1"]
    x2 = box["x2"]
    y2 = box["y2"]

    # Convert to YOLO format (normalized center x, y, width, height)
    cx = ((x1 + x2) / 2) / img_width
    cy = ((y1 + y2) / 2) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    # Clamp values to [0, 1]
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    w = max(0, min(1, w))
    h = max(0, min(1, h))

    # Skip tiny boxes
    if w < 0.001 or h < 0.001:
        return None

    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def process_label_file(json_path: Path) -> List[str]:
    """Process a BDD100K JSON label file and return YOLO format lines."""
    labels = []

    try:
        with open(json_path) as f:
            data = json.load(f)

        # BDD100K format: frames[0].objects[] contains the annotations
        frames = data.get("frames", [])
        if frames:
            objects = frames[0].get("objects", [])
            for obj in objects:
                yolo_line = convert_bbox_to_yolo(obj)
                if yolo_line:
                    labels.append(yolo_line)
    except Exception as e:
        print(f"Error processing {json_path}: {e}")

    return labels


def setup_dataset(
    data_dir: str,
    output_dir: str,
    num_train: int = 8000,
    num_val: int = 2000,
    seed: int = 42
):
    """Set up complete YOLO dataset from BDD100K."""
    random.seed(seed)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Find image-label pairs
    data = find_image_label_pairs(data_dir)

    if not data["train"] and not data["val"]:
        print("Error: No image-label pairs found!")
        print(f"Expected structure: {data_dir}/train/*.jpg + *.json")
        return

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
    class_counts = defaultdict(int)

    for split, items in [("train", train_items), ("val", val_items)]:
        print(f"\nProcessing {split}...")

        for img_name, info in tqdm(items, desc=f"Converting {split}"):
            # Copy image
            src_img = info["image"]
            dst_img = output_dir / "images" / split / f"{img_name}.jpg"
            shutil.copy2(src_img, dst_img)

            # Convert label
            labels = process_label_file(info["label"])
            label_file = output_dir / "labels" / split / f"{img_name}.txt"

            with open(label_file, "w") as f:
                f.write("\n".join(labels))

            stats[f"{split}_images"] += 1
            stats[f"{split}_objects"] += len(labels)

            # Count classes
            for label in labels:
                class_id = int(label.split()[0])
                class_counts[CLASS_NAMES[class_id]] += 1

    # Create dataset.yaml
    yaml_content = f"""# BDD100K Object Detection Dataset
# Auto-generated for ADAS training on Mac M3

path: {output_dir.absolute()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}

# Dataset Statistics
# Train images: {stats['train_images']}
# Train objects: {stats['train_objects']}
# Val images: {stats['val_images']}
# Val objects: {stats['val_objects']}
"""

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # Print summary
    print(f"\n{'='*60}")
    print("Dataset Setup Complete!")
    print(f"{'='*60}")
    print(f"\nImages:")
    print(f"  Train: {stats['train_images']:,} images ({stats['train_objects']:,} objects)")
    print(f"  Val:   {stats['val_images']:,} images ({stats['val_objects']:,} objects)")
    print(f"\nClass Distribution:")
    for cls_name in CLASS_NAMES:
        count = class_counts.get(cls_name, 0)
        print(f"  {cls_name:15s}: {count:,}")
    print(f"\nOutput: {output_dir.absolute()}")
    print(f"Config: {yaml_path}")
    print(f"\n{'='*60}")
    print("To start training:")
    print(f"{'='*60}")
    print(f"  make train")
    print(f"  # or")
    print(f"  python -m training.train --data {yaml_path} --model m --epochs 50 --device mps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup BDD100K data for local training")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to BDD100K 100k folder (containing train/ and val/)")
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
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        seed=args.seed
    )

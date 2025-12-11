#!/usr/bin/env python3
"""
Download sample driving images for testing the pipeline.

This script downloads a small set of public domain driving images
to test the data pipeline without needing the full BDD100K dataset.
"""

import os
import urllib.request
from pathlib import Path
import json


# Sample images from public sources (Pexels, Unsplash - free to use)
SAMPLE_IMAGES = [
    {
        "url": "https://images.pexels.com/photos/1000738/pexels-photo-1000738.jpeg?auto=compress&cs=tinysrgb&w=1280",
        "name": "driving_001.jpg",
        "description": "Highway driving"
    },
    {
        "url": "https://images.pexels.com/photos/210182/pexels-photo-210182.jpeg?auto=compress&cs=tinysrgb&w=1280",
        "name": "driving_002.jpg",
        "description": "City traffic"
    },
    {
        "url": "https://images.pexels.com/photos/1647976/pexels-photo-1647976.jpeg?auto=compress&cs=tinysrgb&w=1280",
        "name": "driving_003.jpg",
        "description": "Urban street"
    },
]


def download_sample_images(output_dir: str = "demo/examples"):
    """Download sample images for demo."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Downloading sample driving images...")
    print(f"Output directory: {output_path}")
    print()

    for img_info in SAMPLE_IMAGES:
        output_file = output_path / img_info["name"]

        if output_file.exists():
            print(f"  ✓ {img_info['name']} (already exists)")
            continue

        try:
            print(f"  Downloading {img_info['name']}...")
            urllib.request.urlretrieve(img_info["url"], output_file)
            print(f"  ✓ {img_info['name']}")
        except Exception as e:
            print(f"  ✗ {img_info['name']}: {e}")

    print()
    print("Done! Sample images saved to:", output_path)


def create_sample_annotations(output_dir: str = "demo/examples"):
    """Create sample YOLO annotations for demo images."""
    output_path = Path(output_dir)

    # Sample annotations (approximate boxes for demo)
    annotations = {
        "driving_001.jpg": [
            # class_id cx cy w h (normalized)
            "0 0.5 0.6 0.15 0.12",  # car
            "0 0.3 0.58 0.12 0.10",  # car
        ],
        "driving_002.jpg": [
            "0 0.4 0.55 0.18 0.15",  # car
            "0 0.7 0.52 0.14 0.12",  # car
            "6 0.25 0.35 0.03 0.06",  # traffic light
        ],
        "driving_003.jpg": [
            "0 0.5 0.65 0.20 0.18",  # car
            "3 0.3 0.6 0.05 0.15",  # pedestrian
        ],
    }

    print("Creating sample annotations...")

    for img_name, labels in annotations.items():
        label_name = img_name.replace(".jpg", ".txt")
        label_path = output_path / label_name

        with open(label_path, 'w') as f:
            f.write("\n".join(labels))

        print(f"  ✓ {label_name}")

    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download sample data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demo/examples",
        help="Output directory for sample images"
    )
    parser.add_argument(
        "--with-annotations",
        action="store_true",
        help="Also create sample annotations"
    )

    args = parser.parse_args()

    download_sample_images(args.output_dir)

    if args.with_annotations:
        create_sample_annotations(args.output_dir)

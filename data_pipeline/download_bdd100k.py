"""
Download and extract BDD100K dataset.

BDD100K requires registration at https://bdd-data.berkeley.edu/
After registration, you can download:
- bdd100k_images_10k.zip (10K subset for quick experiments)
- bdd100k_images_100k.zip (full 100K dataset)
- bdd100k_labels_release.zip (annotations)

This script handles extraction and organization of the dataset.
"""

import os
import zipfile
import shutil
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import requests


class BDD100KDownloader:
    """
    Download and organize BDD100K dataset.

    Note: Due to licensing, automatic download is not available.
    Users must manually download from https://bdd-data.berkeley.edu/
    This class handles extraction and organization.
    """

    # URLs for reference (requires authentication)
    DATASET_INFO = {
        "images_10k": {
            "filename": "bdd100k_images_10k.zip",
            "size_gb": 1.1,
            "description": "10K image subset for benchmarking"
        },
        "images_100k": {
            "filename": "bdd100k_images_100k.zip",
            "size_gb": 6.2,
            "description": "Full 100K image dataset"
        },
        "labels": {
            "filename": "bdd100k_labels_release.zip",
            "size_gb": 0.1,
            "description": "Detection annotations (JSON format)"
        }
    }

    # Target ADAS classes and their BDD100K names
    TARGET_CLASSES = {
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "pedestrian": "person",
        "cyclist": "rider",
        "motorcycle": "bike",  # includes motor
        "traffic_light": "traffic light",
        "traffic_sign": "traffic sign"
    }

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed"
    ):
        """
        Initialize downloader.

        Args:
            raw_dir: Directory for raw downloaded files
            processed_dir: Directory for processed YOLO format data
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def print_download_instructions(self):
        """Print instructions for manual download."""
        print("=" * 60)
        print("BDD100K Dataset Download Instructions")
        print("=" * 60)
        print()
        print("1. Register at: https://bdd-data.berkeley.edu/")
        print("2. After registration, download the following files:")
        print()
        for name, info in self.DATASET_INFO.items():
            print(f"   - {info['filename']} ({info['size_gb']} GB)")
            print(f"     {info['description']}")
            print()
        print(f"3. Place downloaded files in: {self.raw_dir.absolute()}")
        print()
        print("4. Run this script again to extract and organize the data.")
        print("=" * 60)

    def check_downloaded_files(self) -> dict:
        """
        Check which files have been downloaded.

        Returns:
            Dict mapping dataset name to (exists, path)
        """
        status = {}
        for name, info in self.DATASET_INFO.items():
            filepath = self.raw_dir / info["filename"]
            status[name] = {
                "exists": filepath.exists(),
                "path": filepath,
                "filename": info["filename"]
            }
        return status

    def extract_zip(self, zip_path: Path, extract_to: Path):
        """
        Extract a zip file with progress bar.

        Args:
            zip_path: Path to zip file
            extract_to: Directory to extract to
        """
        print(f"Extracting {zip_path.name}...")

        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            for member in tqdm(members, desc="Extracting"):
                zf.extract(member, extract_to)

        print(f"Extracted to {extract_to}")

    def setup_dataset(self, use_10k: bool = True) -> bool:
        """
        Set up the dataset by extracting downloaded files.

        Args:
            use_10k: If True, use 10K subset; else use full 100K

        Returns:
            True if setup successful, False otherwise
        """
        status = self.check_downloaded_files()

        # Check for required files
        images_key = "images_10k" if use_10k else "images_100k"

        if not status[images_key]["exists"]:
            print(f"Error: {status[images_key]['filename']} not found!")
            self.print_download_instructions()
            return False

        if not status["labels"]["exists"]:
            print(f"Error: {status['labels']['filename']} not found!")
            self.print_download_instructions()
            return False

        # Extract images
        images_extract_dir = self.raw_dir / "images"
        if not images_extract_dir.exists():
            self.extract_zip(status[images_key]["path"], self.raw_dir)
        else:
            print(f"Images already extracted at {images_extract_dir}")

        # Extract labels
        labels_extract_dir = self.raw_dir / "labels"
        if not labels_extract_dir.exists():
            self.extract_zip(status["labels"]["path"], self.raw_dir)
        else:
            print(f"Labels already extracted at {labels_extract_dir}")

        print("\nDataset setup complete!")
        print(f"Images: {images_extract_dir}")
        print(f"Labels: {labels_extract_dir}")

        return True

    def get_dataset_stats(self) -> dict:
        """
        Get statistics about the downloaded dataset.

        Returns:
            Dict with dataset statistics
        """
        stats = {
            "images": {"train": 0, "val": 0},
            "labels": {"train": 0, "val": 0}
        }

        # Count images
        for split in ["train", "val"]:
            img_dir = self.raw_dir / "bdd100k" / "images" / "100k" / split
            if img_dir.exists():
                stats["images"][split] = len(list(img_dir.glob("*.jpg")))

            # Also check 10k path
            img_dir_10k = self.raw_dir / "bdd100k" / "images" / "10k" / split
            if img_dir_10k.exists():
                stats["images"][split] = len(list(img_dir_10k.glob("*.jpg")))

        return stats


def main():
    """Main function for dataset download/setup."""
    import argparse

    parser = argparse.ArgumentParser(description="BDD100K Dataset Setup")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory for raw dataset files"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory for processed data"
    )
    parser.add_argument(
        "--use-full",
        action="store_true",
        help="Use full 100K dataset instead of 10K subset"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print download instructions only"
    )

    args = parser.parse_args()

    downloader = BDD100KDownloader(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir
    )

    if args.info:
        downloader.print_download_instructions()
        return

    # Check status and setup
    status = downloader.check_downloaded_files()
    print("\nDownload Status:")
    for name, info in status.items():
        emoji = "✓" if info["exists"] else "✗"
        print(f"  {emoji} {info['filename']}")
    print()

    # If files exist, extract them
    has_any = any(info["exists"] for info in status.values())
    if has_any:
        success = downloader.setup_dataset(use_10k=not args.use_full)
        if success:
            stats = downloader.get_dataset_stats()
            print(f"\nDataset Statistics:")
            print(f"  Train images: {stats['images']['train']}")
            print(f"  Val images: {stats['images']['val']}")
    else:
        downloader.print_download_instructions()


if __name__ == "__main__":
    main()

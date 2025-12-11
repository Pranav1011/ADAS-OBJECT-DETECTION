#!/bin/bash
# BDD100K Dataset Setup Script

set -e

DATA_DIR="data/raw"
PROCESSED_DIR="data/processed"

echo "=========================================="
echo "BDD100K Dataset Setup"
echo "=========================================="
echo ""

# Create directories
mkdir -p $DATA_DIR
mkdir -p $PROCESSED_DIR

# Check if files exist
if [ -f "$DATA_DIR/bdd100k_images_10k.zip" ] || [ -f "$DATA_DIR/bdd100k_images_100k.zip" ]; then
    echo "✓ Image archive found"
else
    echo "✗ Image archive not found"
    echo ""
    echo "Please download from https://bdd-data.berkeley.edu/"
    echo "Required files:"
    echo "  - bdd100k_images_10k.zip (1.1 GB) - for quick experiments"
    echo "  OR"
    echo "  - bdd100k_images_100k.zip (6.2 GB) - full dataset"
    echo ""
    echo "Place downloaded files in: $DATA_DIR"
    exit 1
fi

if [ -f "$DATA_DIR/bdd100k_labels_release.zip" ]; then
    echo "✓ Labels archive found"
else
    echo "✗ Labels archive not found"
    echo ""
    echo "Please download bdd100k_labels_release.zip from https://bdd-data.berkeley.edu/"
    echo "Place it in: $DATA_DIR"
    exit 1
fi

# Extract images
echo ""
echo "Extracting images..."
if [ -f "$DATA_DIR/bdd100k_images_10k.zip" ]; then
    unzip -q -o "$DATA_DIR/bdd100k_images_10k.zip" -d "$DATA_DIR"
    echo "✓ Extracted 10K images"
elif [ -f "$DATA_DIR/bdd100k_images_100k.zip" ]; then
    unzip -q -o "$DATA_DIR/bdd100k_images_100k.zip" -d "$DATA_DIR"
    echo "✓ Extracted 100K images"
fi

# Extract labels
echo "Extracting labels..."
unzip -q -o "$DATA_DIR/bdd100k_labels_release.zip" -d "$DATA_DIR"
echo "✓ Extracted labels"

# Convert to YOLO format
echo ""
echo "Converting annotations to YOLO format..."
python -m data_pipeline.convert_annotations \
    --bdd-root "$DATA_DIR" \
    --output-dir "$PROCESSED_DIR"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Dataset ready at: $PROCESSED_DIR"
echo "To start training, run:"
echo "  python -m training.train --data $PROCESSED_DIR/dataset.yaml"

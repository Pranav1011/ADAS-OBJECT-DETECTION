# ADAS Object Detection Pipeline

Real-time object detection for Advanced Driver Assistance Systems (ADAS), built with YOLOv8 and optimized for multi-platform deployment.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project demonstrates an end-to-end ML pipeline for autonomous driving perception:

- **Training**: YOLOv8 trained on BDD100K autonomous driving dataset
- **Quantization**: Multi-platform optimization (ONNX Runtime, TensorRT, CoreML)
- **Deployment**: FastAPI service + Gradio demo on HuggingFace Spaces

### Demo

🚀 **[Try the Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/pranav1011/adas-detection)**

![Demo GIF](docs/demo.gif)

## Features

- **8 ADAS-relevant classes**: car, truck, bus, pedestrian, cyclist, motorcycle, traffic light, traffic sign
- **Multi-platform quantization**: 4x model compression with >95% accuracy retention
- **Real-time inference**: <50ms on Apple Silicon, <10ms on NVIDIA GPUs
- **Production-ready API**: FastAPI with batch processing support

## Benchmark Results

| Model | Format | Size | Mac M3 Pro | T4 GPU | mAP@0.5 |
|-------|--------|------|------------|--------|---------|
| YOLOv8m | FP32 ONNX | 52MB | 45ms | 25ms | 0.72 |
| YOLOv8m | INT8 ONNX | 13MB | 25ms | 15ms | 0.71 |
| YOLOv8m | FP16 TensorRT | 26MB | - | 12ms | 0.72 |
| YOLOv8m | INT8 TensorRT | 13MB | - | 8ms | 0.71 |
| YOLOv8m | FP16 CoreML | 26MB | 20ms | - | 0.72 |

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Pranav1011/ADAS-OBJECT-DETECTION.git
cd ADAS-OBJECT-DETECTION

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Mac-specific (Apple Silicon)
pip install -r requirements-mac.txt

# GPU-specific (NVIDIA)
pip install -r requirements-gpu.txt
```

### Run Detection

```bash
# Single image
python -m inference.detector --model weights/best.onnx --image test.jpg

# Start API server
python -m api.main --host 0.0.0.0 --port 8000

# Start Gradio demo
python demo/app.py
```

### API Usage

```bash
# Detect objects in image
curl -X POST -F "file=@driving.jpg" http://localhost:8000/detect

# Get annotated image
curl -X POST -F "file=@driving.jpg" http://localhost:8000/detect/visualize -o output.jpg

# Benchmark model
curl http://localhost:8000/benchmark
```

## Project Structure

```
adas-object-detection/
├── configs/              # Training and quantization configs
├── data_pipeline/        # Dataset download and preprocessing
├── training/             # YOLOv8 training scripts
├── quantization/         # ONNX, TensorRT, CoreML optimization
├── inference/            # Unified detector interface
├── api/                  # FastAPI service
├── demo/                 # Gradio app for HuggingFace Spaces
├── notebooks/            # Jupyter notebooks for experiments
└── tests/                # Unit tests
```

## Training

### Dataset Setup

1. Register at [BDD100K](https://bdd-data.berkeley.edu/)
2. Download `bdd100k_images_10k.zip` and `bdd100k_labels_release.zip`
3. Place in `data/raw/`

```bash
# Convert annotations to YOLO format
python -m data_pipeline.convert_annotations \
    --bdd-root data/raw \
    --output-dir data/processed
```

### Train Model

```bash
# Train on local machine (Mac/CPU)
python -m training.train \
    --data data/processed/dataset.yaml \
    --model m \
    --epochs 100 \
    --batch 8

# Train on cloud GPU (recommended)
# See notebooks/02_baseline_training.ipynb
```

## Quantization

### ONNX Runtime (Cross-platform)

```bash
# Export to ONNX
python -m training.train --export onnx

# Quantize to INT8
python -m quantization.onnx_quantize \
    --model weights/best.onnx \
    --calibration-dir data/processed/images/val \
    --num-calibration 500
```

### TensorRT (NVIDIA GPU)

```bash
# Requires NVIDIA GPU with CUDA
python -m quantization.tensorrt_quantize \
    --model weights/best.onnx \
    --fp16 --int8 \
    --calibration-dir data/processed/images/val
```

### CoreML (Apple Silicon)

```bash
# Mac only
python -m quantization.coreml_convert \
    --model weights/best.pt \
    --half
```

### Run Benchmarks

```bash
python -m quantization.benchmark \
    --models-dir weights \
    --runs 100 \
    --output benchmark_results.json
```

## Deployment

### HuggingFace Spaces

1. Fork this repository
2. Create new Space at huggingface.co/new-space
3. Connect GitHub repository
4. Upload ONNX models to `weights/`

### Docker

```bash
docker build -t adas-detection .
docker run -p 8000:8000 adas-detection
```

## Technical Details

### Model Architecture

- **Backbone**: CSPDarknet53 (YOLOv8m variant)
- **Neck**: PANet with C2f modules
- **Head**: Decoupled head with anchor-free detection
- **Parameters**: 25.9M

### Training Configuration

- **Optimizer**: AdamW (lr=0.001, weight_decay=0.0005)
- **Scheduler**: Cosine annealing
- **Augmentation**: Mosaic, horizontal flip, HSV augmentation
- **Loss**: CIoU + DFL + BCE

### Quantization Strategy

- **ONNX Runtime**: Static INT8 quantization with entropy calibration
- **TensorRT**: FP16/INT8 with layer fusion and kernel autotuning
- **CoreML**: FP16 with Neural Engine optimization

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [ONNX Runtime](https://onnxruntime.ai/)

## Contact

- GitHub: [@Pranav1011](https://github.com/Pranav1011)
- LinkedIn: [Your LinkedIn]

---

Built as a portfolio project demonstrating ML engineering skills for autonomous vehicle perception systems.

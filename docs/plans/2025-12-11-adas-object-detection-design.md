# ADAS Object Detection Pipeline - Design Document

**Date:** 2025-12-11
**Target Role:** Rivian ML Engineer II, ADAS Platform
**Repository:** https://github.com/Pranav1011/ADAS-OBJECT-DETECTION

---

## Overview

Build an end-to-end computer vision pipeline for autonomous driving that demonstrates expertise in object detection, model optimization, and cross-platform deployment.

**Goal:** Train YOLOv8 on autonomous driving data, implement quantization across three platforms (ONNX Runtime, TensorRT, CoreML), and deploy a shareable demo on HuggingFace Spaces.

---

## Key Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Development | Local Mac (MPS) + Cloud GPU | Develop locally, train/benchmark on cloud |
| Dataset | BDD100K subset (10-20K images) | Real driving data, manageable size |
| Model | YOLOv8m (Ultralytics) | Best export pipeline, industry standard |
| Quantization | ONNX + TensorRT + CoreML | Comprehensive cross-platform comparison |
| Pruning | Out of scope | Focus on quantization depth |
| Demo | Gradio on HuggingFace Spaces | Free hosting, shareable URL |
| API | FastAPI | Local inference service |

---

## Project Structure

```
adas-object-detection/
├── configs/                  # YAML configs for training, quantization
│   ├── dataset.yaml
│   ├── training.yaml
│   └── quantization.yaml
├── data/
│   ├── raw/                  # Downloaded BDD100K
│   └── processed/            # YOLO format
├── data_pipeline/            # BDD100K download, conversion, augmentation
│   ├── __init__.py
│   ├── download_bdd100k.py
│   ├── convert_annotations.py
│   └── data_augmentation.py
├── training/                 # YOLOv8 training scripts
│   ├── __init__.py
│   ├── train.py
│   └── evaluate.py
├── quantization/
│   ├── __init__.py
│   ├── onnx_quantize.py      # ONNX Runtime INT8
│   ├── tensorrt_quantize.py  # TensorRT FP16/INT8 (cloud only)
│   ├── coreml_convert.py     # CoreML for Mac
│   └── benchmark.py          # Unified benchmarking
├── inference/                # Unified inference interface
│   ├── __init__.py
│   ├── detector.py
│   └── visualizer.py
├── api/                      # FastAPI service
│   ├── __init__.py
│   └── main.py
├── demo/                     # Gradio app for HuggingFace Spaces
│   ├── app.py
│   ├── requirements.txt
│   └── examples/
├── notebooks/                # Training & benchmarking notebooks
├── tests/
├── scripts/                  # Shell scripts
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Data Pipeline

### Dataset
- **Source:** BDD100K subset (~10-20K images)
- **Classes (8):** car, truck, bus, pedestrian, cyclist, motorcycle, traffic_light, traffic_sign
- **Format:** Convert to YOLO format (normalized xywh)
- **Split:** Train (~80%), Val (~20%)

### Augmentation
- Geometric: horizontal flip, scale (±20%)
- Photometric: brightness, contrast
- Weather: rain, fog simulation
- YOLOv8 built-in: mosaic, mixup

---

## Training Configuration

```yaml
model: yolov8m.pt
epochs: 100
batch_size: 16
img_size: 640
patience: 20
optimizer: AdamW
lr0: 0.001
weight_decay: 0.0005
warmup_epochs: 5
cos_lr: true
```

### Environments
| Environment | Hardware | Purpose |
|-------------|----------|---------|
| Local | Mac M3 Pro (MPS) | Debugging, small experiments |
| Cloud | Colab T4/A100 | Full training, TensorRT benchmarks |

### Tracking
- Weights & Biases (wandb) for experiment tracking
- Checkpoints: best.pt, best.onnx

---

## Quantization Strategy

### Three-Platform Comparison

| Platform | Format | Precision | Runs On |
|----------|--------|-----------|---------|
| ONNX Runtime | .onnx | FP32, INT8 | Everywhere |
| TensorRT | .engine | FP16, INT8 | NVIDIA GPU |
| CoreML | .mlpackage | FP16 | Apple Silicon |

### Flow
```
best.pt (PyTorch)
    ├──► best.onnx (FP32) ──► best_int8.onnx (ONNX Runtime PTQ)
    ├──► best.engine (TensorRT FP16/INT8) [Cloud]
    └──► best.mlpackage (CoreML FP16) [Mac]
```

### Calibration
- 500-1000 images from validation set
- Used for INT8 calibration

### Benchmark Table (Portfolio Centerpiece)

| Model | Size | Mac M3 (ms) | T4 (ms) | A100 (ms) | mAP@0.5 |
|-------|------|-------------|---------|-----------|---------|
| FP32 ONNX | ~50MB | X | X | X | baseline |
| INT8 ONNX | ~13MB | X | X | X | X% |
| FP16 TensorRT | ~25MB | - | X | X | X% |
| INT8 TensorRT | ~13MB | - | X | X | X% |
| FP16 CoreML | ~25MB | X | - | - | X% |

---

## API Design

### FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/detect` | POST | JSON detections |
| `/detect/visualize` | POST | Annotated image |
| `/models` | GET | List models |
| `/benchmark` | GET | Latency stats |
| `/health` | GET | Health check |

### Unified Detector Interface
```python
class ObjectDetector:
    def __init__(self, model_path: str, backend: str = "auto")
    def detect(self, image) -> List[Detection]
    def benchmark(self, num_runs=100) -> BenchmarkResult
```

---

## Gradio Demo (HuggingFace Spaces)

### Tabs
1. **Image Detection** - Upload image, see detections
2. **Video Detection** - Upload short clip, get processed video
3. **Model Comparison** - FP32 vs INT8 side-by-side
4. **Benchmark Results** - Charts and tables

### Deployment
- HuggingFace Spaces free CPU tier
- ONNX models only (portable)
- Sample driving images included

---

## Implementation Phases

### Phase 1: Foundation
- [ ] Set up project structure
- [ ] Install dependencies
- [ ] Download BDD100K subset
- [ ] Convert annotations to YOLO format
- [ ] Verify data pipeline
- **Commit to GitHub**

### Phase 2: Training
- [ ] Configure training
- [ ] Train YOLOv8m (cloud GPU)
- [ ] Evaluate and generate metrics
- [ ] Export to ONNX
- **Commit to GitHub**

### Phase 3: Quantization
- [ ] ONNX Runtime INT8 quantization
- [ ] TensorRT FP16/INT8 (cloud)
- [ ] CoreML conversion (Mac)
- [ ] Create benchmark comparison table
- **Commit to GitHub**

### Phase 4: Deployment
- [ ] Build unified inference interface
- [ ] Create FastAPI service
- [ ] Build Gradio demo
- [ ] Deploy to HuggingFace Spaces
- **Commit to GitHub**

### Phase 5: Polish
- [ ] Write comprehensive README
- [ ] Record demo GIF/video
- [ ] Final testing
- **Commit to GitHub**

---

## Success Metrics

### Model Performance
- mAP@0.5 > 0.70 on BDD100K validation
- INT8 retains >95% of FP32 accuracy

### Inference Speed
- FP32 ONNX: < 50ms on Mac M3
- INT8 ONNX: < 30ms on Mac M3
- INT8 TensorRT: < 10ms on T4

### Model Size
- FP32: ~50MB
- INT8: ~13MB (4x compression)

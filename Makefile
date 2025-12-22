# ============================================
# ADAS Object Detection - Makefile
# ============================================

.PHONY: help install install-dev install-apple setup data train evaluate export quantize benchmark api demo test lint clean

PYTHON := python
PIP := pip
VENV := venv

# Default target
help:
	@echo "ADAS Object Detection - Available Commands"
	@echo "==========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install core dependencies"
	@echo "  make install-dev    Install with dev tools"
	@echo "  make install-apple  Install with Apple Silicon support"
	@echo "  make setup          Full setup (install + download data)"
	@echo ""
	@echo "Data:"
	@echo "  make data           Download and prepare dataset"
	@echo "  make data-sample    Download sample data for testing"
	@echo ""
	@echo "Training:"
	@echo "  make train          Train YOLOv8 model"
	@echo "  make train-quick    Quick training (10 epochs)"
	@echo "  make evaluate       Evaluate trained model"
	@echo ""
	@echo "Export & Optimization:"
	@echo "  make export         Export to ONNX"
	@echo "  make quantize       Quantize to INT8"
	@echo "  make coreml         Convert to CoreML (Mac only)"
	@echo "  make benchmark      Benchmark all models"
	@echo ""
	@echo "Deployment:"
	@echo "  make api            Start FastAPI server"
	@echo "  make demo           Start Gradio demo"
	@echo ""
	@echo "Development:"
	@echo "  make test           Run tests"
	@echo "  make lint           Run linter"
	@echo "  make clean          Clean build artifacts"

# ============================================
# Setup
# ============================================

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	pre-commit install

install-apple:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-apple.txt

setup: install
	@echo "Setting up project..."
	mkdir -p data/raw data/processed weights runs

# ============================================
# Data
# ============================================

data:
	$(PYTHON) -m data_pipeline.download_bdd100k --info
	@echo ""
	@echo "Please download BDD100K dataset manually from https://bdd-data.berkeley.edu/"
	@echo "Then run: make data-convert"

data-convert:
	$(PYTHON) -m data_pipeline.convert_annotations \
		--bdd-root data/raw \
		--output-dir data/processed

data-sample:
	$(PYTHON) scripts/download_sample_data.py --with-annotations

# ============================================
# Training
# ============================================

train:
	$(PYTHON) -m training.train \
		--data data/processed/dataset.yaml \
		--model m \
		--epochs 100 \
		--batch 8 \
		--device mps

train-quick:
	$(PYTHON) -m training.train \
		--data data/processed/dataset.yaml \
		--model s \
		--epochs 10 \
		--batch 8 \
		--device mps

evaluate:
	$(PYTHON) -m training.train \
		--data data/processed/dataset.yaml \
		--model weights/best.pt \
		--evaluate-only

# ============================================
# Export & Optimization
# ============================================

export:
	$(PYTHON) -c "from ultralytics import YOLO; YOLO('weights/best.pt').export(format='onnx', imgsz=640, simplify=True)"
	@echo "Exported to weights/best.onnx"

quantize:
	$(PYTHON) -m quantization.onnx_quantize \
		--model weights/best.onnx \
		--calibration-dir data/processed/images/val \
		--num-calibration 200

coreml:
	$(PYTHON) -m quantization.coreml_convert \
		--model weights/best.pt \
		--half

benchmark:
	$(PYTHON) -m quantization.benchmark \
		--models-dir weights \
		--runs 100 \
		--output benchmark_results.json

# ============================================
# Deployment
# ============================================

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

demo:
	$(PYTHON) demo/app.py

# ============================================
# Development
# ============================================

test:
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

typecheck:
	mypy data_pipeline training quantization inference api

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info build dist
	rm -rf runs/detect runs/train/*/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================
# Docker (optional)
# ============================================

docker-build:
	docker build -t adas-detection:latest .

docker-run:
	docker run -p 8000:8000 adas-detection:latest

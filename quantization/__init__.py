"""
Quantization module for model optimization.

Supports:
- ONNX Runtime quantization (cross-platform)
- TensorRT optimization (NVIDIA GPU)
- CoreML conversion (Apple Silicon)
"""

from .onnx_quantize import ONNXQuantizer
from .benchmark import ModelBenchmark

__all__ = [
    "ONNXQuantizer",
    "ModelBenchmark",
]

"""
CoreML conversion for Apple Silicon deployment.

Converts YOLOv8 PyTorch model to CoreML format for
efficient inference on Mac, iPhone, and iPad.
"""

from pathlib import Path
from typing import Dict, Optional, List
import numpy as np

# CoreML imports (available on macOS)
try:
    import coremltools as ct
    COREML_AVAILABLE = True
except ImportError:
    COREML_AVAILABLE = False
    print("CoreML tools not available. Install with: pip install coremltools")


class CoreMLConverter:
    """
    Convert YOLOv8 models to CoreML format.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str = "weights"
    ):
        """
        Initialize CoreML converter.

        Args:
            model_path: Path to PyTorch model (.pt) or ONNX model (.onnx)
            output_dir: Directory to save CoreML models
        """
        if not COREML_AVAILABLE:
            raise RuntimeError(
                "CoreML tools not available. "
                "Install with: pip install coremltools"
            )

        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        print(f"CoreML Converter initialized")
        print(f"  Model: {self.model_path}")

    def convert_from_ultralytics(
        self,
        img_size: int = 640,
        half: bool = True,
        nms: bool = False,
        output_name: Optional[str] = None
    ) -> str:
        """
        Convert YOLOv8 model using Ultralytics export.

        This is the recommended method as it handles all
        YOLOv8-specific preprocessing and postprocessing.

        Args:
            img_size: Input image size
            half: Use FP16 half precision
            nms: Include NMS in the model
            output_name: Output model name

        Returns:
            Path to CoreML model
        """
        from ultralytics import YOLO

        print("Converting YOLOv8 to CoreML using Ultralytics export...")

        # Load YOLOv8 model
        model = YOLO(str(self.model_path))

        # Export to CoreML
        export_path = model.export(
            format="coreml",
            imgsz=img_size,
            half=half,
            nms=nms
        )

        print(f"Exported CoreML model: {export_path}")
        return str(export_path)

    def convert_from_onnx(
        self,
        img_size: int = 640,
        compute_units: str = "ALL",
        output_name: Optional[str] = None
    ) -> str:
        """
        Convert ONNX model to CoreML.

        Args:
            img_size: Input image size
            compute_units: Compute units to use
                - "ALL": Use all available (CPU, GPU, Neural Engine)
                - "CPU_ONLY": CPU only
                - "CPU_AND_GPU": CPU and GPU
                - "CPU_AND_NE": CPU and Neural Engine
            output_name: Output model name

        Returns:
            Path to CoreML model
        """
        if output_name is None:
            output_name = self.model_path.stem + ".mlpackage"

        output_path = self.output_dir / output_name

        print(f"Converting ONNX to CoreML...")
        print(f"  Compute units: {compute_units}")

        # Map compute units string to enum
        compute_units_map = {
            "ALL": ct.ComputeUnit.ALL,
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        }

        compute_unit = compute_units_map.get(compute_units, ct.ComputeUnit.ALL)

        # Convert ONNX to CoreML
        mlmodel = ct.convert(
            str(self.model_path),
            convert_to="mlprogram",
            compute_units=compute_unit,
            minimum_deployment_target=ct.target.macOS13
        )

        # Save model
        mlmodel.save(str(output_path))

        print(f"Saved CoreML model: {output_path}")
        return str(output_path)

    def quantize_to_int8(
        self,
        coreml_path: str,
        calibration_images: Optional[List[str]] = None,
        output_name: Optional[str] = None
    ) -> str:
        """
        Quantize CoreML model to INT8.

        Note: CoreML uses weight quantization, which reduces
        model size but keeps activations in FP16.

        Args:
            coreml_path: Path to CoreML model
            calibration_images: Optional calibration images
            output_name: Output model name

        Returns:
            Path to quantized model
        """
        if output_name is None:
            stem = Path(coreml_path).stem
            output_name = f"{stem}_int8.mlpackage"

        output_path = self.output_dir / output_name

        print("Quantizing CoreML model to INT8...")

        # Load model
        mlmodel = ct.models.MLModel(coreml_path)

        # Quantize weights to 8-bit
        op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            weight_threshold=512
        )

        config = ct.optimize.coreml.OptimizationConfig(
            global_config=op_config
        )

        quantized_model = ct.optimize.coreml.linear_quantize_weights(
            mlmodel,
            config=config
        )

        # Save quantized model
        quantized_model.save(str(output_path))

        print(f"Saved quantized CoreML model: {output_path}")
        return str(output_path)

    def get_model_info(self, coreml_path: str) -> Dict:
        """
        Get information about a CoreML model.

        Args:
            coreml_path: Path to CoreML model

        Returns:
            Dict with model information
        """
        mlmodel = ct.models.MLModel(coreml_path)
        spec = mlmodel.get_spec()

        info = {
            "size_mb": self._get_model_size(coreml_path),
            "inputs": [],
            "outputs": [],
            "compute_units": str(mlmodel.compute_unit)
        }

        # Input info
        for input_desc in spec.description.input:
            info["inputs"].append({
                "name": input_desc.name,
                "type": str(input_desc.type.WhichOneof("Type"))
            })

        # Output info
        for output_desc in spec.description.output:
            info["outputs"].append({
                "name": output_desc.name,
                "type": str(output_desc.type.WhichOneof("Type"))
            })

        return info

    def _get_model_size(self, model_path: str) -> float:
        """Get model size in MB."""
        path = Path(model_path)
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        elif path.is_dir():
            # For .mlpackage directories
            total = 0
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
            return total / (1024 * 1024)
        return 0


class CoreMLInference:
    """
    CoreML inference wrapper for YOLOv8.
    """

    def __init__(self, model_path: str):
        """
        Load CoreML model for inference.

        Args:
            model_path: Path to .mlpackage model
        """
        if not COREML_AVAILABLE:
            raise RuntimeError("CoreML tools not available")

        self.model = ct.models.MLModel(model_path)
        print(f"Loaded CoreML model: {model_path}")

    def preprocess(self, image: np.ndarray, img_size: int = 640) -> np.ndarray:
        """
        Preprocess image for CoreML inference.

        Args:
            image: Input image (BGR, HWC)
            img_size: Target size

        Returns:
            Preprocessed image
        """
        import cv2

        # Resize
        img = cv2.resize(image, (img_size, img_size))

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        # Add batch dimension
        img = np.expand_dims(img, 0)

        return img

    def infer(self, image: np.ndarray) -> Dict:
        """
        Run inference on preprocessed image.

        Args:
            image: Preprocessed image (NCHW, float32)

        Returns:
            Model predictions
        """
        # CoreML expects specific input format
        predictions = self.model.predict({"images": image})
        return predictions

    def benchmark(
        self,
        img_size: int = 640,
        num_runs: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Benchmark inference speed.

        Args:
            img_size: Input image size
            num_runs: Number of benchmark iterations
            warmup: Number of warmup iterations

        Returns:
            Dict with timing statistics
        """
        import time

        # Create dummy input
        dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

        # Warmup
        for _ in range(warmup):
            self.infer(dummy_input)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.infer(dummy_input)
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "fps": float(1000 / np.mean(times))
        }


def main():
    """Main function for CoreML conversion."""
    import argparse

    if not COREML_AVAILABLE:
        print("CoreML tools not available on this system.")
        print("Install with: pip install coremltools")
        return

    parser = argparse.ArgumentParser(description="CoreML Conversion")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to PyTorch (.pt) or ONNX (.onnx) model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights",
        help="Output directory"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 precision"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize to INT8"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark after conversion"
    )

    args = parser.parse_args()

    converter = CoreMLConverter(
        model_path=args.model,
        output_dir=args.output_dir
    )

    # Convert based on file type
    if args.model.endswith('.pt'):
        coreml_path = converter.convert_from_ultralytics(
            img_size=args.img_size,
            half=args.half
        )
    else:
        coreml_path = converter.convert_from_onnx(
            img_size=args.img_size
        )

    # Quantize if requested
    if args.quantize:
        coreml_path = converter.quantize_to_int8(coreml_path)

    # Print model info
    info = converter.get_model_info(coreml_path)
    print(f"\nModel Info:")
    print(f"  Size: {info['size_mb']:.2f} MB")
    print(f"  Compute Units: {info['compute_units']}")

    # Benchmark if requested
    if args.benchmark:
        inference = CoreMLInference(coreml_path)
        results = inference.benchmark()
        print(f"\nBenchmark Results:")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  P95: {results['p95_ms']:.2f} ms")
        print(f"  FPS: {results['fps']:.1f}")


if __name__ == "__main__":
    main()

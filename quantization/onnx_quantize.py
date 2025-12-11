"""
ONNX Runtime quantization for cross-platform deployment.

Supports:
- Post-Training Quantization (PTQ) to INT8
- Dynamic quantization
- Static quantization with calibration
"""

from pathlib import Path
from typing import Dict, List, Optional, Generator
import numpy as np
import cv2
from tqdm import tqdm

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    quantize_dynamic,
    CalibrationDataReader,
    QuantFormat,
    QuantType
)


class CalibrationDataLoader(CalibrationDataReader):
    """
    Calibration data loader for ONNX static quantization.
    """

    def __init__(
        self,
        calibration_images: List[str],
        input_name: str = "images",
        img_size: int = 640,
        batch_size: int = 1
    ):
        """
        Initialize calibration data loader.

        Args:
            calibration_images: List of image paths for calibration
            input_name: Name of the input tensor
            img_size: Input image size
            batch_size: Batch size (usually 1 for calibration)
        """
        self.image_paths = calibration_images
        self.input_name = input_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.index = 0

        # Preprocessing parameters (same as YOLOv8)
        self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def preprocess(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for model input.

        Args:
            image_path: Path to image

        Returns:
            Preprocessed image array (NCHW format)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0

        # Apply normalization
        img = (img - self.mean) / self.std

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Get next calibration sample.

        Returns:
            Dict with input tensor or None if exhausted
        """
        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        try:
            img = self.preprocess(image_path)
            return {self.input_name: img}
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return self.get_next()

    def rewind(self):
        """Reset the data loader."""
        self.index = 0


class ONNXQuantizer:
    """
    ONNX Runtime quantization for YOLOv8 models.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str = "weights"
    ):
        """
        Initialize ONNX quantizer.

        Args:
            model_path: Path to FP32 ONNX model
            output_dir: Directory to save quantized models
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verify model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        # Load model to get input/output names
        self.model = onnx.load(str(self.model_path))
        self.input_name = self.model.graph.input[0].name

        print(f"Loaded ONNX model: {self.model_path}")
        print(f"  Input name: {self.input_name}")

    def quantize_dynamic(
        self,
        output_name: Optional[str] = None
    ) -> str:
        """
        Apply dynamic quantization (no calibration needed).

        Dynamic quantization quantizes weights to INT8 but keeps
        activations in FP32. Fast but less accurate than static.

        Args:
            output_name: Output model filename (optional)

        Returns:
            Path to quantized model
        """
        if output_name is None:
            output_name = self.model_path.stem + "_dynamic_int8.onnx"

        output_path = self.output_dir / output_name

        print("Applying dynamic quantization...")

        quantize_dynamic(
            model_input=str(self.model_path),
            model_output=str(output_path),
            weight_type=QuantType.QUInt8
        )

        print(f"Saved dynamic quantized model: {output_path}")
        return str(output_path)

    def quantize_static(
        self,
        calibration_images: List[str],
        output_name: Optional[str] = None,
        per_channel: bool = True,
        quant_format: QuantFormat = QuantFormat.QDQ
    ) -> str:
        """
        Apply static quantization with calibration.

        Static quantization quantizes both weights and activations
        to INT8. Requires calibration data but gives best speedup.

        Args:
            calibration_images: List of image paths for calibration
            output_name: Output model filename (optional)
            per_channel: Use per-channel quantization (more accurate)
            quant_format: Quantization format (QDQ or QOperator)

        Returns:
            Path to quantized model
        """
        if output_name is None:
            output_name = self.model_path.stem + "_static_int8.onnx"

        output_path = self.output_dir / output_name

        print(f"Applying static quantization with {len(calibration_images)} calibration images...")

        # Create calibration data reader
        calibration_reader = CalibrationDataLoader(
            calibration_images=calibration_images,
            input_name=self.input_name,
            img_size=640
        )

        # Quantize
        quantize_static(
            model_input=str(self.model_path),
            model_output=str(output_path),
            calibration_data_reader=calibration_reader,
            quant_format=quant_format,
            per_channel=per_channel,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8
        )

        print(f"Saved static quantized model: {output_path}")
        return str(output_path)

    def verify_quantization(
        self,
        quantized_path: str,
        test_images: List[str],
        num_images: int = 10
    ) -> Dict:
        """
        Verify quantization by comparing FP32 and INT8 outputs.

        Args:
            quantized_path: Path to quantized model
            test_images: List of test image paths
            num_images: Number of images to test

        Returns:
            Dict with verification metrics
        """
        print("Verifying quantization accuracy...")

        # Load both models
        fp32_session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )
        int8_session = ort.InferenceSession(
            quantized_path,
            providers=['CPUExecutionProvider']
        )

        # Preprocess function
        def preprocess(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (640, 640))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            return np.expand_dims(img, 0)

        # Compare outputs
        differences = []
        test_images = test_images[:num_images]

        for img_path in tqdm(test_images, desc="Verifying"):
            try:
                input_data = preprocess(img_path)

                fp32_out = fp32_session.run(None, {self.input_name: input_data})[0]
                int8_out = int8_session.run(None, {self.input_name: input_data})[0]

                # Calculate difference
                diff = np.abs(fp32_out - int8_out).mean()
                differences.append(diff)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        differences = np.array(differences)

        metrics = {
            "mean_absolute_diff": float(differences.mean()),
            "max_absolute_diff": float(differences.max()),
            "std_absolute_diff": float(differences.std()),
            "num_images_tested": len(differences)
        }

        print(f"\nVerification Results:")
        print(f"  Mean absolute difference: {metrics['mean_absolute_diff']:.6f}")
        print(f"  Max absolute difference: {metrics['max_absolute_diff']:.6f}")

        return metrics

    def get_model_size(self, model_path: str) -> float:
        """Get model size in MB."""
        return Path(model_path).stat().st_size / (1024 * 1024)

    def compare_sizes(self, quantized_path: str) -> Dict:
        """
        Compare FP32 and quantized model sizes.

        Args:
            quantized_path: Path to quantized model

        Returns:
            Dict with size comparison
        """
        fp32_size = self.get_model_size(self.model_path)
        int8_size = self.get_model_size(quantized_path)

        return {
            "fp32_size_mb": fp32_size,
            "int8_size_mb": int8_size,
            "compression_ratio": fp32_size / int8_size,
            "size_reduction_percent": (1 - int8_size / fp32_size) * 100
        }


def main():
    """Main function for ONNX quantization."""
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser(description="ONNX Quantization")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to FP32 ONNX model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights",
        help="Output directory"
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Directory with calibration images (for static quantization)"
    )
    parser.add_argument(
        "--num-calibration",
        type=int,
        default=500,
        help="Number of calibration images"
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use dynamic quantization (no calibration needed)"
    )

    args = parser.parse_args()

    quantizer = ONNXQuantizer(
        model_path=args.model,
        output_dir=args.output_dir
    )

    if args.dynamic:
        # Dynamic quantization
        quantized_path = quantizer.quantize_dynamic()
    else:
        # Static quantization with calibration
        if args.calibration_dir is None:
            print("Error: --calibration-dir required for static quantization")
            return

        calibration_images = glob(f"{args.calibration_dir}/*.jpg")
        calibration_images = calibration_images[:args.num_calibration]

        if len(calibration_images) == 0:
            print(f"No images found in {args.calibration_dir}")
            return

        quantized_path = quantizer.quantize_static(calibration_images)

    # Compare sizes
    size_comparison = quantizer.compare_sizes(quantized_path)
    print(f"\nModel Size Comparison:")
    print(f"  FP32: {size_comparison['fp32_size_mb']:.2f} MB")
    print(f"  INT8: {size_comparison['int8_size_mb']:.2f} MB")
    print(f"  Compression: {size_comparison['compression_ratio']:.2f}x")
    print(f"  Reduction: {size_comparison['size_reduction_percent']:.1f}%")


if __name__ == "__main__":
    main()

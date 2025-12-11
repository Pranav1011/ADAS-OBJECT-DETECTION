"""
TensorRT optimization for NVIDIA GPU deployment.

Note: This module requires NVIDIA GPU with CUDA and TensorRT installed.
For Mac development, use ONNX Runtime or CoreML instead.
This is intended for cloud GPU deployment (Colab, AWS, etc.)
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# TensorRT imports (will fail on non-NVIDIA systems)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available. This module requires NVIDIA GPU with CUDA.")


class TensorRTOptimizer:
    """
    TensorRT optimization for YOLOv8 models.

    Supports FP16 and INT8 precision modes.
    """

    def __init__(
        self,
        onnx_path: str,
        output_dir: str = "weights",
        workspace_size: int = 1 << 30,  # 1GB
        verbose: bool = False
    ):
        """
        Initialize TensorRT optimizer.

        Args:
            onnx_path: Path to ONNX model
            output_dir: Directory to save TensorRT engines
            workspace_size: GPU workspace size in bytes
            verbose: Enable verbose logging
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError(
                "TensorRT is not available. "
                "This module requires NVIDIA GPU with CUDA and TensorRT installed."
            )

        self.onnx_path = Path(onnx_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_size = workspace_size

        # Set up TensorRT logger
        self.logger = trt.Logger(
            trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
        )

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        print(f"TensorRT Optimizer initialized")
        print(f"  ONNX model: {self.onnx_path}")
        print(f"  TensorRT version: {trt.__version__}")

    def build_engine(
        self,
        fp16: bool = True,
        int8: bool = False,
        int8_calibrator: Optional[object] = None,
        output_name: Optional[str] = None
    ) -> str:
        """
        Build optimized TensorRT engine.

        Args:
            fp16: Enable FP16 precision
            int8: Enable INT8 precision (requires calibrator)
            int8_calibrator: INT8 calibration data provider
            output_name: Output engine filename

        Returns:
            Path to TensorRT engine file
        """
        if output_name is None:
            precision = "int8" if int8 else ("fp16" if fp16 else "fp32")
            output_name = f"{self.onnx_path.stem}_{precision}.engine"

        output_path = self.output_dir / output_name

        print(f"Building TensorRT engine...")
        print(f"  FP16: {fp16}")
        print(f"  INT8: {int8}")

        # Create builder and config
        builder = trt.Builder(self.logger)
        config = builder.create_builder_config()

        # Set workspace size
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            self.workspace_size
        )

        # Set precision flags
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            if int8_calibrator:
                config.int8_calibrator = int8_calibrator

        # Create network
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)

        # Parse ONNX
        parser = trt.OnnxParser(network, self.logger)

        with open(self.onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"ONNX Parse Error: {parser.get_error(i)}")
                raise RuntimeError("Failed to parse ONNX model")

        # Build engine
        print("Building engine (this may take a few minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        with open(output_path, 'wb') as f:
            f.write(serialized_engine)

        print(f"Saved TensorRT engine: {output_path}")
        return str(output_path)

    def get_engine_info(self, engine_path: str) -> Dict:
        """
        Get information about a TensorRT engine.

        Args:
            engine_path: Path to .engine file

        Returns:
            Dict with engine information
        """
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            engine = runtime.deserialize_cuda_engine(f.read())

        info = {
            "num_bindings": engine.num_bindings,
            "bindings": [],
            "size_mb": Path(engine_path).stat().st_size / (1024 * 1024)
        }

        for i in range(engine.num_bindings):
            binding_info = {
                "name": engine.get_binding_name(i),
                "shape": engine.get_binding_shape(i),
                "dtype": str(engine.get_binding_dtype(i)),
                "is_input": engine.binding_is_input(i)
            }
            info["bindings"].append(binding_info)

        return info


class INT8Calibrator(trt.IInt8EntropyCalibrator2):
    """
    INT8 calibration data provider for TensorRT.
    """

    def __init__(
        self,
        calibration_images: List[str],
        batch_size: int = 8,
        img_size: int = 640,
        cache_file: str = "calibration.cache"
    ):
        """
        Initialize INT8 calibrator.

        Args:
            calibration_images: List of calibration image paths
            batch_size: Batch size for calibration
            img_size: Input image size
            cache_file: Path to save calibration cache
        """
        super().__init__()

        self.calibration_images = calibration_images
        self.batch_size = batch_size
        self.img_size = img_size
        self.cache_file = cache_file

        self.index = 0

        # Allocate GPU memory for batch
        self.batch_data = np.zeros(
            (batch_size, 3, img_size, img_size),
            dtype=np.float32
        )
        self.d_input = cuda.mem_alloc(self.batch_data.nbytes)

    def preprocess(self, image_path: str) -> np.ndarray:
        """Preprocess image for calibration."""
        import cv2

        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return img

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """Get next batch of calibration data."""
        if self.index >= len(self.calibration_images):
            return None

        # Load batch
        batch_images = self.calibration_images[
            self.index:self.index + self.batch_size
        ]
        self.index += self.batch_size

        if len(batch_images) < self.batch_size:
            return None

        for i, img_path in enumerate(batch_images):
            self.batch_data[i] = self.preprocess(img_path)

        # Copy to GPU
        cuda.memcpy_htod(self.d_input, self.batch_data)

        return [int(self.d_input)]

    def read_calibration_cache(self):
        """Read calibration cache from file."""
        if Path(self.cache_file).exists():
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Write calibration cache to file."""
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


class TensorRTInference:
    """
    TensorRT inference engine wrapper.
    """

    def __init__(self, engine_path: str):
        """
        Load TensorRT engine for inference.

        Args:
            engine_path: Path to .engine file
        """
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            size = np.prod(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                self.inputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape
                })
            else:
                self.outputs.append({
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape
                })

        self.stream = cuda.Stream()

    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on input data.

        Args:
            input_data: Preprocessed input (NCHW format)

        Returns:
            List of output arrays
        """
        # Copy input to host buffer
        np.copyto(self.inputs[0]['host'], input_data.ravel())

        # Transfer to GPU
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Transfer outputs back
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(
                output['host'],
                output['device'],
                self.stream
            )
            outputs.append(output['host'].reshape(output['shape']))

        self.stream.synchronize()

        return outputs

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
    """Main function for TensorRT optimization."""
    import argparse

    if not TENSORRT_AVAILABLE:
        print("TensorRT is not available on this system.")
        print("This script requires NVIDIA GPU with CUDA and TensorRT.")
        return

    parser = argparse.ArgumentParser(description="TensorRT Optimization")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights",
        help="Output directory"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 precision"
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Enable INT8 precision"
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Directory with calibration images (required for INT8)"
    )

    args = parser.parse_args()

    optimizer = TensorRTOptimizer(
        onnx_path=args.model,
        output_dir=args.output_dir
    )

    # Build INT8 calibrator if needed
    calibrator = None
    if args.int8 and args.calibration_dir:
        from glob import glob
        calibration_images = glob(f"{args.calibration_dir}/*.jpg")[:500]
        calibrator = INT8Calibrator(calibration_images)

    # Build engine
    engine_path = optimizer.build_engine(
        fp16=args.fp16,
        int8=args.int8,
        int8_calibrator=calibrator
    )

    # Print engine info
    info = optimizer.get_engine_info(engine_path)
    print(f"\nEngine Info:")
    print(f"  Size: {info['size_mb']:.2f} MB")
    print(f"  Bindings: {info['num_bindings']}")
    for binding in info['bindings']:
        io_type = "Input" if binding['is_input'] else "Output"
        print(f"    {io_type}: {binding['name']} {binding['shape']}")


if __name__ == "__main__":
    main()

"""
Unified benchmarking for comparing model performance across formats.

Compares:
- ONNX Runtime (FP32, INT8)
- TensorRT (FP16, INT8) - NVIDIA only
- CoreML (FP16) - Apple Silicon only
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import time
import json
from dataclasses import dataclass, asdict
import platform


@dataclass
class BenchmarkResult:
    """Benchmark result for a single model."""
    model_name: str
    format: str
    precision: str
    size_mb: float
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float
    device: str
    hardware: str


class ModelBenchmark:
    """
    Unified benchmark runner for all model formats.
    """

    def __init__(
        self,
        img_size: int = 640,
        num_runs: int = 100,
        warmup: int = 20
    ):
        """
        Initialize benchmark runner.

        Args:
            img_size: Input image size
            num_runs: Number of benchmark iterations
            warmup: Number of warmup iterations
        """
        self.img_size = img_size
        self.num_runs = num_runs
        self.warmup = warmup

        # Detect hardware
        self.hardware = self._detect_hardware()
        print(f"Benchmark initialized")
        print(f"  Hardware: {self.hardware}")
        print(f"  Image size: {img_size}")
        print(f"  Runs: {num_runs}")

    def _detect_hardware(self) -> str:
        """Detect hardware platform."""
        system = platform.system()
        machine = platform.machine()

        if system == "Darwin":
            # macOS
            if machine == "arm64":
                return "Apple Silicon"
            return "Intel Mac"
        elif system == "Linux":
            try:
                import torch
                if torch.cuda.is_available():
                    return f"NVIDIA {torch.cuda.get_device_name(0)}"
            except ImportError:
                pass
            return "Linux CPU"
        return f"{system} {machine}"

    def _create_dummy_input(self) -> np.ndarray:
        """Create dummy input for benchmarking."""
        return np.random.randn(
            1, 3, self.img_size, self.img_size
        ).astype(np.float32)

    def _get_model_size(self, path: str) -> float:
        """Get model size in MB."""
        path = Path(path)
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

    def benchmark_onnx(
        self,
        model_path: str,
        precision: str = "fp32",
        use_gpu: bool = False
    ) -> BenchmarkResult:
        """
        Benchmark ONNX Runtime model.

        Args:
            model_path: Path to ONNX model
            precision: Model precision (fp32, int8)
            use_gpu: Use GPU if available

        Returns:
            BenchmarkResult
        """
        import onnxruntime as ort

        print(f"\nBenchmarking ONNX: {model_path}")

        # Select execution provider
        if use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Create session
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name

        # Create input
        dummy_input = self._create_dummy_input()

        # Warmup
        for _ in range(self.warmup):
            session.run(None, {input_name: dummy_input})

        # Benchmark
        times = []
        for _ in range(self.num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)

        return BenchmarkResult(
            model_name=Path(model_path).stem,
            format="ONNX",
            precision=precision,
            size_mb=self._get_model_size(model_path),
            mean_ms=float(np.mean(times)),
            std_ms=float(np.std(times)),
            p50_ms=float(np.percentile(times, 50)),
            p95_ms=float(np.percentile(times, 95)),
            p99_ms=float(np.percentile(times, 99)),
            fps=float(1000 / np.mean(times)),
            device="GPU" if use_gpu else "CPU",
            hardware=self.hardware
        )

    def benchmark_coreml(
        self,
        model_path: str,
        precision: str = "fp16"
    ) -> Optional[BenchmarkResult]:
        """
        Benchmark CoreML model (Apple Silicon only).

        Args:
            model_path: Path to .mlpackage model
            precision: Model precision

        Returns:
            BenchmarkResult or None if not available
        """
        try:
            import coremltools as ct
        except ImportError:
            print("CoreML not available")
            return None

        print(f"\nBenchmarking CoreML: {model_path}")

        # Load model
        model = ct.models.MLModel(model_path)

        # Create input
        dummy_input = self._create_dummy_input()

        # Warmup
        for _ in range(self.warmup):
            model.predict({"images": dummy_input})

        # Benchmark
        times = []
        for _ in range(self.num_runs):
            start = time.perf_counter()
            model.predict({"images": dummy_input})
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)

        return BenchmarkResult(
            model_name=Path(model_path).stem,
            format="CoreML",
            precision=precision,
            size_mb=self._get_model_size(model_path),
            mean_ms=float(np.mean(times)),
            std_ms=float(np.std(times)),
            p50_ms=float(np.percentile(times, 50)),
            p95_ms=float(np.percentile(times, 95)),
            p99_ms=float(np.percentile(times, 99)),
            fps=float(1000 / np.mean(times)),
            device="ANE+GPU",
            hardware=self.hardware
        )

    def benchmark_tensorrt(
        self,
        engine_path: str,
        precision: str = "fp16"
    ) -> Optional[BenchmarkResult]:
        """
        Benchmark TensorRT engine (NVIDIA GPU only).

        Args:
            engine_path: Path to .engine file
            precision: Model precision

        Returns:
            BenchmarkResult or None if not available
        """
        try:
            from .tensorrt_quantize import TensorRTInference, TENSORRT_AVAILABLE
            if not TENSORRT_AVAILABLE:
                return None
        except ImportError:
            print("TensorRT not available")
            return None

        print(f"\nBenchmarking TensorRT: {engine_path}")

        inference = TensorRTInference(engine_path)
        results = inference.benchmark(
            img_size=self.img_size,
            num_runs=self.num_runs,
            warmup=self.warmup
        )

        return BenchmarkResult(
            model_name=Path(engine_path).stem,
            format="TensorRT",
            precision=precision,
            size_mb=self._get_model_size(engine_path),
            mean_ms=results["mean_ms"],
            std_ms=results["std_ms"],
            p50_ms=results["p50_ms"],
            p95_ms=results["p95_ms"],
            p99_ms=results["p99_ms"],
            fps=results["fps"],
            device="GPU",
            hardware=self.hardware
        )

    def benchmark_pytorch(
        self,
        model_path: str,
        device: str = "auto"
    ) -> BenchmarkResult:
        """
        Benchmark PyTorch model.

        Args:
            model_path: Path to .pt model
            device: Device to benchmark on

        Returns:
            BenchmarkResult
        """
        import torch
        from ultralytics import YOLO

        print(f"\nBenchmarking PyTorch: {model_path}")

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Load model
        model = YOLO(model_path)

        # Create input
        dummy_input = torch.randn(
            1, 3, self.img_size, self.img_size,
            device=device
        )

        # Warmup
        for _ in range(self.warmup):
            model.model(dummy_input)

        # Benchmark
        times = []
        for _ in range(self.num_runs):
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            model.model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()

            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)

        return BenchmarkResult(
            model_name=Path(model_path).stem,
            format="PyTorch",
            precision="fp32",
            size_mb=self._get_model_size(model_path),
            mean_ms=float(np.mean(times)),
            std_ms=float(np.std(times)),
            p50_ms=float(np.percentile(times, 50)),
            p95_ms=float(np.percentile(times, 95)),
            p99_ms=float(np.percentile(times, 99)),
            fps=float(1000 / np.mean(times)),
            device=device.upper(),
            hardware=self.hardware
        )

    def run_all(
        self,
        models: Dict[str, str]
    ) -> List[BenchmarkResult]:
        """
        Run benchmarks on all provided models.

        Args:
            models: Dict mapping model description to path
                   e.g., {"fp32_onnx": "weights/best.onnx"}

        Returns:
            List of BenchmarkResults
        """
        results = []

        for name, path in models.items():
            path = Path(path)

            if not path.exists():
                print(f"Model not found: {path}")
                continue

            suffix = path.suffix.lower()

            try:
                if suffix == ".onnx":
                    precision = "int8" if "int8" in name else "fp32"
                    result = self.benchmark_onnx(str(path), precision)

                elif suffix == ".engine":
                    precision = "int8" if "int8" in name else "fp16"
                    result = self.benchmark_tensorrt(str(path), precision)

                elif suffix == ".mlpackage" or path.is_dir():
                    precision = "fp16"
                    result = self.benchmark_coreml(str(path), precision)

                elif suffix == ".pt":
                    result = self.benchmark_pytorch(str(path))

                else:
                    print(f"Unknown format: {suffix}")
                    continue

                if result:
                    results.append(result)

            except Exception as e:
                print(f"Error benchmarking {path}: {e}")

        return results

    def generate_report(
        self,
        results: List[BenchmarkResult],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate benchmark comparison report.

        Args:
            results: List of benchmark results
            output_path: Optional path to save JSON report

        Returns:
            Formatted report string
        """
        # Sort by mean latency
        results = sorted(results, key=lambda x: x.mean_ms)

        # Generate table
        report = []
        report.append("\n" + "=" * 80)
        report.append("MODEL BENCHMARK COMPARISON")
        report.append("=" * 80)
        report.append(f"Hardware: {self.hardware}")
        report.append(f"Image Size: {self.img_size}x{self.img_size}")
        report.append(f"Benchmark Runs: {self.num_runs}")
        report.append("=" * 80)
        report.append("")

        # Table header
        header = f"{'Model':<30} {'Format':<10} {'Prec':<6} {'Size(MB)':<10} {'Mean(ms)':<10} {'P95(ms)':<10} {'FPS':<8}"
        report.append(header)
        report.append("-" * 80)

        for r in results:
            row = f"{r.model_name:<30} {r.format:<10} {r.precision:<6} {r.size_mb:<10.2f} {r.mean_ms:<10.2f} {r.p95_ms:<10.2f} {r.fps:<8.1f}"
            report.append(row)

        report.append("-" * 80)
        report.append("")

        # Summary
        if len(results) >= 2:
            baseline = results[-1]  # Slowest (usually FP32)
            fastest = results[0]

            report.append("SUMMARY:")
            report.append(f"  Fastest: {fastest.model_name} ({fastest.format} {fastest.precision})")
            report.append(f"  Speedup vs baseline: {baseline.mean_ms / fastest.mean_ms:.2f}x")
            report.append(f"  Size reduction: {(1 - fastest.size_mb / baseline.size_mb) * 100:.1f}%")

        report_str = "\n".join(report)
        print(report_str)

        # Save JSON report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(
                    [asdict(r) for r in results],
                    f,
                    indent=2
                )
            print(f"\nSaved detailed results to: {output_path}")

        return report_str


def main():
    """Main function for benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="Model Benchmark")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="weights",
        help="Directory containing models to benchmark"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path"
    )

    args = parser.parse_args()

    # Find all models
    models_dir = Path(args.models_dir)
    models = {}

    for ext in ["*.onnx", "*.engine", "*.mlpackage", "*.pt"]:
        for path in models_dir.glob(ext):
            name = path.stem
            models[name] = str(path)

    if not models:
        print(f"No models found in {models_dir}")
        return

    print(f"Found {len(models)} models to benchmark")

    # Run benchmarks
    benchmark = ModelBenchmark(
        img_size=args.img_size,
        num_runs=args.runs
    )

    results = benchmark.run_all(models)

    # Generate report
    benchmark.generate_report(results, args.output)


if __name__ == "__main__":
    main()

"""
Comprehensive Benchmark Suite for Object Detection Models.
Tests accuracy, speed, and memory across different conditions.
"""

import time
import json
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Warning: ultralytics not installed")


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    model_size: str
    input_size: int
    batch_size: int
    device: str
    precision: str

    # Accuracy metrics
    map50: Optional[float] = None
    map50_95: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None

    # Speed metrics (ms)
    inference_mean_ms: float = 0.0
    inference_std_ms: float = 0.0
    inference_p50_ms: float = 0.0
    inference_p95_ms: float = 0.0
    inference_p99_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    fps: float = 0.0

    # Memory metrics (MB)
    model_size_mb: float = 0.0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0

    # Export formats
    export_formats: Dict[str, float] = None  # format -> size in MB

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = {}


class BenchmarkSuite:
    """
    Comprehensive benchmark suite for object detection models.
    Tests various model sizes, input resolutions, and batch sizes.
    """

    MODEL_VARIANTS = ["n", "s", "m", "l", "x"]
    INPUT_SIZES = [320, 416, 640, 1280]
    BATCH_SIZES = [1, 4, 8, 16, 32]
    EXPORT_FORMATS = ["onnx", "torchscript"]

    def __init__(
        self,
        output_dir: str = "benchmark_results",
        n_warmup: int = 10,
        n_iterations: int = 100
    ):
        """
        Initialize the benchmark suite.

        Args:
            output_dir: Directory to save results
            n_warmup: Number of warmup iterations
            n_iterations: Number of benchmark iterations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_warmup = n_warmup
        self.n_iterations = n_iterations
        self.results: List[BenchmarkResult] = []

    def _get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _get_memory_mb(self, device: str) -> float:
        """Get current memory usage in MB."""
        if device == "cuda":
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0

    def _clear_memory(self, device: str) -> None:
        """Clear GPU memory."""
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    def benchmark_inference_speed(
        self,
        model,
        input_size: int,
        batch_size: int,
        device: str,
        half: bool = False
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            model: YOLO model
            input_size: Input image size
            batch_size: Batch size
            device: Device to run on
            half: Use FP16

        Returns:
            Dict with timing metrics
        """
        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, input_size, input_size)
        if half and device != "cpu":
            dummy_input = dummy_input.half()
        dummy_input = dummy_input.to(device)

        # Warmup
        for _ in range(self.n_warmup):
            with torch.no_grad():
                _ = model.model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(self.n_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            with torch.no_grad():
                _ = model.model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        times = np.array(times)

        return {
            "inference_mean_ms": float(np.mean(times)),
            "inference_std_ms": float(np.std(times)),
            "inference_p50_ms": float(np.percentile(times, 50)),
            "inference_p95_ms": float(np.percentile(times, 95)),
            "inference_p99_ms": float(np.percentile(times, 99)),
            "fps": float(1000 / np.mean(times) * batch_size)
        }

    def benchmark_memory(
        self,
        model,
        input_size: int,
        batch_size: int,
        device: str,
        half: bool = False
    ) -> Dict[str, float]:
        """
        Benchmark memory usage.

        Args:
            model: YOLO model
            input_size: Input image size
            batch_size: Batch size
            device: Device to run on
            half: Use FP16

        Returns:
            Dict with memory metrics
        """
        if device != "cuda":
            return {"peak_memory_mb": 0.0, "avg_memory_mb": 0.0}

        self._clear_memory(device)
        torch.cuda.reset_peak_memory_stats()

        dummy_input = torch.randn(batch_size, 3, input_size, input_size)
        if half:
            dummy_input = dummy_input.half()
        dummy_input = dummy_input.to(device)

        memory_samples = []

        for _ in range(10):
            with torch.no_grad():
                _ = model.model(dummy_input)
            torch.cuda.synchronize()
            memory_samples.append(torch.cuda.memory_allocated() / 1024 / 1024)

        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        return {
            "peak_memory_mb": float(peak_memory),
            "avg_memory_mb": float(np.mean(memory_samples))
        }

    def benchmark_model_size(self, model) -> Dict[str, float]:
        """
        Benchmark model file size for different export formats.

        Args:
            model: YOLO model

        Returns:
            Dict mapping format to size in MB
        """
        sizes = {}

        # PyTorch model size
        if hasattr(model, 'ckpt_path') and model.ckpt_path:
            pt_path = Path(model.ckpt_path)
            if pt_path.exists():
                sizes["pytorch"] = pt_path.stat().st_size / 1024 / 1024

        return sizes

    def run_accuracy_benchmark(
        self,
        model,
        data_yaml: str,
        img_size: int = 640
    ) -> Dict[str, float]:
        """
        Run accuracy benchmark on validation set.

        Args:
            model: YOLO model
            data_yaml: Path to dataset YAML
            img_size: Input image size

        Returns:
            Dict with accuracy metrics
        """
        try:
            results = model.val(
                data=data_yaml,
                imgsz=img_size,
                verbose=False
            )

            return {
                "map50": float(results.box.map50),
                "map50_95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr)
            }
        except Exception as e:
            print(f"Accuracy benchmark failed: {e}")
            return {}

    def benchmark_single_config(
        self,
        model_variant: str,
        input_size: int,
        batch_size: int,
        device: str,
        half: bool = False,
        data_yaml: Optional[str] = None
    ) -> BenchmarkResult:
        """
        Benchmark a single configuration.

        Args:
            model_variant: Model size variant (n, s, m, l, x)
            input_size: Input image size
            batch_size: Batch size
            device: Device to run on
            half: Use FP16
            data_yaml: Optional path to dataset for accuracy benchmark

        Returns:
            BenchmarkResult object
        """
        print(f"  Benchmarking YOLOv8{model_variant} @ {input_size}x{input_size}, batch={batch_size}")

        # Load model
        model = YOLO(f"yolov8{model_variant}.pt")
        if half and device != "cpu":
            model.model.half()
        model.to(device)

        # Create result object
        result = BenchmarkResult(
            model_name=f"YOLOv8{model_variant}",
            model_size=model_variant,
            input_size=input_size,
            batch_size=batch_size,
            device=device,
            precision="FP16" if half else "FP32"
        )

        try:
            # Speed benchmark
            speed_metrics = self.benchmark_inference_speed(
                model, input_size, batch_size, device, half
            )
            result.inference_mean_ms = speed_metrics["inference_mean_ms"]
            result.inference_std_ms = speed_metrics["inference_std_ms"]
            result.inference_p50_ms = speed_metrics["inference_p50_ms"]
            result.inference_p95_ms = speed_metrics["inference_p95_ms"]
            result.inference_p99_ms = speed_metrics["inference_p99_ms"]
            result.fps = speed_metrics["fps"]

            # Memory benchmark
            if device == "cuda":
                memory_metrics = self.benchmark_memory(
                    model, input_size, batch_size, device, half
                )
                result.peak_memory_mb = memory_metrics["peak_memory_mb"]
                result.avg_memory_mb = memory_metrics["avg_memory_mb"]

            # Model size
            size_metrics = self.benchmark_model_size(model)
            result.model_size_mb = size_metrics.get("pytorch", 0.0)

            # Accuracy benchmark (if dataset provided)
            if data_yaml and batch_size == 1:  # Only run accuracy on batch=1
                accuracy_metrics = self.run_accuracy_benchmark(model, data_yaml, input_size)
                result.map50 = accuracy_metrics.get("map50")
                result.map50_95 = accuracy_metrics.get("map50_95")

        except Exception as e:
            print(f"    Error: {e}")

        finally:
            # Cleanup
            del model
            self._clear_memory(device)

        return result

    def run_full_benchmark(
        self,
        model_variants: List[str] = None,
        input_sizes: List[int] = None,
        batch_sizes: List[int] = None,
        data_yaml: Optional[str] = None,
        half: bool = False
    ) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark across all configurations.

        Args:
            model_variants: List of model variants to test
            input_sizes: List of input sizes to test
            batch_sizes: List of batch sizes to test
            data_yaml: Optional path to dataset for accuracy benchmark
            half: Use FP16 precision

        Returns:
            List of BenchmarkResult objects
        """
        if model_variants is None:
            model_variants = self.MODEL_VARIANTS
        if input_sizes is None:
            input_sizes = [640]  # Default to standard size
        if batch_sizes is None:
            batch_sizes = [1, 4, 8]

        device = self._get_device()
        print(f"Running benchmarks on: {device}")
        print(f"Model variants: {model_variants}")
        print(f"Input sizes: {input_sizes}")
        print(f"Batch sizes: {batch_sizes}")
        print()

        results = []
        total_configs = len(model_variants) * len(input_sizes) * len(batch_sizes)

        with tqdm(total=total_configs, desc="Benchmarking") as pbar:
            for variant in model_variants:
                for input_size in input_sizes:
                    for batch_size in batch_sizes:
                        result = self.benchmark_single_config(
                            model_variant=variant,
                            input_size=input_size,
                            batch_size=batch_size,
                            device=device,
                            half=half,
                            data_yaml=data_yaml
                        )
                        results.append(result)
                        pbar.update(1)

        self.results = results
        return results

    def run_latency_vs_accuracy_benchmark(
        self,
        data_yaml: str,
        input_sizes: List[int] = None
    ) -> List[Dict]:
        """
        Benchmark latency vs accuracy tradeoff for different model sizes.

        Args:
            data_yaml: Path to dataset YAML
            input_sizes: Input sizes to test

        Returns:
            List of results showing latency-accuracy tradeoff
        """
        if input_sizes is None:
            input_sizes = [320, 640]

        device = self._get_device()
        results = []

        for variant in self.MODEL_VARIANTS:
            for input_size in input_sizes:
                print(f"Testing YOLOv8{variant} @ {input_size}...")

                model = YOLO(f"yolov8{variant}.pt")
                model.to(device)

                # Speed
                speed = self.benchmark_inference_speed(model, input_size, 1, device)

                # Accuracy
                accuracy = self.run_accuracy_benchmark(model, data_yaml, input_size)

                results.append({
                    "model": f"YOLOv8{variant}",
                    "input_size": input_size,
                    "latency_ms": speed["inference_mean_ms"],
                    "fps": speed["fps"],
                    "mAP50": accuracy.get("map50", 0),
                    "mAP50-95": accuracy.get("map50_95", 0)
                })

                del model
                self._clear_memory(device)

        return results

    def save_results(self, filename: str = "benchmark_results.json") -> str:
        """
        Save benchmark results to JSON file.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename

        results_data = [asdict(r) for r in self.results]

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"Results saved to: {output_path}")
        return str(output_path)

    def generate_report(self) -> str:
        """
        Generate markdown report from benchmark results.

        Returns:
            Markdown formatted report
        """
        if not self.results:
            return "No benchmark results available."

        report = """# Object Detection Benchmark Report

## Overview

This report contains benchmark results for YOLOv8 object detection models
tested across different configurations.

## Speed Benchmarks (batch_size=1)

| Model | Input Size | Inference (ms) | FPS | Memory (MB) |
|-------|------------|----------------|-----|-------------|
"""
        # Filter for batch_size=1 results
        speed_results = [r for r in self.results if r.batch_size == 1]

        for r in sorted(speed_results, key=lambda x: (x.model_size, x.input_size)):
            report += f"| {r.model_name} | {r.input_size} | "
            report += f"{r.inference_mean_ms:.2f}±{r.inference_std_ms:.2f} | "
            report += f"{r.fps:.1f} | {r.peak_memory_mb:.1f} |\n"

        # Batch size scaling
        report += "\n## Batch Size Scaling\n\n"
        report += "| Model | Batch Size | FPS | Latency/Image (ms) |\n"
        report += "|-------|------------|-----|--------------------|\n"

        for r in sorted(self.results, key=lambda x: (x.model_size, x.batch_size)):
            if r.input_size == 640:  # Only show standard size
                latency_per_img = r.inference_mean_ms / r.batch_size
                report += f"| {r.model_name} | {r.batch_size} | {r.fps:.1f} | {latency_per_img:.2f} |\n"

        # Accuracy results (if available)
        accuracy_results = [r for r in self.results if r.map50 is not None]
        if accuracy_results:
            report += "\n## Accuracy Metrics\n\n"
            report += "| Model | Input Size | mAP50 | mAP50-95 |\n"
            report += "|-------|------------|-------|----------|\n"

            for r in sorted(accuracy_results, key=lambda x: x.model_size):
                report += f"| {r.model_name} | {r.input_size} | "
                report += f"{r.map50:.4f} | {r.map50_95:.4f} |\n"

        # Recommendations
        report += """
## Recommendations

### For Real-time Applications (>30 FPS)
"""
        realtime = [r for r in speed_results if r.fps > 30]
        if realtime:
            best_realtime = max(realtime, key=lambda x: x.map50 or 0)
            report += f"- Recommended: **{best_realtime.model_name}** at {best_realtime.input_size}x{best_realtime.input_size}\n"
            report += f"- Achieves {best_realtime.fps:.1f} FPS with {best_realtime.inference_mean_ms:.1f}ms latency\n"

        report += "\n### For High Accuracy"
        high_acc = [r for r in accuracy_results if r.map50 and r.map50 > 0.5]
        if high_acc:
            best_acc = max(high_acc, key=lambda x: x.map50)
            report += f"\n- Recommended: **{best_acc.model_name}** at {best_acc.input_size}x{best_acc.input_size}\n"
            report += f"- Achieves mAP50: {best_acc.map50:.4f}\n"

        report += "\n### For Edge Deployment"
        edge = [r for r in speed_results if 'n' in r.model_size or 's' in r.model_size]
        if edge:
            best_edge = min(edge, key=lambda x: x.inference_mean_ms)
            report += f"\n- Recommended: **{best_edge.model_name}**\n"
            report += f"- Smallest model with {best_edge.inference_mean_ms:.1f}ms latency\n"

        return report

    def save_report(self, filename: str = "benchmark_report.md") -> str:
        """Save report to file."""
        output_path = self.output_dir / filename
        report = self.generate_report()

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"Report saved to: {output_path}")
        return str(output_path)


def run_benchmarks(
    output_dir: str = "benchmark_results",
    data_yaml: Optional[str] = None,
    quick: bool = False
):
    """
    Run full benchmark suite.

    Args:
        output_dir: Directory to save results
        data_yaml: Optional path to dataset for accuracy benchmarks
        quick: If True, run quick benchmark (fewer configurations)
    """
    print("=" * 60)
    print("OBJECT DETECTION BENCHMARK SUITE")
    print("=" * 60)

    suite = BenchmarkSuite(output_dir=output_dir)

    if quick:
        # Quick benchmark
        results = suite.run_full_benchmark(
            model_variants=["n", "m", "x"],
            input_sizes=[640],
            batch_sizes=[1, 8],
            data_yaml=data_yaml
        )
    else:
        # Full benchmark
        results = suite.run_full_benchmark(
            data_yaml=data_yaml
        )

    # Save results
    suite.save_results()
    suite.save_report()

    print("\nBenchmark complete!")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run object detection benchmarks")
    parser.add_argument("--output", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset YAML for accuracy benchmarks")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")

    args = parser.parse_args()

    run_benchmarks(
        output_dir=args.output,
        data_yaml=args.data,
        quick=args.quick
    )

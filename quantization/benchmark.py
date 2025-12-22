"""
Comprehensive Model Benchmarking Suite

Industry-standard benchmarking for:
- Latency (mean, p50, p95, p99)
- Throughput (FPS)
- Model size
- Memory usage
- Accuracy metrics (mAP)

Supports: PyTorch, ONNX Runtime, CoreML, TensorRT
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np
import time
import json
import platform
import psutil
from datetime import datetime
from tabulate import tabulate
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import torch

console = Console()


@dataclass
class SystemInfo:
    """System hardware information."""
    platform: str
    processor: str
    python_version: str
    torch_version: str
    device: str
    cpu_count: int
    memory_gb: float
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Benchmark result for a single model."""
    model_name: str
    format: str
    precision: str
    size_mb: float

    # Latency metrics (ms)
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_min: float
    latency_max: float

    # Throughput
    throughput_fps: float

    # Memory
    memory_peak_mb: float

    # Metadata
    num_runs: int
    warmup_runs: int
    input_size: Tuple[int, int, int, int]
    device: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ModelBenchmark:
    """
    Unified benchmark runner for all model formats.
    """

    def __init__(
        self,
        img_size: int = 640,
        num_runs: int = 100,
        warmup: int = 20,
        batch_size: int = 1
    ):
        self.img_size = img_size
        self.num_runs = num_runs
        self.warmup = warmup
        self.batch_size = batch_size
        self.input_shape = (batch_size, 3, img_size, img_size)

        self.system_info = self._get_system_info()
        self._print_system_info()

    def _get_system_info(self) -> SystemInfo:
        """Collect system information."""
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        elif torch.backends.mps.is_available():
            device = "mps"
            gpu_name = "Apple Silicon"
            gpu_memory = None  # MPS shares system memory
        else:
            device = "cpu"
            gpu_name = None
            gpu_memory = None

        return SystemInfo(
            platform=f"{platform.system()} {platform.release()}",
            processor=platform.processor() or platform.machine(),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            device=device,
            cpu_count=psutil.cpu_count(logical=False),
            memory_gb=psutil.virtual_memory().total / 1e9,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory
        )

    def _print_system_info(self):
        """Print system information."""
        console.print("\n[bold blue]System Information[/bold blue]")
        console.print(f"  Platform: {self.system_info.platform}")
        console.print(f"  Processor: {self.system_info.processor}")
        console.print(f"  CPU Cores: {self.system_info.cpu_count}")
        console.print(f"  Memory: {self.system_info.memory_gb:.1f} GB")
        console.print(f"  PyTorch: {self.system_info.torch_version}")
        console.print(f"  Device: {self.system_info.device}")
        if self.system_info.gpu_name:
            console.print(f"  GPU: {self.system_info.gpu_name}")
        console.print()

    def _get_model_size(self, path: str) -> float:
        """Get model size in MB."""
        path = Path(path)
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        elif path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            return total / (1024 * 1024)
        return 0

    def _compute_statistics(self, times: List[float]) -> Dict[str, float]:
        """Compute latency statistics."""
        times = np.array(times)
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p50": float(np.percentile(times, 50)),
            "p95": float(np.percentile(times, 95)),
            "p99": float(np.percentile(times, 99)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
        }

    def benchmark_pytorch(
        self,
        model_path: str,
        device: Optional[str] = None
    ) -> BenchmarkResult:
        """Benchmark PyTorch model."""
        from ultralytics import YOLO

        console.print(f"[yellow]Benchmarking PyTorch:[/yellow] {model_path}")

        device = device or self.system_info.device
        model = YOLO(model_path)

        # Create input tensor
        dummy = torch.randn(*self.input_shape)
        if device == "cuda":
            dummy = dummy.cuda()
            model.model = model.model.cuda()
        elif device == "mps":
            dummy = dummy.to("mps")
            model.model = model.model.to("mps")

        # Warmup
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Warming up...", total=None)
            for _ in range(self.warmup):
                with torch.no_grad():
                    model.model(dummy)
                if device == "cuda":
                    torch.cuda.synchronize()

        # Benchmark
        times = []
        peak_memory = 0

        for _ in range(self.num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()
            with torch.no_grad():
                model.model(dummy)

            if device == "cuda":
                torch.cuda.synchronize()
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated() / 1e6)

            times.append((time.perf_counter() - start) * 1000)

        stats = self._compute_statistics(times)

        return BenchmarkResult(
            model_name=Path(model_path).stem,
            format="PyTorch",
            precision="FP32",
            size_mb=self._get_model_size(model_path),
            latency_mean=stats["mean"],
            latency_std=stats["std"],
            latency_p50=stats["p50"],
            latency_p95=stats["p95"],
            latency_p99=stats["p99"],
            latency_min=stats["min"],
            latency_max=stats["max"],
            throughput_fps=1000 / stats["mean"],
            memory_peak_mb=peak_memory,
            num_runs=self.num_runs,
            warmup_runs=self.warmup,
            input_size=self.input_shape,
            device=device
        )

    def benchmark_onnx(
        self,
        model_path: str,
        precision: str = "FP32"
    ) -> BenchmarkResult:
        """Benchmark ONNX Runtime model."""
        import onnxruntime as ort

        console.print(f"[yellow]Benchmarking ONNX:[/yellow] {model_path}")

        # Select providers
        providers = ['CPUExecutionProvider']
        device = "cpu"

        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            device = "cuda"
        elif 'CoreMLExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CoreMLExecutionProvider')
            device = "coreml"

        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name

        dummy = np.random.randn(*self.input_shape).astype(np.float32)

        # Warmup
        for _ in range(self.warmup):
            session.run(None, {input_name: dummy})

        # Benchmark
        times = []
        for _ in range(self.num_runs):
            start = time.perf_counter()
            session.run(None, {input_name: dummy})
            times.append((time.perf_counter() - start) * 1000)

        stats = self._compute_statistics(times)

        return BenchmarkResult(
            model_name=Path(model_path).stem,
            format="ONNX",
            precision=precision,
            size_mb=self._get_model_size(model_path),
            latency_mean=stats["mean"],
            latency_std=stats["std"],
            latency_p50=stats["p50"],
            latency_p95=stats["p95"],
            latency_p99=stats["p99"],
            latency_min=stats["min"],
            latency_max=stats["max"],
            throughput_fps=1000 / stats["mean"],
            memory_peak_mb=0,  # ONNX Runtime doesn't expose this easily
            num_runs=self.num_runs,
            warmup_runs=self.warmup,
            input_size=self.input_shape,
            device=device
        )

    def benchmark_coreml(
        self,
        model_path: str,
        precision: str = "FP16"
    ) -> Optional[BenchmarkResult]:
        """Benchmark CoreML model (Apple Silicon only)."""
        try:
            import coremltools as ct
        except ImportError:
            console.print("[red]CoreML not available[/red]")
            return None

        console.print(f"[yellow]Benchmarking CoreML:[/yellow] {model_path}")

        model = ct.models.MLModel(model_path)
        dummy = np.random.randn(*self.input_shape).astype(np.float32)

        # Warmup
        for _ in range(self.warmup):
            model.predict({"images": dummy})

        # Benchmark
        times = []
        for _ in range(self.num_runs):
            start = time.perf_counter()
            model.predict({"images": dummy})
            times.append((time.perf_counter() - start) * 1000)

        stats = self._compute_statistics(times)

        return BenchmarkResult(
            model_name=Path(model_path).stem,
            format="CoreML",
            precision=precision,
            size_mb=self._get_model_size(model_path),
            latency_mean=stats["mean"],
            latency_std=stats["std"],
            latency_p50=stats["p50"],
            latency_p95=stats["p95"],
            latency_p99=stats["p99"],
            latency_min=stats["min"],
            latency_max=stats["max"],
            throughput_fps=1000 / stats["mean"],
            memory_peak_mb=0,
            num_runs=self.num_runs,
            warmup_runs=self.warmup,
            input_size=self.input_shape,
            device="ane+gpu"
        )

    def run_all(self, models_dir: str = "weights") -> List[BenchmarkResult]:
        """Run benchmarks on all models in directory."""
        models_dir = Path(models_dir)
        results = []

        # Find all models
        model_files = {
            "pt": list(models_dir.glob("*.pt")),
            "onnx": list(models_dir.glob("*.onnx")),
            "mlpackage": list(models_dir.glob("*.mlpackage")),
        }

        console.print(f"\n[bold]Found models:[/bold]")
        for fmt, files in model_files.items():
            if files:
                console.print(f"  {fmt}: {[f.name for f in files]}")

        # Benchmark each
        for pt_file in model_files["pt"]:
            try:
                result = self.benchmark_pytorch(str(pt_file))
                results.append(result)
            except Exception as e:
                console.print(f"[red]Error benchmarking {pt_file}: {e}[/red]")

        for onnx_file in model_files["onnx"]:
            try:
                precision = "INT8" if "int8" in onnx_file.name.lower() else "FP32"
                result = self.benchmark_onnx(str(onnx_file), precision)
                results.append(result)
            except Exception as e:
                console.print(f"[red]Error benchmarking {onnx_file}: {e}[/red]")

        for coreml_file in model_files["mlpackage"]:
            try:
                result = self.benchmark_coreml(str(coreml_file))
                if result:
                    results.append(result)
            except Exception as e:
                console.print(f"[red]Error benchmarking {coreml_file}: {e}[/red]")

        return results

    def generate_report(
        self,
        results: List[BenchmarkResult],
        output_path: Optional[str] = None
    ) -> str:
        """Generate benchmark report."""
        if not results:
            return "No benchmark results to report."

        # Sort by latency
        results = sorted(results, key=lambda x: x.latency_mean)

        # Create rich table
        table = Table(title="Model Benchmark Results", show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Format", style="magenta")
        table.add_column("Precision", style="green")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("P95 (ms)", justify="right")
        table.add_column("FPS", justify="right", style="yellow")
        table.add_column("Device", style="blue")

        for r in results:
            table.add_row(
                r.model_name[:20],
                r.format,
                r.precision,
                f"{r.size_mb:.1f}",
                f"{r.latency_mean:.2f}",
                f"{r.latency_p95:.2f}",
                f"{r.throughput_fps:.1f}",
                r.device
            )

        console.print(table)

        # Summary
        if len(results) >= 2:
            baseline = results[-1]
            fastest = results[0]

            console.print(f"\n[bold green]Summary:[/bold green]")
            console.print(f"  Fastest: {fastest.model_name} ({fastest.format} {fastest.precision})")
            console.print(f"  Speedup vs baseline: {baseline.latency_mean / fastest.latency_mean:.2f}x")
            if baseline.size_mb > 0:
                console.print(f"  Size reduction: {(1 - fastest.size_mb / baseline.size_mb) * 100:.1f}%")

        # Save JSON report
        if output_path:
            report = {
                "system_info": asdict(self.system_info),
                "benchmark_config": {
                    "num_runs": self.num_runs,
                    "warmup_runs": self.warmup,
                    "input_size": self.input_shape,
                },
                "results": [asdict(r) for r in results],
                "generated_at": datetime.now().isoformat()
            }

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            console.print(f"\n[dim]Saved detailed report to: {output_path}[/dim]")

        return "Benchmark complete!"


def main():
    """Main benchmark entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Model Benchmark Suite")
    parser.add_argument("--models-dir", type=str, default="weights", help="Models directory")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup runs")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON path")

    args = parser.parse_args()

    benchmark = ModelBenchmark(
        img_size=args.img_size,
        num_runs=args.runs,
        warmup=args.warmup
    )

    results = benchmark.run_all(args.models_dir)
    benchmark.generate_report(results, args.output)


if __name__ == "__main__":
    main()

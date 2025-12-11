"""
Train YOLOv8 model on BDD100K dataset.
"""

from pathlib import Path
from typing import Dict, Optional, Any
import yaml
import torch
from ultralytics import YOLO


class ADASTrainer:
    """
    YOLOv8 trainer for ADAS object detection.
    """

    MODEL_SIZES = {
        "n": {"params": "3.2M", "speed": "fastest", "use_case": "edge/mobile"},
        "s": {"params": "11.2M", "speed": "fast", "use_case": "balanced"},
        "m": {"params": "25.9M", "speed": "medium", "use_case": "default"},
        "l": {"params": "43.7M", "speed": "slow", "use_case": "high accuracy"},
        "x": {"params": "68.2M", "speed": "slowest", "use_case": "best accuracy"},
    }

    def __init__(
        self,
        model_size: str = "m",
        pretrained: bool = True,
        device: str = "auto"
    ):
        """
        Initialize YOLOv8 trainer.

        Args:
            model_size: Model size variant (n, s, m, l, x)
            pretrained: Whether to use pretrained COCO weights
            device: Device to train on ('auto', 'cpu', 'cuda:0', 'mps')
        """
        if model_size not in self.MODEL_SIZES:
            raise ValueError(
                f"Invalid model size. Choose from: {list(self.MODEL_SIZES.keys())}"
            )

        self.model_size = model_size
        self.device = self._get_device(device)

        # Load model
        model_name = f"yolov8{model_size}.pt" if pretrained else f"yolov8{model_size}.yaml"
        self.model = YOLO(model_name)

        print(f"Loaded YOLOv8{model_size} model")
        print(f"  Parameters: {self.MODEL_SIZES[model_size]['params']}")
        print(f"  Device: {self.device}")

    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device != "auto":
            return device

        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        img_size: int = 640,
        patience: int = 20,
        project: str = "runs/train",
        name: str = "yolov8m-bdd100k",
        resume: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            data_yaml: Path to dataset configuration YAML
            epochs: Number of training epochs
            batch_size: Batch size (adjust based on GPU memory)
            img_size: Input image size
            patience: Early stopping patience
            project: Project directory for saving results
            name: Experiment name
            resume: Whether to resume from last checkpoint
            **kwargs: Additional arguments passed to YOLO.train()

        Returns:
            Dict with training results
        """
        # Default training configuration
        train_args = {
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": img_size,
            "patience": patience,
            "device": self.device,
            "project": project,
            "name": name,
            "exist_ok": True,
            "pretrained": True,
            "verbose": True,

            # Optimizer settings
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "warmup_epochs": 5,
            "warmup_momentum": 0.8,
            "cos_lr": True,

            # Loss weights
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,

            # Augmentation (built into YOLOv8)
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        }

        # Override with user-provided kwargs
        train_args.update(kwargs)

        if resume:
            train_args["resume"] = True

        print(f"\nStarting training...")
        print(f"  Dataset: {data_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {img_size}")
        print()

        # Train
        results = self.model.train(**train_args)

        return results

    def evaluate(
        self,
        data_yaml: str,
        split: str = "val",
        batch_size: int = 16,
        img_size: int = 640,
        conf: float = 0.001,
        iou: float = 0.6,
        save_json: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on validation/test set.

        Args:
            data_yaml: Path to dataset configuration YAML
            split: Dataset split to evaluate ('val' or 'test')
            batch_size: Batch size for evaluation
            img_size: Input image size
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save_json: Whether to save results as JSON

        Returns:
            Dict with evaluation metrics
        """
        results = self.model.val(
            data=data_yaml,
            split=split,
            batch=batch_size,
            imgsz=img_size,
            conf=conf,
            iou=iou,
            save_json=save_json,
            device=self.device,
            verbose=True
        )

        # Extract key metrics
        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
            "per_class_ap50": dict(zip(
                results.names.values(),
                results.box.ap50.tolist()
            )) if hasattr(results.box, 'ap50') else {}
        }

        return metrics

    def export(
        self,
        format: str = "onnx",
        img_size: int = 640,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 12
    ) -> str:
        """
        Export model to specified format.

        Args:
            format: Export format ('onnx', 'torchscript', 'engine', 'coreml', etc.)
            img_size: Input image size
            half: Use FP16 half precision
            dynamic: Enable dynamic input shapes
            simplify: Simplify ONNX model
            opset: ONNX opset version

        Returns:
            Path to exported model
        """
        print(f"\nExporting model to {format}...")

        export_args = {
            "format": format,
            "imgsz": img_size,
            "half": half,
            "dynamic": dynamic,
        }

        if format == "onnx":
            export_args["simplify"] = simplify
            export_args["opset"] = opset

        path = self.model.export(**export_args)

        print(f"Exported to: {path}")
        return str(path)

    def predict(
        self,
        source: str,
        conf: float = 0.25,
        iou: float = 0.45,
        save: bool = True,
        show: bool = False
    ):
        """
        Run inference on images/video.

        Args:
            source: Path to image, video, or directory
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Save annotated results
            show: Display results (requires GUI)

        Returns:
            Prediction results
        """
        results = self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            save=save,
            show=show,
            device=self.device
        )

        return results

    def benchmark(
        self,
        img_size: int = 640,
        batch_size: int = 1,
        half: bool = False
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            img_size: Input image size
            batch_size: Batch size
            half: Use FP16 half precision

        Returns:
            Dict with timing metrics
        """
        import time
        import numpy as np

        # Create dummy input
        dummy_input = torch.randn(batch_size, 3, img_size, img_size)
        if half:
            dummy_input = dummy_input.half()

        dummy_input = dummy_input.to(self.device)

        # Warmup
        for _ in range(10):
            _ = self.model.model(dummy_input)

        # Benchmark
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = self.model.model(dummy_input)
            if self.device != "cpu":
                torch.cuda.synchronize() if torch.cuda.is_available() else None
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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "auto"
    ) -> "ADASTrainer":
        """
        Load trainer from checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint
            device: Device to load model on

        Returns:
            ADASTrainer instance with loaded model
        """
        trainer = cls.__new__(cls)
        trainer.device = trainer._get_device(trainer, device)
        trainer.model = YOLO(checkpoint_path)
        trainer.model_size = "custom"

        print(f"Loaded checkpoint from {checkpoint_path}")
        return trainer


def main():
    """Main function for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8 on BDD100K")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/dataset.yaml",
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="m",
        choices=["n", "s", "m", "l", "x"],
        help="Model size variant"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cpu, cuda:0, mps)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export format after training (onnx, torchscript, etc.)"
    )

    args = parser.parse_args()

    # Create trainer
    trainer = ADASTrainer(
        model_size=args.model,
        device=args.device
    )

    # Train
    results = trainer.train(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        resume=args.resume
    )

    # Evaluate
    metrics = trainer.evaluate(data_yaml=args.data)
    print("\nEvaluation Results:")
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for cls_name, ap in v.items():
                print(f"    {cls_name}: {ap:.4f}")
        else:
            print(f"  {k}: {v:.4f}")

    # Export if requested
    if args.export:
        trainer.export(format=args.export)


if __name__ == "__main__":
    main()

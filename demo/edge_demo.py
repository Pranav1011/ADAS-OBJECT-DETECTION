"""
Edge Deployment Demo for Object Detection.
Simulates deployment on edge devices with optimized inference.
"""

import time
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
import queue

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    print("Warning: onnxruntime not installed")

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: ultralytics not installed")


@dataclass
class Detection:
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2

    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox
        }


@dataclass
class InferenceResult:
    """Container for inference results."""
    detections: List[Detection]
    inference_time_ms: float
    preprocess_time_ms: float
    postprocess_time_ms: float
    total_time_ms: float
    frame_shape: Tuple[int, int]


class EdgeOptimizedDetector:
    """
    Optimized object detector for edge deployment.
    Supports ONNX Runtime for efficient inference.
    """

    ADAS_CLASSES = {
        0: "car",
        1: "truck",
        2: "bus",
        3: "person",
        4: "bicycle",
        5: "motorcycle",
        6: "traffic_light",
        7: "traffic_sign",
        8: "train",
        9: "rider"
    }

    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_gpu: bool = False,
        num_threads: int = 4
    ):
        """
        Initialize the edge detector.

        Args:
            model_path: Path to model file (.pt, .onnx, or .engine)
            input_size: Model input size
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            use_gpu: Use GPU acceleration
            num_threads: Number of CPU threads
        """
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_gpu = use_gpu
        self.num_threads = num_threads

        self._model = None
        self._model_type = None

        self._load_model()

    def _load_model(self) -> None:
        """Load model based on file extension."""
        suffix = self.model_path.suffix.lower()

        if suffix == ".onnx" and ort is not None:
            self._load_onnx_model()
        elif suffix == ".pt" and YOLO is not None:
            self._load_pytorch_model()
        else:
            raise ValueError(f"Unsupported model format: {suffix}")

    def _load_onnx_model(self) -> None:
        """Load ONNX model with optimized settings."""
        # Session options for edge optimization
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = self.num_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Provider selection
        if self.use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self._model = ort.InferenceSession(
            str(self.model_path),
            sess_options,
            providers=providers
        )
        self._model_type = "onnx"

        # Get input/output names
        self._input_name = self._model.get_inputs()[0].name
        self._output_names = [o.name for o in self._model.get_outputs()]

        print(f"Loaded ONNX model from {self.model_path}")
        print(f"  Providers: {self._model.get_providers()}")

    def _load_pytorch_model(self) -> None:
        """Load PyTorch/Ultralytics model."""
        self._model = YOLO(str(self.model_path))
        self._model_type = "pytorch"
        print(f"Loaded PyTorch model from {self.model_path}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Preprocessed input tensor
        """
        # Resize
        img = cv2.resize(frame, (self.input_size, self.input_size))

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(
        self,
        outputs: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> List[Detection]:
        """
        Post-process model outputs.

        Args:
            outputs: Raw model outputs
            original_shape: Original image shape (H, W)

        Returns:
            List of Detection objects
        """
        # Handle different output formats
        if len(outputs.shape) == 3:
            # YOLOv8 format: (1, num_classes + 4, num_detections)
            outputs = outputs[0].T  # (num_detections, num_classes + 4)

        detections = []
        h_orig, w_orig = original_shape

        for pred in outputs:
            if len(pred) < 5:
                continue

            # Extract bbox and scores
            x_center, y_center, width, height = pred[:4]
            class_scores = pred[4:]

            # Get best class
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self.conf_threshold:
                continue

            # Convert to corner format and scale
            x1 = int((x_center - width / 2) * w_orig / self.input_size)
            y1 = int((y_center - height / 2) * h_orig / self.input_size)
            x2 = int((x_center + width / 2) * w_orig / self.input_size)
            y2 = int((y_center + height / 2) * h_orig / self.input_size)

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_orig, x2), min(h_orig, y2)

            class_name = self.ADAS_CLASSES.get(class_id, f"class_{class_id}")

            detections.append(Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2)
            ))

        # Apply NMS
        if detections:
            detections = self._nms(detections)

        return detections

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        # Group by class
        class_detections = {}
        for det in detections:
            if det.class_id not in class_detections:
                class_detections[det.class_id] = []
            class_detections[det.class_id].append(det)

        # Apply NMS per class
        final_detections = []
        for class_id, dets in class_detections.items():
            # Sort by confidence
            dets.sort(key=lambda x: x.confidence, reverse=True)

            keep = []
            while dets:
                best = dets.pop(0)
                keep.append(best)

                dets = [d for d in dets if self._iou(best.bbox, d.bbox) < self.iou_threshold]

            final_detections.extend(keep)

        return final_detections

    def _iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def detect(self, frame: np.ndarray) -> InferenceResult:
        """
        Run detection on a frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            InferenceResult with detections and timing
        """
        original_shape = frame.shape[:2]

        # Preprocess
        t_pre_start = time.perf_counter()
        input_tensor = self.preprocess(frame)
        preprocess_time = (time.perf_counter() - t_pre_start) * 1000

        # Inference
        t_inf_start = time.perf_counter()
        if self._model_type == "onnx":
            outputs = self._model.run(
                self._output_names,
                {self._input_name: input_tensor}
            )[0]
        else:
            # PyTorch inference
            results = self._model.predict(
                frame,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]
            # Convert to detection format
            return self._convert_pytorch_results(results, original_shape, preprocess_time)

        inference_time = (time.perf_counter() - t_inf_start) * 1000

        # Postprocess
        t_post_start = time.perf_counter()
        detections = self.postprocess(outputs, original_shape)
        postprocess_time = (time.perf_counter() - t_post_start) * 1000

        total_time = preprocess_time + inference_time + postprocess_time

        return InferenceResult(
            detections=detections,
            inference_time_ms=inference_time,
            preprocess_time_ms=preprocess_time,
            postprocess_time_ms=postprocess_time,
            total_time_ms=total_time,
            frame_shape=original_shape
        )

    def _convert_pytorch_results(
        self,
        results,
        original_shape: Tuple[int, int],
        preprocess_time: float
    ) -> InferenceResult:
        """Convert Ultralytics results to our format."""
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.ADAS_CLASSES.get(class_id, f"class_{class_id}")

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2))
                ))

        return InferenceResult(
            detections=detections,
            inference_time_ms=results.speed.get('inference', 0),
            preprocess_time_ms=preprocess_time,
            postprocess_time_ms=results.speed.get('postprocess', 0),
            total_time_ms=preprocess_time + results.speed.get('inference', 0) + results.speed.get('postprocess', 0),
            frame_shape=original_shape
        )

    def draw_detections(
        self,
        frame: np.ndarray,
        result: InferenceResult,
        show_fps: bool = True
    ) -> np.ndarray:
        """
        Draw detections on frame.

        Args:
            frame: Original frame
            result: Detection result
            show_fps: Whether to show FPS counter

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Color map for classes
        colors = {
            "car": (0, 255, 0),
            "truck": (0, 200, 0),
            "bus": (0, 150, 0),
            "person": (255, 0, 0),
            "bicycle": (255, 165, 0),
            "motorcycle": (255, 100, 0),
            "traffic_light": (255, 255, 0),
            "traffic_sign": (255, 200, 0),
        }

        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (128, 128, 128))

            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )

        # Draw FPS
        if show_fps:
            fps = 1000 / result.total_time_ms if result.total_time_ms > 0 else 0
            fps_text = f"FPS: {fps:.1f} | Latency: {result.total_time_ms:.1f}ms"
            cv2.putText(
                annotated, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        return annotated


class EdgeSimulator:
    """
    Simulates edge deployment scenarios with performance constraints.
    """

    EDGE_PROFILES = {
        "jetson_nano": {
            "name": "NVIDIA Jetson Nano",
            "target_fps": 15,
            "max_power_w": 10,
            "memory_mb": 4096,
            "recommended_model": "yolov8n",
            "recommended_size": 320
        },
        "raspberry_pi_4": {
            "name": "Raspberry Pi 4",
            "target_fps": 5,
            "max_power_w": 5,
            "memory_mb": 4096,
            "recommended_model": "yolov8n",
            "recommended_size": 320
        },
        "coral_edge_tpu": {
            "name": "Google Coral Edge TPU",
            "target_fps": 30,
            "max_power_w": 2,
            "memory_mb": 1024,
            "recommended_model": "yolov8n-tflite",
            "recommended_size": 320
        },
        "intel_ncs2": {
            "name": "Intel Neural Compute Stick 2",
            "target_fps": 20,
            "max_power_w": 1,
            "memory_mb": 512,
            "recommended_model": "yolov8n-openvino",
            "recommended_size": 416
        }
    }

    def __init__(self, profile: str = "jetson_nano"):
        """
        Initialize edge simulator.

        Args:
            profile: Edge device profile name
        """
        if profile not in self.EDGE_PROFILES:
            raise ValueError(f"Unknown profile. Choose from: {list(self.EDGE_PROFILES.keys())}")

        self.profile = self.EDGE_PROFILES[profile]
        print(f"Edge Simulator: {self.profile['name']}")
        print(f"  Target FPS: {self.profile['target_fps']}")
        print(f"  Recommended: {self.profile['recommended_model']} @ {self.profile['recommended_size']}")

    def simulate_latency(self, actual_latency_ms: float) -> float:
        """
        Simulate latency on edge device based on profile.

        Args:
            actual_latency_ms: Measured latency on current hardware

        Returns:
            Simulated latency on edge device
        """
        # Simple scaling factor (rough approximation)
        scale_factors = {
            "jetson_nano": 2.0,
            "raspberry_pi_4": 5.0,
            "coral_edge_tpu": 0.8,
            "intel_ncs2": 1.5
        }

        scale = scale_factors.get(self.profile.get("name", "").lower().replace(" ", "_"), 1.0)
        return actual_latency_ms * scale

    def check_constraints(self, result: InferenceResult) -> Dict:
        """
        Check if inference meets edge device constraints.

        Args:
            result: Inference result

        Returns:
            Dict with constraint check results
        """
        target_latency = 1000 / self.profile["target_fps"]
        simulated_latency = self.simulate_latency(result.total_time_ms)

        return {
            "target_fps": self.profile["target_fps"],
            "actual_fps": 1000 / simulated_latency if simulated_latency > 0 else 0,
            "target_latency_ms": target_latency,
            "simulated_latency_ms": simulated_latency,
            "meets_target": simulated_latency <= target_latency,
            "recommendations": self._get_recommendations(simulated_latency, target_latency)
        }

    def _get_recommendations(self, actual: float, target: float) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        if actual > target:
            ratio = actual / target
            if ratio > 2:
                recommendations.append("Consider using a smaller model (e.g., YOLOv8n)")
            if ratio > 1.5:
                recommendations.append("Reduce input resolution (320x320)")
            recommendations.append("Enable FP16/INT8 quantization")
            recommendations.append("Use hardware-optimized runtime (TensorRT, OpenVINO)")

        return recommendations


def run_edge_demo(
    model_path: str,
    source: str = "0",
    profile: str = "jetson_nano",
    save_output: bool = False,
    output_path: str = "edge_demo_output.mp4"
):
    """
    Run edge deployment demo.

    Args:
        model_path: Path to model file
        source: Video source (camera index, video file, or image path)
        profile: Edge device profile
        save_output: Whether to save output video
        output_path: Output video path
    """
    print("=" * 60)
    print("EDGE DEPLOYMENT DEMO")
    print("=" * 60)

    # Initialize detector
    detector = EdgeOptimizedDetector(
        model_path=model_path,
        input_size=320 if "nano" in profile or "raspberry" in profile else 640
    )

    # Initialize simulator
    simulator = EdgeSimulator(profile=profile)

    # Open video source
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Failed to open source: {source}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Video writer
    writer = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Stats
    frame_count = 0
    total_time = 0
    latencies = []

    print(f"\nRunning demo on: {source}")
    print("Press 'q' to quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect
            result = detector.detect(frame)
            latencies.append(result.total_time_ms)

            # Check constraints
            constraints = simulator.check_constraints(result)

            # Draw results
            annotated = detector.draw_detections(frame, result)

            # Add edge simulation info
            info_text = f"Edge ({simulator.profile['name']}): {constraints['simulated_latency_ms']:.1f}ms"
            color = (0, 255, 0) if constraints['meets_target'] else (0, 0, 255)
            cv2.putText(
                annotated, info_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            # Display
            cv2.imshow("Edge Demo", annotated)

            # Save
            if writer:
                writer.write(annotated)

            frame_count += 1
            total_time += result.total_time_ms

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # Print summary
    if latencies:
        avg_latency = np.mean(latencies)
        avg_fps = 1000 / avg_latency

        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(f"Frames processed: {frame_count}")
        print(f"Average latency: {avg_latency:.1f}ms")
        print(f"Average FPS: {avg_fps:.1f}")
        print(f"Edge target FPS: {simulator.profile['target_fps']}")

        constraints = simulator.check_constraints(InferenceResult(
            detections=[],
            inference_time_ms=avg_latency,
            preprocess_time_ms=0,
            postprocess_time_ms=0,
            total_time_ms=avg_latency,
            frame_shape=(height, width)
        ))

        if constraints['meets_target']:
            print("\n✓ Meets edge deployment requirements!")
        else:
            print("\n✗ Does NOT meet edge deployment requirements")
            print("Recommendations:")
            for rec in constraints['recommendations']:
                print(f"  - {rec}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Edge Deployment Demo")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to model")
    parser.add_argument("--source", type=str, default="0", help="Video source")
    parser.add_argument("--profile", type=str, default="jetson_nano",
                       choices=list(EdgeSimulator.EDGE_PROFILES.keys()),
                       help="Edge device profile")
    parser.add_argument("--save", action="store_true", help="Save output video")
    parser.add_argument("--output", type=str, default="edge_demo_output.mp4", help="Output path")

    args = parser.parse_args()

    run_edge_demo(
        model_path=args.model,
        source=args.source,
        profile=args.profile,
        save_output=args.save,
        output_path=args.output
    )

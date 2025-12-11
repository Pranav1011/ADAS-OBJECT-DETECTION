"""
Unified object detector interface supporting multiple backends.

Supports:
- ONNX Runtime
- TensorRT
- CoreML
- PyTorch (Ultralytics)
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import cv2
import time


@dataclass
class Detection:
    """Single object detection result."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def box(self) -> Tuple[float, float, float, float]:
        """Return box as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        """Return box center (cx, cy)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        """Return box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name
        }


class ObjectDetector:
    """
    Unified object detector supporting multiple backends.
    """

    CLASS_NAMES = [
        "car", "truck", "bus", "pedestrian",
        "cyclist", "motorcycle", "traffic_light", "traffic_sign"
    ]

    def __init__(
        self,
        model_path: str,
        backend: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640
    ):
        """
        Initialize detector.

        Args:
            model_path: Path to model file
            backend: Backend to use ("auto", "onnx", "tensorrt", "coreml", "pytorch")
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            img_size: Input image size
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size

        # Auto-detect backend from file extension
        if backend == "auto":
            backend = self._detect_backend()

        self.backend = backend
        self._load_model()

        print(f"ObjectDetector initialized")
        print(f"  Model: {self.model_path}")
        print(f"  Backend: {self.backend}")

    def _detect_backend(self) -> str:
        """Detect backend from model file extension."""
        suffix = self.model_path.suffix.lower()

        if suffix == ".onnx":
            return "onnx"
        elif suffix == ".engine":
            return "tensorrt"
        elif suffix == ".mlpackage" or self.model_path.is_dir():
            return "coreml"
        elif suffix == ".pt":
            return "pytorch"
        else:
            raise ValueError(f"Unknown model format: {suffix}")

    def _load_model(self):
        """Load model based on backend."""
        if self.backend == "onnx":
            self._load_onnx()
        elif self.backend == "tensorrt":
            self._load_tensorrt()
        elif self.backend == "coreml":
            self._load_coreml()
        elif self.backend == "pytorch":
            self._load_pytorch()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_onnx(self):
        """Load ONNX Runtime model."""
        import onnxruntime as ort

        # Select best available provider
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name

    def _load_tensorrt(self):
        """Load TensorRT engine."""
        from quantization.tensorrt_quantize import TensorRTInference
        self.engine = TensorRTInference(str(self.model_path))

    def _load_coreml(self):
        """Load CoreML model."""
        import coremltools as ct
        self.model = ct.models.MLModel(str(self.model_path))

    def _load_pytorch(self):
        """Load PyTorch model via Ultralytics."""
        from ultralytics import YOLO
        self.model = YOLO(str(self.model_path))

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for inference.

        Args:
            image: Input image (BGR, HWC)

        Returns:
            Tuple of (preprocessed_image, scale, original_size)
        """
        original_size = (image.shape[1], image.shape[0])  # (w, h)

        # Resize with letterbox
        img, scale, pad = self._letterbox(image, self.img_size)

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        # Add batch dimension
        img = np.expand_dims(img, 0)

        return img, scale, original_size, pad

    def _letterbox(
        self,
        img: np.ndarray,
        new_shape: int = 640,
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image with letterbox padding.

        Args:
            img: Input image
            new_shape: Target size
            color: Padding color

        Returns:
            Tuple of (resized_image, scale, padding)
        """
        shape = img.shape[:2]  # (h, w)

        # Calculate scale
        r = min(new_shape / shape[0], new_shape / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        # Padding
        dw = new_shape - new_unpad[0]
        dh = new_shape - new_unpad[1]
        dw /= 2
        dh /= 2

        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Add padding
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )

        return img, r, (dw, dh)

    def postprocess(
        self,
        output: np.ndarray,
        original_size: Tuple[int, int],
        scale: float,
        pad: Tuple[float, float]
    ) -> List[Detection]:
        """
        Postprocess model output to detections.

        Args:
            output: Model output
            original_size: Original image size (w, h)
            scale: Resize scale
            pad: Padding (dw, dh)

        Returns:
            List of Detection objects
        """
        # YOLOv8 output shape: [1, num_classes + 4, num_anchors]
        # Transpose to [num_anchors, num_classes + 4]
        if len(output.shape) == 3:
            output = output[0].T

        # Extract boxes and scores
        boxes = output[:, :4]  # x_center, y_center, width, height
        scores = output[:, 4:]  # class scores

        # Get best class for each box
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert to x1y1x2y2
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Remove padding and scale to original size
        boxes_xyxy[:, [0, 2]] -= pad[0]  # x
        boxes_xyxy[:, [1, 3]] -= pad[1]  # y
        boxes_xyxy /= scale

        # Clip to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, original_size[0])
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, original_size[1])

        # Apply NMS
        indices = self._nms(boxes_xyxy, confidences, self.iou_threshold)

        # Create Detection objects
        detections = []
        for i in indices:
            class_id = int(class_ids[i])
            class_name = self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else str(class_id)

            detections.append(Detection(
                x1=float(boxes_xyxy[i, 0]),
                y1=float(boxes_xyxy[i, 1]),
                x2=float(boxes_xyxy[i, 2]),
                y2=float(boxes_xyxy[i, 3]),
                confidence=float(confidences[i]),
                class_id=class_id,
                class_name=class_name
            ))

        return detections

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> List[int]:
        """
        Non-maximum suppression.

        Args:
            boxes: Boxes [N, 4] in x1y1x2y2 format
            scores: Confidence scores [N]
            iou_threshold: IoU threshold

        Returns:
            Indices of kept boxes
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Run detection on image.

        Args:
            image: Input image (BGR, HWC)

        Returns:
            List of Detection objects
        """
        # For PyTorch/Ultralytics, use built-in inference
        if self.backend == "pytorch":
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )[0]

            detections = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                detections.append(Detection(
                    x1=float(box.xyxy[0][0]),
                    y1=float(box.xyxy[0][1]),
                    x2=float(box.xyxy[0][2]),
                    y2=float(box.xyxy[0][3]),
                    confidence=float(box.conf[0]),
                    class_id=class_id,
                    class_name=self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else str(class_id)
                ))
            return detections

        # For other backends, use manual preprocessing/postprocessing
        preprocessed, scale, original_size, pad = self.preprocess(image)

        # Run inference
        if self.backend == "onnx":
            output = self.session.run(None, {self.input_name: preprocessed})[0]
        elif self.backend == "tensorrt":
            output = self.engine.infer(preprocessed)[0]
        elif self.backend == "coreml":
            output = self.model.predict({"images": preprocessed})
            # Extract the right output key
            output = list(output.values())[0]

        # Postprocess
        detections = self.postprocess(output, original_size, scale, pad)

        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """
        Run detection on batch of images.

        Args:
            images: List of input images

        Returns:
            List of detection lists
        """
        return [self.detect(img) for img in images]

    def benchmark(
        self,
        num_runs: int = 100,
        warmup: int = 10
    ) -> Dict:
        """
        Benchmark inference speed.

        Args:
            num_runs: Number of benchmark iterations
            warmup: Number of warmup iterations

        Returns:
            Dict with timing statistics
        """
        # Create dummy image
        dummy_image = np.random.randint(
            0, 255,
            (self.img_size, self.img_size, 3),
            dtype=np.uint8
        )

        # Warmup
        for _ in range(warmup):
            self.detect(dummy_image)

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self.detect(dummy_image)
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


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    line_thickness: int = 2,
    font_scale: float = 0.6
) -> np.ndarray:
    """
    Draw detections on image.

    Args:
        image: Input image (BGR)
        detections: List of Detection objects
        line_thickness: Box line thickness
        font_scale: Label font scale

    Returns:
        Annotated image
    """
    img = image.copy()

    # Colors for each class
    colors = [
        (255, 0, 0),    # car - blue
        (0, 128, 255),  # truck - orange
        (0, 255, 255),  # bus - yellow
        (0, 255, 0),    # pedestrian - green
        (255, 0, 255),  # cyclist - magenta
        (128, 0, 255),  # motorcycle - purple
        (0, 0, 255),    # traffic_light - red
        (255, 255, 0),  # traffic_sign - cyan
    ]

    for det in detections:
        color = colors[det.class_id % len(colors)]

        # Draw box
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_thickness)

        # Draw label
        label = f"{det.class_name} {det.confidence:.2f}"
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Label background
        cv2.rectangle(
            img,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )

        # Label text
        cv2.putText(
            img, label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1
        )

    return img


def main():
    """Demo detection on sample image."""
    import argparse

    parser = argparse.ArgumentParser(description="Object Detection Demo")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model file"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="detection_output.jpg",
        help="Path to output image"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark"
    )

    args = parser.parse_args()

    # Create detector
    detector = ObjectDetector(
        model_path=args.model,
        conf_threshold=args.conf
    )

    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image {args.image}")
        return

    # Run detection
    start = time.perf_counter()
    detections = detector.detect(image)
    elapsed = (time.perf_counter() - start) * 1000

    print(f"\nDetected {len(detections)} objects in {elapsed:.2f}ms")
    for det in detections:
        print(f"  {det.class_name}: {det.confidence:.2f} at ({det.x1:.0f}, {det.y1:.0f}, {det.x2:.0f}, {det.y2:.0f})")

    # Draw and save
    output_image = draw_detections(image, detections)
    cv2.imwrite(args.output, output_image)
    print(f"\nSaved output to {args.output}")

    # Benchmark if requested
    if args.benchmark:
        print("\nRunning benchmark...")
        results = detector.benchmark()
        print(f"  Mean: {results['mean_ms']:.2f}ms")
        print(f"  P95: {results['p95_ms']:.2f}ms")
        print(f"  FPS: {results['fps']:.1f}")


if __name__ == "__main__":
    main()

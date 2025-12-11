"""
FastAPI service for ADAS object detection.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import cv2
import io
import time
from pathlib import Path

# Import detector
import sys
sys.path.append(str(Path(__file__).parent.parent))
from inference.detector import ObjectDetector, Detection, draw_detections

# Initialize FastAPI app
app = FastAPI(
    title="ADAS Object Detection API",
    description="Real-time object detection for autonomous driving applications",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector: Optional[ObjectDetector] = None

# Available models
AVAILABLE_MODELS = {
    "yolov8m-fp32": "weights/best.onnx",
    "yolov8m-int8": "weights/best_int8.onnx",
}

# Pydantic models for API
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class DetectionResponse(BaseModel):
    image_width: int
    image_height: int
    inference_time_ms: float
    model_name: str
    num_detections: int
    detections: List[BoundingBox]


class ModelInfo(BaseModel):
    name: str
    path: str
    available: bool
    precision: str


class BenchmarkResult(BaseModel):
    model_name: str
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    fps: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    current_model: Optional[str]


def get_detector(model_name: str = "yolov8m-fp32") -> ObjectDetector:
    """Get or create detector instance."""
    global detector

    model_path = AVAILABLE_MODELS.get(model_name)
    if model_path is None:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not found. Available: {list(AVAILABLE_MODELS.keys())}"
        )

    if not Path(model_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model file not found: {model_path}"
        )

    # Create new detector if needed
    if detector is None or detector.model_path != Path(model_path):
        detector = ObjectDetector(model_path=model_path)

    return detector


def read_image(file: UploadFile) -> np.ndarray:
    """Read image from upload."""
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image"
        )

    return image


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "ADAS Object Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=detector is not None,
        current_model=str(detector.model_path) if detector else None
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available detection models."""
    models = []
    for name, path in AVAILABLE_MODELS.items():
        precision = "int8" if "int8" in name else "fp32"
        models.append(ModelInfo(
            name=name,
            path=path,
            available=Path(path).exists(),
            precision=precision
        ))
    return models


@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(...),
    model: str = Query("yolov8m-fp32", description="Model to use"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold"),
    iou: float = Query(0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
):
    """
    Detect objects in uploaded image.

    Returns JSON with bounding boxes, confidence scores, and class labels.
    """
    # Get detector
    det = get_detector(model)
    det.conf_threshold = confidence
    det.iou_threshold = iou

    # Read image
    image = read_image(file)
    h, w = image.shape[:2]

    # Run detection
    start = time.perf_counter()
    detections = det.detect(image)
    inference_time = (time.perf_counter() - start) * 1000

    # Convert to response format
    boxes = [
        BoundingBox(
            x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
            confidence=d.confidence,
            class_id=d.class_id,
            class_name=d.class_name
        )
        for d in detections
    ]

    return DetectionResponse(
        image_width=w,
        image_height=h,
        inference_time_ms=inference_time,
        model_name=model,
        num_detections=len(boxes),
        detections=boxes
    )


@app.post("/detect/visualize")
async def detect_and_visualize(
    file: UploadFile = File(...),
    model: str = Query("yolov8m-fp32", description="Model to use"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold")
):
    """
    Detect objects and return annotated image.

    Returns JPEG image with bounding boxes drawn.
    """
    # Get detector
    det = get_detector(model)
    det.conf_threshold = confidence

    # Read image
    image = read_image(file)

    # Run detection
    detections = det.detect(image)

    # Draw boxes
    annotated = draw_detections(image, detections)

    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/jpeg",
        headers={
            "X-Detections-Count": str(len(detections)),
            "X-Model": model
        }
    )


@app.post("/detect/batch", response_model=List[DetectionResponse])
async def detect_batch(
    files: List[UploadFile] = File(...),
    model: str = Query("yolov8m-fp32", description="Model to use"),
    confidence: float = Query(0.25, ge=0.0, le=1.0, description="Confidence threshold")
):
    """
    Batch detection for multiple images.

    More efficient than individual requests for processing multiple images.
    """
    # Get detector
    det = get_detector(model)
    det.conf_threshold = confidence

    results = []
    for file in files:
        image = read_image(file)
        h, w = image.shape[:2]

        start = time.perf_counter()
        detections = det.detect(image)
        inference_time = (time.perf_counter() - start) * 1000

        boxes = [
            BoundingBox(
                x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name
            )
            for d in detections
        ]

        results.append(DetectionResponse(
            image_width=w,
            image_height=h,
            inference_time_ms=inference_time,
            model_name=model,
            num_detections=len(boxes),
            detections=boxes
        ))

    return results


@app.get("/benchmark", response_model=BenchmarkResult)
async def benchmark_model(
    model: str = Query("yolov8m-fp32", description="Model to benchmark"),
    runs: int = Query(100, ge=10, le=1000, description="Number of benchmark runs")
):
    """
    Benchmark model inference speed.

    Returns latency statistics and FPS.
    """
    det = get_detector(model)
    results = det.benchmark(num_runs=runs)

    return BenchmarkResult(
        model_name=model,
        **results
    )


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ADAS Detection API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")

    args = parser.parse_args()
    start_server(args.host, args.port)

"""Optional Python 3.8+ CPU object proposals from raw YOLOv8/11 ONNX heads.

No dependency imports, model downloads, or device access occur at import time.
Metadata schema 1 describes batch-one RGB / 255, fixed-shape letterboxing,
cxcywh input-pixel boxes, class probabilities, and no objectness or built-in NMS.
A proposal is an object bounding box, not evidence of strings or fret geometry.
Instances serialize OpenCV's stateful Net with a lock.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time


def _dependencies():
    import cv2
    import numpy as np
    return cv2, np


def _integer(value, name, low=1, high=8192):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError("{} must be an integer in [{}, {}]".format(name, low, high))
    return value


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be finite and in [0, 1]".format(name))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("{} must be finite and in [0, 1]".format(name))
    return float(value)


def _size(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("{} must contain [width, height]".format(name))
    return tuple(_integer(item, name, high=100000) for item in value)


def load_metadata(source):
    """Read an explicit class set. Never infer names from a generic model."""
    if isinstance(source, dict):
        data = deepcopy(source)
    else:
        with Path(source).open(encoding="utf-8") as stream:
            data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("metadata must be a JSON object")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("metadata schema_version must be 1")
    if data.get("model_family") not in ("yolov8", "yolo11") or data.get("task") != "detect":
        raise ValueError("only YOLOv8/YOLO11 detection models are supported")
    classes = data.get("classes")
    if (not isinstance(classes, list) or not classes or len(classes) > 1000
            or any(not isinstance(item, str) or not item.strip() or item != item.strip()
                   for item in classes) or len(set(classes)) != len(classes)):
        raise ValueError("metadata classes must be a nonempty ordered list of unique class names")
    width, height = _size(data.get("input_size"), "input_size")
    if max(width, height) > 8192 or width % 32 or height % 32:
        raise ValueError("input_size must contain multiples of 32, at most 8192")
    if data.get("output_layout") not in ("BCN", "BNC"):
        raise ValueError("output_layout must explicitly be BCN or BNC")
    required = {"box_format": "cxcywh", "coordinate_units": "input_pixels",
                "color_order": "RGB", "has_objectness": False, "embedded_nms": False}
    for key, expected in required.items():
        if key not in data or type(data[key]) is not type(expected) or data[key] != expected:
            raise ValueError("metadata {} must be {!r}".format(key, expected))
    digest = data.get("onnx_sha256")
    raw_head = data.get('raw_head')
    if raw_head is not None:
        if (raw_head != 'yolov8_dfl' or data['model_family'] != 'yolov8' or
                data['output_layout'] != 'BCN' or type(data.get('reg_max')) is not int or
                data['reg_max'] != 16 or data.get('strides') != [8, 16, 32]):
            raise ValueError('only explicit YOLOv8 DFL BCN, reg_max 16, strides 8/16/32 are supported')
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                               or any(char not in "0123456789abcdef" for char in digest)):
        raise ValueError("onnx_sha256 must be a lowercase SHA-256 hex digest")
    return data


def resolve_classes(classes, filters=None):
    """Filters are exact metadata names or integer IDs; None means all classes."""
    if filters is None:
        return tuple(range(len(classes)))
    if not isinstance(filters, (list, tuple, set)) or not filters:
        raise ValueError("class_filter must be a nonempty collection of names or IDs")
    resolved = set()
    for item in filters:
        if isinstance(item, str) and item in classes:
            resolved.add(classes.index(item))
        elif type(item) is int and 0 <= item < len(classes):
            resolved.add(item)
        else:
            raise ValueError("unknown class filter: {!r}".format(item))
    return tuple(sorted(resolved))


@dataclass(frozen=True)
class LetterboxTransform:
    original_size: tuple
    input_size: tuple
    resized_size: tuple
    pad_left: int
    pad_top: int

    def __post_init__(self):
        _size(self.original_size, "original_size")
        iw, ih = _size(self.input_size, "input_size")
        rw, rh = _size(self.resized_size, "resized_size")
        _integer(self.pad_left, "pad_left", low=0, high=100000)
        _integer(self.pad_top, "pad_top", low=0, high=100000)
        if rw > iw or rh > ih or self.pad_left != (iw - rw) // 2 or self.pad_top != (ih - rh) // 2:
            raise ValueError("inconsistent centered letterbox transform")


def _validate_image(image, np):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) < 1):
        raise ValueError("image must be a nonempty uint8 HxWx3 BGR array")
    if max(image.shape[:2]) > 100000:
        raise ValueError("image dimensions exceed supported limits")


def letterbox(image, input_size):
    """Return the padded BGR image and exact rounded-resize inverse transform."""
    cv2, np = _dependencies()
    _validate_image(image, np)
    iw, ih = _size(input_size, "input_size")
    if max(iw, ih) > 8192:
        raise ValueError("input_size exceeds 8192")
    height, width = image.shape[:2]
    ratio = min(iw / width, ih / height)
    rw, rh = min(iw, max(1, round(width * ratio))), min(ih, max(1, round(height * ratio)))
    left, top = (iw - rw) // 2, (ih - rh) // 2
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, ih - rh - top, left, iw - rw - left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, LetterboxTransform((width, height), (iw, ih), (rw, rh), left, top)


def non_max_suppression(boxes, scores, class_ids, iou_threshold=0.45,
                        max_detections=100, max_candidates=1000):
    """Stable, class-aware NMS on normalized xyxy boxes; return original indices."""
    _, np = _dependencies()
    threshold = _probability(iou_threshold, "iou_threshold")
    _integer(max_detections, "max_detections", high=10000)
    _integer(max_candidates, "max_candidates", high=30000)
    boxes, scores, class_ids = np.asarray(boxes), np.asarray(scores), np.asarray(class_ids)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or scores.shape != (len(boxes),)
            or class_ids.shape != scores.shape or boxes.dtype.kind not in "fiu"
            or scores.dtype.kind not in "fiu" or class_ids.dtype.kind not in "iu"):
        raise ValueError("NMS requires Nx4 real boxes, N real scores, and N integer class IDs")
    boxes, scores = boxes.astype(np.float64), scores.astype(np.float64)
    if (not np.isfinite(boxes).all() or not np.isfinite(scores).all()
            or (boxes < 0).any() or (boxes > 1).any() or (scores < 0).any()
            or (scores > 1).any() or (class_ids < 0).any()
            or (boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("NMS requires finite normalized positive-area boxes and probabilities")
    order = np.argsort(-scores, kind="stable")[:max_candidates]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    kept = []
    while order.size and len(kept) < max_detections:
        current = int(order[0])
        kept.append(current)
        rest = order[1:]
        intersection = np.maximum(0.0, np.minimum(boxes[current, 2:], boxes[rest, 2:])
                                  - np.maximum(boxes[current, :2], boxes[rest, :2]))
        intersection = intersection[:, 0] * intersection[:, 1]
        union = areas[current] + areas[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[(class_ids[rest] != class_ids[current]) | (overlap <= threshold)]
    return kept


def decode_predictions(output, metadata, transform, confidence_threshold=0.25,
                       iou_threshold=0.45, class_filter=None, max_detections=100,
                       max_candidates=1000):
    """Decode a single raw head. Reject incompatible or nonfinite output entirely.

    Each candidate's highest-scoring class is chosen before class filtering,
    matching single-label Ultralytics detection. Padding-only boxes are dropped.
    """
    _, np = _dependencies()
    metadata = load_metadata(metadata)
    threshold = _probability(confidence_threshold, "confidence_threshold")
    _probability(iou_threshold, "iou_threshold")
    _integer(max_detections, "max_detections", high=10000)
    _integer(max_candidates, "max_candidates", high=30000)
    if not isinstance(transform, LetterboxTransform) or tuple(metadata["input_size"]) != transform.input_size:
        raise ValueError("transform must match metadata input_size")
    allowed = resolve_classes(metadata["classes"], class_filter)
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise ValueError("expected one raw detection output; segmentation/pose/NMS heads are unsupported")
        output = output[0]
    prediction = np.asarray(output)
    if prediction.dtype.kind not in "fiu":
        raise ValueError("output tensor must be real numeric values")
    if prediction.ndim == 3 and prediction.shape[0] == 1:
        prediction = prediction[0]
    if prediction.ndim != 2:
        raise ValueError("output must have batch 1 and shape [1, 4+C, N] or [1, N, 4+C]")
    if metadata["output_layout"] == "BCN":
        prediction = prediction.T
    if prediction.shape[1] != 4 + len(metadata["classes"]):
        raise ValueError("output channels must equal 4 + explicit class count; check layout/export")
    if prediction.shape[0] > 1000000:
        raise ValueError("too many output candidates")
    prediction = prediction.astype(np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("output tensor contains nonfinite values")
    probabilities = prediction[:, 4:]
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("class scores must be probabilities in [0, 1], not logits")
    if (prediction[:, 2:4] < 0).any():
        raise ValueError("box dimensions must be nonnegative")
    class_ids = np.argmax(probabilities, axis=1)
    scores = probabilities[np.arange(len(prediction)), class_ids]
    selected = (scores >= threshold) & np.isin(class_ids, allowed)
    selected &= (prediction[:, 2] > 0) & (prediction[:, 3] > 0)
    raw, scores, class_ids = prediction[selected, :4], scores[selected], class_ids[selected]
    with np.errstate(over="ignore", invalid="ignore"):
        boxes = np.column_stack((raw[:, :2] - raw[:, 2:] / 2,
                                 raw[:, :2] + raw[:, 2:] / 2))
    if not np.isfinite(boxes).all():
        raise ValueError("box coordinates overflowed")
    # Dividing by rounded resized dimensions both undoes resize and normalizes.
    rw, rh = transform.resized_size
    boxes = (boxes - [transform.pad_left, transform.pad_top] * 2) / [rw, rh, rw, rh]
    boxes = np.clip(boxes, 0.0, 1.0)
    positive_area = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, class_ids = boxes[positive_area], scores[positive_area], class_ids[positive_area]
    kept = non_max_suppression(boxes, scores, class_ids, iou_threshold, max_detections, max_candidates)
    return [{"bbox": boxes[index].tolist(), "confidence": float(scores[index]),
             "class_id": int(class_ids[index]),
             "class_name": metadata["classes"][int(class_ids[index])]}
            for index in kept]


def _result(status, reason, **extra):
    result = {"status": status, "reason": reason, "backend": "opencv_dnn_onnx_cpu",
              "kind": "object_proposal", "bbox_format": "xyxy",
              "coordinate_space": "normalized_original_image", "bbox": None,
              "confidence": 0.0, "class_id": None, "class_name": None,
              "detections": [], "image_size": None, "inference_ms": None}
    result.update(extra)
    return result


class YoloDetector:
    """Optional CPU backend. Use detect(BGR uint8 image) or status() for readiness.

    Constructor options are validated immediately. Missing files/dependencies,
    invalid metadata, and OpenCV import failures return not_ready. Invalid frames
    or model output return error with no stale proposal. Metadata may be a JSON
    path or dict. No class set or model is supplied implicitly.
    """

    def __init__(self, weights=None, metadata=None, confidence_threshold=0.25,
                 iou_threshold=0.45, class_filter=None, max_detections=100,
                 max_candidates=1000):
        self.weights = Path(weights) if weights is not None else None
        self.metadata_source = metadata
        self.confidence_threshold = _probability(confidence_threshold, "confidence_threshold")
        self.iou_threshold = _probability(iou_threshold, "iou_threshold")
        self.max_detections = _integer(max_detections, "max_detections", high=10000)
        self.max_candidates = _integer(max_candidates, "max_candidates", high=30000)
        self.class_filter = deepcopy(class_filter)
        self._lock = threading.Lock()
        self._net = self._cv2 = self._np = self._metadata = self._allowed = None
        self._snapshot = _result("not_ready", "not_loaded")

    def _prepare(self):
        if self._net is not None:
            return None
        if self.weights is None or not self.weights.is_file():
            return _result("not_ready", "weights_missing")
        if self.weights.suffix.lower() != ".onnx":
            return _result("not_ready", "weights_must_be_onnx")
        if self.metadata_source is None:
            return _result("not_ready", "metadata_missing")
        try:
            metadata = load_metadata(self.metadata_source)
            allowed = resolve_classes(metadata["classes"], self.class_filter)
            expected = metadata.get("onnx_sha256")
            if expected is not None:
                digest = hashlib.sha256()
                with self.weights.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    return _result("not_ready", "weights_checksum_mismatch")
        except (OSError, ValueError, TypeError) as exc:
            return _result("not_ready", "invalid_metadata", error=str(exc))
        try:
            cv2, np = _dependencies()
        except (ImportError, OSError) as exc:
            return _result("not_ready", "dependencies_unavailable", error=str(exc))
        try:
            net = cv2.dnn.readNetFromONNX(str(self.weights))
            if net.empty():
                return _result("not_ready", "model_empty")
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            if len(net.getUnconnectedOutLayersNames()) != 1:
                return _result("not_ready", "model_requires_one_raw_output")
        except (cv2.error, OSError) as exc:
            return _result("not_ready", "model_load_failed", error=str(exc), opencv_version=cv2.__version__)
        self._net, self._cv2, self._np = net, cv2, np
        self._metadata, self._allowed = metadata, allowed
        return None

    def status(self):
        """Load on first call. Ready means imported; detect still validates output."""
        with self._lock:
            failure = self._prepare()
            if failure is not None:
                return failure
            return _result("ready", "model_loaded", classes=list(self._metadata["classes"]),
                           class_filter=list(self._allowed), input_size=list(self._metadata["input_size"]),
                           opencv_version=self._cv2.__version__)

    def snapshot(self):
        with self._lock:
            return deepcopy(self._snapshot)

    def detect(self, image):
        with self._lock:
            failure = self._prepare()
            if failure is not None:
                self._snapshot = failure
                return deepcopy(failure)
            start = time.perf_counter()
            try:
                padded, transform = letterbox(image, self._metadata["input_size"])
                blob = self._cv2.dnn.blobFromImage(padded, scalefactor=1.0 / 255.0,
                                                  size=tuple(self._metadata["input_size"]),
                                                  swapRB=True, crop=False)
                self._net.setInput(blob)
                output = self._net.forward()
                if self._metadata.get('raw_head') == 'yolov8_dfl':
                    from yolo_dfl import decode_dfl
                    output = decode_dfl(output, self._metadata['input_size'],
                                        len(self._metadata['classes']),
                                        self._metadata['reg_max'], self._metadata['strides'])
                detections = decode_predictions(output, self._metadata, transform,
                                                self.confidence_threshold, self.iou_threshold,
                                                self._allowed, self.max_detections, self.max_candidates)
                result = _result("proposal" if detections else "no_detection",
                                 "object_detected" if detections else "no_matching_detection",
                                 detections=detections, image_size=list(transform.original_size),
                                 inference_ms=(time.perf_counter() - start) * 1000,
                                 opencv_version=self._cv2.__version__)
                if detections:
                    result.update(detections[0])
            except (ValueError, TypeError, OverflowError, self._cv2.error) as exc:
                result = _result("error", "invalid_image_or_model_output", error=str(exc),
                                 opencv_version=self._cv2.__version__)
            self._snapshot = result
            return deepcopy(result)

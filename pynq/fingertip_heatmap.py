"""Four-fingertip heatmaps with PS preprocessing/decoding and PL learned MACs.

Use ``FingertipHeatmap(directory, engine).infer_bgr(frame, roi)`` or
``infer_rgba(frame, roi)``. Frames are full-frame uint8 arrays; ROI is pixel
``(x, y, width, height)`` or a mapping with those keys. ROIs must be entirely
inside the frame. Fractional edges use floor/ceil as in the training dataset;
returned coordinates refer to that actual integer crop, reported as ``roi``.

Results contain ``points[name] = {x, y, raw_score}`` for index/middle/ring/little
and ``timings_ms`` for preprocessing, graph execution, decoding, and total.
Scores are raw model peaks, not probabilities or visibility estimates. No
threshold, occlusion decision, temporal smoothing, or CPU model fallback is
applied. Instances share mutable graph input state and are not thread-safe.

Only NumPy is imported eagerly. ``graph=`` accepts a supplied setInput/forward
graph for offline tests. OpenCV is loaded only when a crop needs resizing.
"""

from collections.abc import Mapping
import math
from numbers import Real
import time

import numpy as np


POINT_NAMES = ("index", "middle", "ring", "little")
INPUT_SIZE = 208
HEATMAP_SIZE = 26
MEAN = np.asarray([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.asarray([58.395, 57.12, 57.375], dtype=np.float32)


def _roi_values(roi):
    try:
        values = tuple(roi[key] for key in ("x", "y", "width", "height")) if isinstance(roi, Mapping) else tuple(roi)
    except (KeyError, TypeError) as exc:
        raise ValueError("ROI requires pixel x, y, width, height") from exc
    if len(values) != 4 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in values):
        raise ValueError("ROI requires four real pixel values")
    x, y, width, height = map(float, values)
    if not all(math.isfinite(v) for v in (x, y, width, height, x + width, y + height)):
        raise ValueError("ROI must be finite")
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("ROI origin must be nonnegative and extent positive")
    return x, y, width, height


def _pixel_roi(roi, frame_shape):
    x, y, width, height = _roi_values(roi)
    if x + width > frame_shape[1] or y + height > frame_shape[0]:
        raise ValueError("ROI must be entirely inside the frame")
    x0, y0 = math.floor(x), math.floor(y)
    x1, y1 = math.ceil(x + width), math.ceil(y + height)
    return x0, y0, x1 - x0, y1 - y0


def _preprocess(frame, roi, channels):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != channels
            or frame.shape[0] == 0 or frame.shape[1] == 0):
        raise ValueError("Frame must be a nonempty HWC uint8 array with {} channels".format(channels))
    pixel_roi = _pixel_roi(roi, frame.shape)
    x, y, width, height = pixel_roi
    crop = frame[y:y + height, x:x + width, :3]
    if crop.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
        import cv2
        crop = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    # Match the dataset: resize bytes first, then RGB float32 normalization.
    rgb = (crop[:, :, ::-1] if channels == 3 else crop).astype(np.float32)
    rgb = (rgb - MEAN) / STD
    blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    return blob, pixel_roi


def decode_heatmaps(heatmaps, roi):
    """Return pixel coordinates (4, 2) and raw peaks (4,), without OpenCV.

    Accept exactly (1, 4, 26, 26) or (4, 26, 26) floating heatmaps. Argmax
    uses the first row-major maximum, with x/26 and y/26, as in training.
    ``roi`` is the effective pixel crop (x, y, width, height).
    """
    x, y, width, height = _roi_values(roi)
    maps = np.asarray(heatmaps)
    if maps.shape == (1, 4, HEATMAP_SIZE, HEATMAP_SIZE):
        maps = maps[0]
    if (maps.shape != (4, HEATMAP_SIZE, HEATMAP_SIZE)
            or not np.issubdtype(maps.dtype, np.floating)
            or not np.isfinite(maps).all()):
        raise ValueError("Expected finite floating heatmaps shaped (1,4,26,26) or (4,26,26)")
    flat = maps.reshape(4, -1)
    indices = flat.argmax(axis=1)
    raw_scores = flat[np.arange(4), indices].copy()
    coords = np.empty((4, 2), dtype=np.float64)
    coords[:, 0] = x + (indices % HEATMAP_SIZE) / HEATMAP_SIZE * width
    coords[:, 1] = y + (indices // HEATMAP_SIZE) / HEATMAP_SIZE * height
    return coords, raw_scores


class FingertipHeatmap:
    """Reuse a caller-owned PL engine; do not create an overlay or change it.

    Supply either ``directory, engine[, model_path=...]`` for HandGraph, or
    only ``graph=...`` for an explicit offline/test implementation. Production
    graphs are restricted to this candidate's PLConvolution/Clip/Add ops.
    HandGraph validates the weight checksum and optional source model hash.
    """

    def __init__(self, directory=None, engine=None, *, graph=None, model_path=None):
        if graph is not None:
            if directory is not None or engine is not None or model_path is not None:
                raise ValueError("Supply graph alone, or directory and engine")
        else:
            if directory is None or not callable(getattr(engine, "convolve", None)):
                raise ValueError("A compiled graph directory and PL engine are required")
            if __package__:
                from .hand_graph import HandGraph
            else:
                from hand_graph import HandGraph
            graph = HandGraph(directory, engine, model_path=model_path)
            if (graph.outputs != ["fingertip_heatmaps"] or not graph.nodes
                    or not any(node["op"] == "PLConvolution" for node in graph.nodes)
                    or any(node["op"] not in ("PLConvolution", "Clip", "Add") for node in graph.nodes)):
                raise ValueError("Unsupported compiled fingertip heatmap graph")
        if not callable(getattr(graph, "setInput", None)) or not callable(getattr(graph, "forward", None)):
            raise ValueError("Graph must provide setInput and forward")
        self.graph = graph

    def infer_bgr(self, frame, roi):
        """Infer from a full BGR frame and an explicit pixel ROI."""
        return self._infer(frame, roi, 3)

    def infer_rgba(self, frame, roi):
        """Infer from a full RGBA frame; alpha is ignored, RGB is preserved."""
        return self._infer(frame, roi, 4)

    def _infer(self, frame, roi, channels):
        started = time.perf_counter()
        blob, pixel_roi = _preprocess(frame, roi, channels)
        preprocessed = time.perf_counter()
        self.graph.setInput(blob)
        outputs = self.graph.forward()
        forwarded = time.perf_counter()
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise ValueError("Graph must return exactly one fingertip heatmap tensor")
        # A graph invocation always has batch one, even though the pure decoder
        # also accepts an already unbatched tensor.
        if np.shape(outputs[0]) != (1, 4, HEATMAP_SIZE, HEATMAP_SIZE):
            raise ValueError("Graph output must have shape (1,4,26,26)")
        coords, raw_scores = decode_heatmaps(outputs[0], pixel_roi)
        points = {name: {"x": float(coord[0]), "y": float(coord[1]), "raw_score": float(score)}
                  for name, coord, score in zip(POINT_NAMES, coords, raw_scores)}
        decoded = time.perf_counter()
        return {"points": points, "roi": pixel_roi,
                "timings_ms": {"preprocess": (preprocessed - started) * 1000,
                               "graph": (forwarded - preprocessed) * 1000,
                               "decode": (decoded - forwarded) * 1000,
                               "total": (decoded - started) * 1000}}

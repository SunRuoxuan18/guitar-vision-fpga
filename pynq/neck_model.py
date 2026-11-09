"""Optional PS CPU YOLO neck proposal adapter for offline/PC bridge use."""

import threading
import time
from copy import deepcopy

import cv2
import numpy as np

from neck_region import find_neck_regions
from yolo_detector import YoloDetector


class NeckModel:
    """Turn a YOLO fretboard proposal into refined, original-frame geometry."""

    def __init__(self, model, metadata, confidence_threshold=0.5):
        self.detector = YoloDetector(model, metadata, class_filter=["fretboard"])
        self.confidence_threshold = float(confidence_threshold)
        self.lock = threading.Lock()
        self.last = {"status": "starting", "reason": "not_loaded"}
        self.last_detection = None
        self.last_error = None
        self.load_ms = None
        load_started = time.perf_counter()
        self.last = self.detector.status()
        self.load_ms = (time.perf_counter() - load_started) * 1000
        if self.last.get("status") != "ready":
            self.last_error = self.last.get("reason")

    def status(self):
        with self.lock:
            status = dict(self.last)
            status.update(load_ms=self.load_ms, last_detection=self.last_detection,
                          last_error=self.last_error, backend="ps_opencv_dnn_cpu")
            return deepcopy(status)

    def _publish(self, state, detection=None, error=None):
        with self.lock:
            self.last = state
            self.last_detection = detection
            self.last_error = error

    @staticmethod
    def _select_proposal(proposals, shape):
        """Prefer the longest visible border extent, matching bridge policy."""
        height, width = shape[:2]
        def extent(proposal):
            points = np.float32([[p["x"] * (width - 1), p["y"] * (height - 1)]
                                 for p in proposal])
            return float(np.linalg.norm(points[1] - points[0]) +
                         np.linalg.norm(points[2] - points[3]))
        lengths = [extent(proposal) for proposal in proposals]
        eligible = [proposal for proposal, length in zip(proposals, lengths)
                    if length >= .95 * max(lengths)]
        return min(eligible, key=lambda proposal: cv2.contourArea(
            np.float32([[p['x'] * (width - 1), p['y'] * (height - 1)] for p in proposal])))

    @staticmethod
    def _model_supported(proposals, bbox):
        """Reject unrelated full-frame borders before comparing visible extent."""
        x0, y0, x1, y1 = bbox
        rectangle = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        supported = []
        for proposal in proposals:
            points = np.float32([[p['x'], p['y']] for p in proposal])
            area = cv2.contourArea(points)
            if area <= 0 or not cv2.isContourConvex(points):
                continue
            overlap, _ = cv2.intersectConvexConvex(points, rectangle)
            if overlap >= .65 * area:
                supported.append(proposal)
        return supported

    @staticmethod
    def _validated_proposals(proposals):
        """Accept only full-frame normalized, convex four-corner observations."""
        if proposals is None:
            return None
        if not isinstance(proposals, (list, tuple)):
            raise ValueError("proposals_must_be_a_sequence")
        valid = []
        for proposal in proposals:
            if not isinstance(proposal, (list, tuple)) or len(proposal) != 4:
                continue
            try:
                points = np.float32([[float(point["x"]), float(point["y"])]
                                     for point in proposal])
            except (KeyError, TypeError, ValueError):
                continue
            if (not np.isfinite(points).all() or (points < 0).any() or (points > 1).any()
                    or cv2.contourArea(points) <= 0 or not cv2.isContourConvex(points)):
                continue
            valid.append([{"x": float(point[0]), "y": float(point[1])} for point in points])
        return valid

    @classmethod
    def _model_gated_proposals(cls, proposals, bbox, shape):
        """Keep observed borders inside the padded model ROI with enough extent."""
        supported = cls._model_supported(proposals, bbox)
        height, width = shape[:2]
        x0, y0, x1, y1 = bbox
        required_extent = .65 * max((x1 - x0) * (width - 1),
                                    (y1 - y0) * (height - 1))
        return [proposal for proposal in supported
                if cls._proposal_extent(proposal, shape) >= required_extent]

    @staticmethod
    def _proposal_extent(proposal, shape):
        height, width = shape[:2]
        points = np.float32([[point["x"] * (width - 1), point["y"] * (height - 1)]
                             for point in proposal])
        return float((np.linalg.norm(points[1] - points[0]) +
                      np.linalg.norm(points[2] - points[3])) / 2.0)

    @staticmethod
    def _detector_diagnostics(detector):
        """Preserve threshold evidence without changing the detector threshold."""
        keys = ("confidence", "max_candidate_confidence", "max_confidence",
                "candidates_considered", "candidate_count", "inference_ms")
        return {key: detector[key] for key in keys if key in detector}

    @staticmethod
    def _map_proposal(proposal, x0, y0, x1, y1, width, height):
        mapped = []
        for point in proposal:
            mapped.append({"x": float((x0 + point["x"] * (x1 - x0 - 1)) / (width - 1)),
                           "y": float((y0 + point["y"] * (y1 - y0 - 1)) / (height - 1))})
        return mapped

    def detect(self, bgr, proposals=None):
        started = time.perf_counter()
        try:
            if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
                raise ValueError("expected_uint8_bgr")
            observed = self._validated_proposals(proposals)
            detector = self.detector.detect(bgr)
            diagnostics = self._detector_diagnostics(detector)
            if detector.get("status") != "proposal":
                state = {"status": "no_detection", "reason": detector.get("reason", "no_proposal"),
                         "inference_ms": detector.get("inference_ms"),
                         "model_diagnostics": diagnostics}
                self._publish(state)
                return dict(state, detector=detector)
            if float(detector.get("confidence", 0.0)) < self.confidence_threshold:
                state = {"status": "no_detection", "reason": "confidence_below_threshold",
                         "inference_ms": detector.get("inference_ms"),
                         "model_diagnostics": diagnostics}
                self._publish(state)
                return dict(state, detector=detector)
            height, width = bgr.shape[:2]
            bx0, by0, bx1, by1 = detector["bbox"]
            pad_x, pad_y = (bx1 - bx0) * .05, (by1 - by0) * .1
            padded_bbox = [max(0., bx0 - pad_x), max(0., by0 - pad_y),
                           min(1., bx1 + pad_x), min(1., by1 + pad_y)]
            if observed:
                gated = self._model_gated_proposals(observed, padded_bbox, bgr.shape)
                if gated:
                    mapped = self._select_proposal(gated, bgr.shape)
                    state = {"status": "tracking", "reason": "model_bbox_geometry_gated",
                             "inference_ms": detector.get("inference_ms"),
                             "bbox": detector["bbox"], "confidence": detector["confidence"],
                             "geometry_candidates": len(observed),
                             "geometry_gated_candidates": len(gated),
                             "total_ms": (time.perf_counter() - started) * 1000}
                    self._publish(state, dict(state))
                    return dict(state, corners=mapped, acquisition="neck_model_geometry_gated",
                                lattice_confirmed=False, detector=detector)
            x0, y0 = max(0, int((bx0 - pad_x) * (width - 1))), max(0, int((by0 - pad_y) * (height - 1)))
            x1, y1 = min(width, int((bx1 + pad_x) * (width - 1)) + 1), min(height, int((by1 + pad_y) * (height - 1)) + 1)
            crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            proposals = [self._map_proposal(p, x0, y0, x1, y1, width, height)
                         for p in find_neck_regions(crop)]
            # Cropping can split a border at an occluding hand. Compare the
            # observed full-frame borders too, but only inside the model ROI.
            proposals += find_neck_regions(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            proposals = self._model_supported(proposals, padded_bbox)
            if not proposals:
                # A valid YOLO box is already a neck localization. Border
                # geometry is optional refinement, not a second detector gate.
                mapped = [dict(x=float(x), y=float(y)) for x, y in (
                    (bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1))]
                state = {"status": "tracking", "reason": "model_bbox_only",
                         "inference_ms": detector.get("inference_ms"),
                         "bbox": detector["bbox"], "confidence": detector["confidence"],
                         "geometry_candidates": 0, "total_ms": (time.perf_counter() - started) * 1000}
                self._publish(state, dict(state))
                return dict(state, corners=mapped, acquisition="yolo_bbox",
                            lattice_confirmed=False, detector=detector)
            mapped = self._select_proposal(proposals, bgr.shape)
            state = {"status": "tracking", "reason": "model_roi_geometry_refined",
                     "inference_ms": detector.get("inference_ms"),
                     "bbox": detector['bbox'], "confidence": detector['confidence'],
                     "total_ms": (time.perf_counter() - started) * 1000}
            self._publish(state, dict(state))
            return dict(state, corners=mapped, acquisition="neck_model_geometry_refined",
                        lattice_confirmed=False, detector=detector)
        except Exception as exc:
            state = {"status": "failed", "reason": str(exc),
                     "total_ms": (time.perf_counter() - started) * 1000}
            self._publish(state, error=str(exc))
            return dict(state)

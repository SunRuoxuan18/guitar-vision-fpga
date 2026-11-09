"""YOLO proposes a crop; observed line geometry still requires image evidence."""
from copy import deepcopy
import math


class YoloFretboardPipeline:
    def __init__(self, detector, geometry_detector=None):
        if geometry_detector is None:
            from auto_fretboard import AutoFretboardTracker
            geometry_detector = AutoFretboardTracker()
        self.detector, self.geometry_detector = detector, geometry_detector
        self.invalidate("not_detected")

    def invalidate(self, reason):
        self._snapshot = {"mode": "auto", "status": "searching", "reason": str(reason),
                          "corners": None, "strings": [], "frets": [],
                          "absolute_frets_known": False, "orientation": "image_order",
                          "backend": "yolo_roi_then_observed_lines", "confidence": 0.0}
        return deepcopy(self._snapshot)

    def update(self, gray, bgr=None):
        import cv2
        import numpy as np
        self.invalidate("no_supported_lattice")
        if (not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or
                gray.ndim != 2 or min(gray.shape) < 3):
            return self.invalidate("invalid_frame")
        if bgr is None:
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if bgr.shape != (*gray.shape, 3):
            return self.invalidate("image_shape_mismatch")
        proposal = self.detector.detect(bgr)
        self._snapshot["proposal_status"] = proposal.get("status")
        self._snapshot["proposal_reason"] = proposal.get("reason")
        if proposal.get("status") != "proposal":
            self._snapshot["reason"] = "yolo_{}".format(proposal.get("status", "error"))
            return deepcopy(self._snapshot)
        box = proposal.get("bbox")
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or not 0 <= value <= 1 for value in box) or
                box[2] <= box[0] or box[3] <= box[1]):
            return self.invalidate("invalid_proposal")
        height, width = gray.shape
        # Add context for outer strings and nearby fret wires.
        dx, dy = (box[2] - box[0]) * .05, (box[3] - box[1]) * .1
        x0 = max(0, math.floor((box[0] - dx) * (width - 1)))
        y0 = max(0, math.floor((box[1] - dy) * (height - 1)))
        x1 = min(width, math.ceil((box[2] + dx) * (width - 1)) + 1)
        y1 = min(height, math.ceil((box[3] + dy) * (height - 1)) + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return self.invalidate("proposal_too_small")
        geometry = self.geometry_detector.update(gray[y0:y1, x0:x1])
        self._snapshot.update(proposal_bbox=list(box), proposal_confidence=proposal.get("confidence"))
        if geometry.get("status") != "tracking":
            return deepcopy(self._snapshot)

        def point(value):
            if (not isinstance(value, dict) or any(not isinstance(value.get(key), (int, float)) or
                    not math.isfinite(value[key]) or not 0 <= value[key] <= 1 for key in ("x", "y"))):
                raise ValueError("Invalid observed geometry")
            return {"x": (x0 + value["x"] * (x1 - x0 - 1)) / (width - 1),
                    "y": (y0 + value["y"] * (y1 - y0 - 1)) / (height - 1)}

        try:
            mapped = deepcopy(geometry)
            if len(mapped["corners"]) != 4 or len(mapped["strings"]) != 6 or not 5 <= len(mapped["frets"]) <= 25:
                raise ValueError("Incomplete observed lattice")
            mapped["corners"] = [point(item) for item in mapped["corners"]]
            for line in mapped["strings"] + mapped["frets"]:
                if line.get("observed") is not True:
                    raise ValueError("Unsupported line")
                line["start"], line["end"] = point(line["start"]), point(line["end"])
            self._snapshot.update(mapped, backend="yolo_roi_then_observed_lines", absolute_frets_known=False)
        except (KeyError, TypeError, ValueError):
            return self.invalidate("invalid_observed_geometry")
        return deepcopy(self._snapshot)

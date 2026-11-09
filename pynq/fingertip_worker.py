"""One inference thread and one replaceable pending ROI; no board ownership.

``submit(rgba, roi, stream_id, generation, sequence)`` copies only the effective
pixel crop. ``capture_monotonic=`` may carry an upstream capture timestamp in
this process's monotonic clock domain; by default it is the submit time.
ROI rounding/validation matches fingertip_heatmap. Accepted submissions return
True; duplicate/out-of-order sequences in the same identity return False.

``snapshot(stream_id, generation, sequence)`` is JSON-safe and never encodes an
image. Top-level points require that exact frame AND age <= max_age_ms (250 by
default). Older frames are stale even with a larger TTL. Same-identity history
is under last_result, with full-frame points, effective ROI, age/stale reason,
and preview {jpeg_base64, mime_type, width, height}. No other identity's history
is exposed. JPEG is encoded once per successful inference on this worker from
its saved ROI, resized to 208x208. Scores remain raw_score, never confidence.

Capture/completed timestamps are monotonic seconds; age/inference are ms.
inference_ms measures infer_rgba only; completed_monotonic includes JPEG work.
dropped counts overwritten/discarded pending jobs and rejected old sequences.
running means a job is in progress (including JPEG); pending is at most one.

request_stop() discards pending work. shutdown(wait=True, timeout=None) joins
the non-daemon thread and returns whether it has exited. A False return means
the caller MUST keep the shared model/engine alive. No stop method interrupts
inference or closes the model/engine. The caller must reserve exclusive model
use for this worker and shut it down before releasing hardware resources.
"""

import base64
import copy
import math
import os
from numbers import Integral, Real
import threading
import time

import numpy as np

if __package__:
    from .fingertip_heatmap import INPUT_SIZE, POINT_NAMES, _pixel_roi
else:
    from fingertip_heatmap import INPUT_SIZE, POINT_NAMES, _pixel_roi


def _source(stream_id, generation, sequence):
    if not isinstance(stream_id, str) or not stream_id:
        raise ValueError("stream_id must be a nonempty string")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0
           for v in (generation, sequence)):
        raise ValueError("generation and sequence must be nonnegative integers")
    return {"stream_id": stream_id, "generation": int(generation), "sequence": int(sequence)}


def _finite_number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be finite and real")
    return float(value)


def _encode_preview(crop):
    import cv2
    resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGBA2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("Fingertip ROI JPEG encoding failed")
    return {"jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            "mime_type": "image/jpeg", "width": INPUT_SIZE, "height": INPUT_SIZE}


def _mapped_points(result, roi):
    points = result["points"]
    if set(points) != set(POINT_NAMES):
        raise ValueError("Model must return exactly four fingertip points")
    x0, y0, width, height = roi
    mapped = {}
    for name in POINT_NAMES:
        point = points[name]
        x = _finite_number(point["x"], "point x")
        y = _finite_number(point["y"], "point y")
        score = _finite_number(point["raw_score"], "raw_score")
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Model point is outside its submitted crop")
        mapped[name] = {"x": x + x0, "y": y + y0, "raw_score": score}
    return mapped


class FingertipWorker:
    def __init__(self, model, *, max_age_ms=250, clock=None):
        if not callable(getattr(model, "infer_rgba", None)):
            raise ValueError("Model must provide infer_rgba")
        self.max_age_ms = _finite_number(max_age_ms, "max_age_ms")
        if self.max_age_ms <= 0:
            raise ValueError("max_age_ms must be positive")
        self._model = model
        self._clock = time.monotonic if clock is None else clock
        if not callable(self._clock):
            raise ValueError("clock must be callable")
        self._condition = threading.Condition()
        self._pending = None
        self._latest_source = None
        self._epoch = 0
        self._last_result = None
        self._last_outcome = None
        self._dropped = 0
        self._running = False
        self._stopping = False
        self._stopped = False
        requested_nice = int(os.environ.get('FRETLENS_FINGERTIP_NICE', '10'))
        if not 0 <= requested_nice <= 19:
            raise ValueError('FRETLENS_FINGERTIP_NICE must be between 0 and 19')
        self._scheduling = {'requested_nice': requested_nice, 'actual_nice': None, 'error': None}
        self._thread = threading.Thread(target=self._run, name="fingertip-worker", daemon=False)
        self._thread.start()

    def submit(self, frame_rgba, roi_pixels, stream_id, generation, sequence, *, capture_monotonic=None):
        source = _source(stream_id, generation, sequence)
        now = self._clock()
        captured = now if capture_monotonic is None else _finite_number(capture_monotonic, "capture_monotonic")
        if captured < 0 or captured > now:
            raise ValueError("capture_monotonic must be nonnegative and not in the future")
        if (not isinstance(frame_rgba, np.ndarray) or frame_rgba.dtype != np.uint8
                or frame_rgba.ndim != 3 or frame_rgba.shape[2] != 4
                or not frame_rgba.shape[0] or not frame_rgba.shape[1]):
            raise ValueError("Frame must be a nonempty HWC uint8 RGBA array")
        roi = _pixel_roi(roi_pixels, frame_rgba.shape)
        x, y, width, height = roi
        with self._condition:
            if self._stopping:
                raise RuntimeError("Fingertip worker is stopping")
            same_identity = (self._latest_source is not None
                             and source["stream_id"] == self._latest_source["stream_id"]
                             and source["generation"] == self._latest_source["generation"])
            if same_identity and sequence <= self._latest_source["sequence"]:
                self._dropped += 1
                return False
            # Copy only the crop while serializing submissions. No caller-owned
            # array survives submit, including views into a camera buffer.
            crop = frame_rgba[y:y + height, x:x + width].copy()
            crop.flags.writeable = False
            if not same_identity:
                self._epoch += 1
                self._last_result = None
                self._last_outcome = None
            if self._pending is not None:
                self._dropped += 1
            self._latest_source = source
            self._pending = {**source, "capture_monotonic": captured, "roi": roi,
                             "crop": crop, "epoch": self._epoch}
            self._condition.notify_all()
        return True

    def _run(self):
        try:
            if hasattr(os, 'setpriority') and hasattr(threading, 'get_native_id'):
                try:
                    tid = threading.get_native_id()
                    os.setpriority(os.PRIO_PROCESS, tid, self._scheduling['requested_nice'])
                    actual = os.getpriority(os.PRIO_PROCESS, tid)
                    with self._condition:
                        self._scheduling['actual_nice'] = actual
                except OSError as exc:
                    with self._condition:
                        self._scheduling['error'] = str(exc)
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or self._pending is not None)
                    if self._stopping:
                        return
                    job, self._pending = self._pending, None
                    self._running = True
                outcome = {key: job[key] for key in ("stream_id", "generation", "sequence",
                                                     "capture_monotonic", "roi")}
                started = self._clock()
                inference_ms = None
                try:
                    width, height = job["roi"][2:]
                    result = self._model.infer_rgba(job["crop"], (0, 0, width, height))
                    inference_ms = (self._clock() - started) * 1000
                    outcome["points"] = _mapped_points(result, job["roi"])
                    outcome["preview"] = _encode_preview(job["crop"])
                    outcome["fault"] = None
                except Exception as exc:
                    if inference_ms is None:
                        inference_ms = (self._clock() - started) * 1000
                    outcome["points"] = None
                    outcome["fault"] = {"type": type(exc).__name__, "message": str(exc)}
                outcome["inference_ms"] = inference_ms
                outcome["completed_monotonic"] = self._clock()
                with self._condition:
                    # An old in-flight job must not publish after an identity
                    # switch, even if a stream name is later reused (ABA).
                    if job["epoch"] == self._epoch:
                        self._last_outcome = outcome
                        if outcome["fault"] is None:
                            self._last_result = outcome
                    self._running = False
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._running = False
                self._stopped = True
                self._condition.notify_all()

    def _visible_result(self, result, current, now):
        if (result is None or result["stream_id"] != current["stream_id"]
                or result["generation"] != current["generation"]
                or result["sequence"] > current["sequence"]):
            return None
        result = copy.deepcopy(result)
        result["age_ms"] = max(0.0, (now - result["capture_monotonic"]) * 1000)
        reasons = []
        if result["age_ms"] > self.max_age_ms:
            reasons.append("age")
        if result["sequence"] != current["sequence"]:
            reasons.append("sequence")
        result["stale"] = bool(reasons)
        result["stale_reasons"] = reasons
        result["status"] = "fault" if result["fault"] else "stale" if reasons else "ok"
        return result

    def snapshot(self, stream_id, generation, sequence):
        """Poll without blocking inference; history never becomes current points."""
        current = _source(stream_id, generation, sequence)
        with self._condition:
            now = self._clock()
            outcome = self._visible_result(self._last_outcome, current, now)
            history = self._visible_result(self._last_result, current, now)
            if self._stopped:
                status = "stopped"
            elif self._stopping:
                status = "stopping"
            elif outcome is not None:
                status = outcome["status"]
            elif self._latest_source is not None and (
                    self._latest_source["stream_id"] != stream_id
                    or self._latest_source["generation"] != generation):
                status = "source_mismatch"
            elif self._last_outcome is not None:
                status = "sequence_mismatch"
            else:
                status = "pending" if self._running or self._pending is not None else "idle"
            metadata = {key: outcome[key] if outcome else None for key in (
                "stream_id", "generation", "sequence", "capture_monotonic",
                "completed_monotonic", "age_ms", "inference_ms", "roi")}
            return {**metadata, "status": status, "current_source": current,
                    "points": copy.deepcopy(outcome["points"]) if status == "ok" else None,
                    "last_result": history, "fault": outcome["fault"] if outcome else None,
                    "running": self._running, "pending": self._pending is not None,
                    "dropped": self._dropped, "stopping": self._stopping,
                    "stopped": self._stopped, "max_age_ms": self.max_age_ms,
                    "scheduling": dict(self._scheduling)}

    def request_stop(self):
        """Nonblocking stop request; an in-flight job finishes on its own thread."""
        with self._condition:
            self._stopping = True
            if self._pending is not None:
                self._pending = None
                self._dropped += 1
            self._condition.notify_all()

    def shutdown(self, wait=True, timeout=None):
        """Return True only after exit; never close or free the shared engine."""
        self.request_stop()
        if wait and threading.current_thread() is not self._thread:
            self._thread.join(timeout)
        return not self._thread.is_alive()

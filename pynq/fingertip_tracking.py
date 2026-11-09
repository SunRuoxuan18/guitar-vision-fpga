"""Independent fingertip LK tracking; no board ownership or hand affine model.

Call update for each analyzed 640x480 uint8 gray frame, in sequence order.
Sequences are camera capture IDs and may skip between analyzed frames.
The worker snapshot's last_result may be delayed: its source frame must still
be in the 64-frame history. Replay is limited to 20 frame pairs / 80 ms per
call (a running backend call cannot be interrupted). Pending replay publishes
no coordinates. A failed neural source is not retried; a newer source can
reseed lost tips. Scores are raw peaks, never visibility probabilities.
Opt-in fast_catchup tries one validated source-to-current pair for a new seed
at most 2.5 seconds old and at least two retained frames behind. A rejected
direct pair restores the seed for replay; its time and step count still apply.
Fault snapshots clear all tips and block their failed or older neural sources.
FB error and 11x11 patch checks use the retained 320x200 image coordinates.

A supplied flow_backend(previous, current, points) must return LK points,
status, error, like vision_hls.PyramidLK. The caller owns PL serialization,
overlay identity checks and backend lifetime. No backend is opened or closed
here. Without one, OpenCV CPU LK is explicitly reported in method.
Faulted supplied backends raise to the caller without further flow calls.
"""

from collections import deque
import math
from numbers import Integral, Real
import time

import cv2
import numpy as np

if __package__:
    from .fingertip_heatmap import POINT_NAMES
else:
    from fingertip_heatmap import POINT_NAMES


_SCALE = np.array([320.0 / 640, 200.0 / 480], dtype=np.float32)


def _number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("Expected a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Expected a finite real number")
    return value


def _identity(value, stream_key="stream_id"):
    if not isinstance(value, dict):
        raise ValueError("Expected source metadata")
    stream = value.get(stream_key)
    generation, sequence = value.get("generation"), value.get("sequence")
    if not isinstance(stream, str) or not stream:
        raise ValueError("stream must be a nonempty string")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0
           for v in (generation, sequence)):
        raise ValueError("generation and sequence must be nonnegative integers")
    return (stream, int(generation)), int(sequence)


class FourTipTracker:
    HISTORY_SIZE = 64
    MAX_STEPS = 20
    BUDGET_SECONDS = .080
    MAX_GAP_SECONDS = .6
    MAX_AGE_SECONDS = 4.0
    MAX_DIRECT_AGE_SECONDS = 2.5
    PATCH_SIZE = 11

    def __init__(self, flow_backend=None, clock=time.monotonic, fast_catchup=False,
                 patch_quality=None):
        if flow_backend is not None and not callable(flow_backend):
            raise ValueError("flow_backend must be callable")
        if not callable(clock):
            raise ValueError("clock must be callable")
        if type(fast_catchup) is not bool:
            raise ValueError("fast_catchup must be a bool")
        if patch_quality is not None and not callable(getattr(patch_quality, "evaluate", None)):
            raise ValueError("patch_quality must supply evaluate")
        self._flow_backend = flow_backend
        self._patch_quality = patch_quality
        self._clock = clock
        self.fast_catchup = fast_catchup
        self.method = (str(getattr(flow_backend, "backend_name", "supplied_lk")) + "_fb_patch"
                       if flow_backend is not None else "ps_opencv_lk_fb_patch")
        self._history = deque(maxlen=self.HISTORY_SIZE)
        self._identity = None
        self._last_sequence = None
        self._last_time = None
        self._attempted_sequence = -1
        self._source_sequence = None
        self._captured = None
        self._tracked_sequence = None
        self._has_flow = False
        self._tips = {}
        self._losses = {}
        self._quality = {}
        self._flow_timings = []
        self._direct_attempt = False
        self._direct_accepted = False

    @classmethod
    def _patch(cls, gray, point):
        x, y = point
        x, y = float(x), float(y)
        radius = cls.PATCH_SIZE // 2
        if (not math.isfinite(x) or not math.isfinite(y)
                or not radius <= x < gray.shape[1] - radius
                or not radius <= y < gray.shape[0] - radius):
            return None
        return cv2.getRectSubPix(gray, (cls.PATCH_SIZE, cls.PATCH_SIZE), (x, y)).astype(np.float32)

    @staticmethod
    def _correlation(first, second):
        first = first - first.mean()
        second = second - second.mean()
        norm = float(np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.sum(first * second) / norm) if norm > 0 else -1.0

    def _lose_all(self, reason):
        self._losses.update({name: reason for name in self._tips})
        self._tips.clear()

    def _flow(self, previous, current, points):
        if self._flow_backend is not None:
            if getattr(self._flow_backend, "fault", False):
                raise RuntimeError("Fingertip flow backend is faulted")
            result = self._flow_backend(previous, current, points)
            timings = getattr(self._flow_backend, "timings", None)
            if isinstance(timings, dict):
                self._flow_timings.append(dict(timings))
            if getattr(self._flow_backend, "fault", False):
                raise RuntimeError("Fingertip flow backend faulted during LK")
            return result
        return cv2.calcOpticalFlowPyrLK(
            previous, current, points, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))

    def _step(self, previous, current):
        names = list(self._tips)
        points = np.array([self._tips[name]["point"] for name in names], np.float32).reshape(-1, 1, 2)
        try:
            moved, forward, _ = self._flow(previous, current, points)
            moved = np.asarray(moved, np.float32).reshape(-1, 1, 2).copy()
            forward = np.asarray(forward).reshape(-1)
            if len(moved) != len(names) or len(forward) != len(names):
                raise ValueError("Invalid forward LK output")
            patches = {i: self._patch(current, moved[i, 0]) for i in range(len(names))
                       if forward[i] == 1}
            valid = [i for i, patch in patches.items() if patch is not None]
            for i, name in enumerate(names):
                if i not in valid:
                    self._losses[name] = "forward_lk"
                    self._tips.pop(name, None)
            if not valid:
                return
            # Pass the same frame objects in reverse so the PL pyramid cache
            # recognizes this pair. Copy returned points before another call.
            # Advanced indexing already makes a detached, contiguous copy.
            reverse_points = moved[valid]
            returned, backward, _ = self._flow(current, previous, reverse_points)
            returned = np.asarray(returned, np.float32).reshape(-1, 2).copy()
            backward = np.asarray(backward).reshape(-1)
            if len(returned) != len(valid) or len(backward) != len(valid):
                raise ValueError("Invalid reverse LK output")
            old_patches = [self._patch(previous, points[i, 0]) for i in valid]
            metrics = None
            if self._patch_quality is not None:
                metrics = self._patch_quality.evaluate(
                    np.asarray(old_patches), np.asarray([patches[i] for i in valid]),
                    np.asarray([self._tips[names[i]]["anchor"] for i in valid]),
                    points[valid, 0], returned)
            for j, i in enumerate(valid):
                name = names[i]
                tip = self._tips[name]
                old_patch, patch = old_patches[j], patches[i]
                if metrics is not None:
                    fb, corr, anchor_corr, texture, old_texture = map(float, metrics[j])
                # Preserve float32 decisions at the acceptance boundaries.
                if (metrics is None or abs(fb - 1.5) <= 1e-5 or abs(corr - .88) <= 1e-5
                        or abs(anchor_corr - .88) <= 1e-5 or abs(texture - 5.) <= 1e-5
                        or abs(old_texture - 5.) <= 1e-5):
                    fb = float(np.linalg.norm(returned[j] - points[i, 0]))
                    texture, old_texture = float(patch.std()), float(old_patch.std())
                    corr = self._correlation(old_patch, patch)
                    anchor_corr = self._correlation(tip["anchor"], patch)
                quality = {"fb_error_px": fb, "patch_correlation": corr,
                           "anchor_correlation": anchor_corr, "texture": texture}
                reason = None
                if backward[j] != 1 or not math.isfinite(fb) or fb >= 1.5:
                    reason = "forward_backward"
                elif texture < 5 or old_texture < 5:
                    reason = "texture"
                elif corr < .88 or anchor_corr < .88:
                    reason = "patch_correlation"
                if reason:
                    self._losses[name] = reason
                    self._tips.pop(name, None)
                else:
                    tip["point"] = moved[i, 0].copy()
                    self._quality[name] = quality
        except Exception as exc:
            self._lose_all("flow_backend:" + type(exc).__name__)
            if getattr(self._flow_backend, "fault", False):
                if isinstance(exc, (OSError, RuntimeError)):
                    raise
                raise RuntimeError("Fingertip flow backend is faulted") from exc

    def _seed(self, result, now, sequence):
        identity, source = _identity(result)
        if identity != self._identity:
            return "detector_identity"
        if source > sequence:
            return "detector_future_sequence"
        if source <= self._attempted_sequence:
            return "detector_already_attempted"
        # Finish pending replay before applying a correction from an older
        # frame. Repeated delayed completions must not rewind catch-up progress.
        if (self._tips and len(self._history) >= 2
                and self._tracked_sequence < self._history[-2][0]
                and source < self._tracked_sequence):
            return "detector_correction_deferred"
        self._attempted_sequence = source
        if result.get("fault"):
            return "detector_fault"
        captured = _number(result.get("capture_monotonic"))
        if captured < 0 or captured > now:
            return "detector_timestamp"
        if now - captured > self.MAX_AGE_SECONDS:
            return "detector_expired"
        frame = next((entry for entry in self._history if entry[0] == source), None)
        if frame is None:
            return "source_not_retained"
        if captured > frame[1] + 1e-6:
            return "detector_timestamp"
        roi = result.get("roi")
        if not isinstance(roi, (list, tuple)) or len(roi) != 4:
            return "source_roi"
        x, y, width, height = (_number(v) for v in roi)
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 640 or y + height > 480:
            return "source_roi"
        points = result.get("points")
        if not isinstance(points, dict):
            return "source_points"
        tips, losses = {}, {}
        for name in POINT_NAMES:
            if name not in points:
                losses[name] = "missing_source_point"
                continue
            try:
                point = points[name]
                if not isinstance(point, dict):
                    raise ValueError("Invalid source point")
                visible = point.get("visible", True)
                if isinstance(visible, (bool, np.bool_)) and not visible:
                    losses[name] = "not_visible"
                    continue
                px, py = _number(point["x"]), _number(point["y"])
                raw_score = _number(point["raw_score"])
                if not x <= px < x + width or not y <= py < y + height:
                    raise ValueError("Source point outside ROI")
                scaled = (np.array([px, py], np.float32) + .5) * _SCALE - .5
                patch = self._patch(frame[2], scaled)
                if patch is None:
                    losses[name] = "patch_bounds"
                elif float(patch.std()) < 5:
                    losses[name] = "texture"
                else:
                    tips[name] = {"point": scaled, "raw_score": raw_score, "anchor": patch}
            except (KeyError, TypeError, ValueError, OverflowError):
                losses[name] = "invalid_source_point"
        self._source_sequence, self._captured = source, captured
        self._tracked_sequence = source
        self._has_flow = False
        self._tips, self._losses, self._quality = tips, losses, {}
        return None

    def _result(self, sequence, started, reason=None, steps=0):
        now = _number(self._clock())
        if self._captured is not None and now - self._captured > self.MAX_AGE_SECONDS:
            self._lose_all("detector_expired")
            self._direct_accepted = False
            reason = "detector_expired"
        current = bool(self._tips) and self._tracked_sequence == sequence
        status = "tracking" if current else "pending" if self._tips else "lost"
        if self._source_sequence is None and reason is None:
            status = "pending"
        points = {}
        if current:
            for name, tip in self._tips.items():
                point = (tip["point"] + .5) / _SCALE - .5
                points[name] = {"x": float(point[0]), "y": float(point[1]), "raw_score": tip["raw_score"]}
        return {"status": status, "points": points, "source_sequence": self._source_sequence,
                "sequence": sequence, "stream_id": self._identity[0], "generation": self._identity[1],
                "keypoint_source": "optical_flow" if self._has_flow else "neural", "method": self.method,
                "detector_age_ms": None if self._captured is None else max(0, (now - self._captured) * 1000),
                "processing_ms": (time.perf_counter() - started) * 1000,
                "diagnostics": {"reason": reason, "tracked_sequence": self._tracked_sequence,
                                "history_size": len(self._history), "catchup_steps": steps,
                                "catchup_remaining": sum(entry[0] > self._tracked_sequence for entry in self._history)
                                if self._tips else 0,
                                "budget_exhausted": bool(self._tips) and not current,
                                "direct_attempt": self._direct_attempt,
                                "direct_accepted": self._direct_accepted,
                                "patch_backend": "native_batch" if self._patch_quality is not None else "numpy",
                                "flow_calls": list(self._flow_timings),
                                "lost_tips": dict(self._losses),
                                "quality": {name: dict(self._quality.get(name, {})) for name in self._tips}}}

    def update(self, gray, neural, options, hand_region=None):
        """Track a new frame; hand_region is advisory and never supplies points.

        Invalid frame/options raise ValueError. Invalid detector data or broken
        continuity produce diagnostics and never produce stale coordinates.
        This instance is single-caller, like its caller-owned flow backend.
        """
        started = time.perf_counter()
        self._flow_timings = []
        self._direct_attempt = self._direct_accepted = False
        identity, sequence = _identity(options, "stream")
        now = _number(self._clock())
        if now < 0:
            raise ValueError("clock must be nonnegative")
        if not isinstance(gray, np.ndarray) or gray.dtype != np.uint8 or gray.shape != (480, 640):
            raise ValueError("gray must be a 480x640 uint8 array")
        if identity != self._identity:
            self._history.clear()
            self._identity = identity
            self._last_sequence = self._last_time = None
            self._attempted_sequence = -1
            self._source_sequence = self._captured = self._tracked_sequence = None
            self._has_flow = False
            self._tips, self._losses, self._quality = {}, {}, {}
        reason = None
        if self._last_sequence is not None:
            if sequence <= self._last_sequence or now < self._last_time:
                self._lose_all("frame_order")
                return self._result(sequence, started, "frame_order")
            if now - self._last_time > self.MAX_GAP_SECONDS:
                reason = "frame_gap"
                self._lose_all(reason)
                self._history.clear()
        small = cv2.resize(gray, (320, 200), interpolation=cv2.INTER_AREA)
        self._history.append((sequence, now, small))
        self._last_sequence, self._last_time = sequence, now
        if isinstance(neural, dict) and (neural.get("status") == "fault" or neural.get("fault")):
            self._lose_all("snapshot_fault")
            for source in (neural, neural.get("last_result")):
                try:
                    source_identity, source_sequence = _identity(source)
                except ValueError:
                    continue
                if source_identity == identity and source_sequence <= sequence:
                    self._attempted_sequence = max(self._attempted_sequence, source_sequence)
            return self._result(sequence, started, "snapshot_fault")
        if self._captured is not None and now - self._captured > self.MAX_AGE_SECONDS:
            self._lose_all("detector_expired")
            reason = "detector_expired"
        seeded = False
        if neural is not None:
            try:
                if not isinstance(neural, dict):
                    raise ValueError("Invalid neural snapshot")
                context = neural.get("current_source")
                if context is not None and _identity(context) != (identity, sequence):
                    reason = "snapshot_identity_or_order"
                else:
                    result = neural.get("last_result")
                    if result is None and neural.get("points") is not None:
                        result = neural
                    if result is not None:
                        seed_reason = self._seed(result, now, sequence)
                        seeded = seed_reason is None and bool(self._tips)
                        reason = seed_reason or reason
            except (KeyError, TypeError, ValueError, OverflowError):
                reason = "invalid_detector"
        steps = 0
        if self._tips:
            frames = list(self._history)
            position = next((i for i, entry in enumerate(frames) if entry[0] == self._tracked_sequence), None)
            if position is None:
                self._lose_all("catchup_history_evicted")
                reason = "catchup_history_evicted"
            else:
                if (self.fast_catchup and seeded and len(frames) - position - 1 >= 2
                        and 0 <= _number(self._clock()) - self._captured <= self.MAX_DIRECT_AGE_SECONDS
                        and all(a[0] < b[0] and 0 <= b[1] - a[1] <= self.MAX_GAP_SECONDS
                                for a, b in zip(frames[position:-1], frames[position + 1:]))
                        and steps < self.MAX_STEPS
                        and time.perf_counter() - started < self.BUDGET_SECONDS):
                    tips = {name: dict(tip) for name, tip in self._tips.items()}
                    losses, quality = dict(self._losses), dict(self._quality)
                    self._direct_attempt = True
                    # Backend faults propagate from _step before rollback or replay.
                    self._step(frames[position][2], frames[-1][2])
                    steps += 1
                    if self._tips:
                        self._direct_accepted = self._has_flow = True
                        self._tracked_sequence = sequence
                        position = len(frames) - 1
                    else:
                        self._tips, self._losses, self._quality = tips, losses, quality
                while position + 1 < len(frames) and self._tips:
                    if steps >= self.MAX_STEPS or time.perf_counter() - started >= self.BUDGET_SECONDS:
                        break
                    previous, current = frames[position], frames[position + 1]
                    if current[0] <= previous[0] or current[1] - previous[1] > self.MAX_GAP_SECONDS:
                        self._lose_all("frame_gap")
                        reason = "frame_gap"
                        break
                    self._step(previous[2], current[2])
                    if self._tips:
                        self._has_flow = True
                    self._tracked_sequence = current[0]
                    position += 1
                    steps += 1
        return self._result(sequence, started, reason, steps)

"""Isolated JPEG vision service; every capture source requires explicit opt-in."""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import signal
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit


def _load_cv2():
    try:
        import cv2
    except (ImportError, OSError) as exc:
        raise RuntimeError("OpenCV unavailable; use opencv-python-headless in an isolated "
                           "environment. Import failed: {}".format(exc)) from exc
    return cv2


class VideoSource:
    def __init__(self, camera=None, video=None, camera_format="MJPG"):
        if (camera is None) == (video is None):
            raise ValueError("Select exactly one camera index or video file")
        if camera is not None and (isinstance(camera, bool) or not isinstance(camera, int) or camera < 0):
            raise ValueError("camera must be an explicit nonnegative integer index")
        if camera_format not in ("MJPG", "YUYV"):
            raise ValueError("camera_format must be MJPG or YUYV")
        self.kind = "camera" if camera is not None else "video_file"
        self.target = camera if camera is not None else str(Path(video).resolve())
        self.camera_format = camera_format
        self.provenance = {"synthetic": False}
        self.provenance["camera_index" if camera is not None else "path"] = self.target
        self.capture = None
        self.raw_mjpeg = False

    def open(self, width, height, fps):
        if self.kind == "video_file" and not Path(self.target).is_file():
            raise FileNotFoundError("Video file does not exist: {}".format(self.target))
        cv2 = _load_cv2()
        cv2.setNumThreads(1)
        if self.kind == "camera" and sys.platform.startswith("linux"):
            self.capture = cv2.VideoCapture(self.target, cv2.CAP_V4L2)
        else:
            self.capture = cv2.VideoCapture(self.target)
        if not self.capture.isOpened():
            raise RuntimeError("Could not open {}: {}".format(self.kind, self.target))
        if self.kind == "camera":
            if sys.platform.startswith("linux"):
                from camera_controls import prefer_fixed_frame_rate
                self.provenance["fixed_exposure_cadence"] = prefer_fixed_frame_rate(self.target)
            requested_fourcc = cv2.VideoWriter_fourcc(*self.camera_format)
            self.capture.set(cv2.CAP_PROP_FOURCC, requested_fourcc)
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_FPS, fps)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)
            fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
            self.provenance["capture_mode"] = {
                "requested_fourcc": self.camera_format,
                "fourcc": "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4)),
                "width": self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                "height": self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                "fps": self.capture.get(cv2.CAP_PROP_FPS),
                "buffer_size": self.capture.get(cv2.CAP_PROP_BUFFERSIZE),
            }
            if self.camera_format == "YUYV" and fourcc != requested_fourcc:
                raise RuntimeError("Camera did not accept YUYV; negotiated {!r}".format(
                    self.provenance["capture_mode"]["fourcc"]))
            self.raw_mjpeg = (self.camera_format == "MJPG" and sys.platform.startswith("linux") and
                fourcc == cv2.VideoWriter_fourcc(*"MJPG") and
                bool(self.capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)))
            self.provenance["capture_mode"]["jpeg_passthrough"] = self.raw_mjpeg

    def read(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


class SyntheticSource:
    kind = "synthetic"

    def __init__(self):
        self.provenance = {"synthetic": True, "pattern": "labeled moving test pattern"}
        self.index = 0

    def open(self, width, height, fps):
        _load_cv2()

    def read(self):
        cv2 = _load_cv2()
        try:
            import numpy as np
        except (ImportError, OSError) as exc:
            raise RuntimeError("Synthetic frames require working numpy: {}".format(exc)) from exc
        frame = np.full((360, 640, 3), 24, dtype=np.uint8)
        for index, color in enumerate(((200, 80, 40), (40, 200, 80), (80, 40, 200))):
            cv2.rectangle(frame, (30 + 200 * index, 50), (180 + 200 * index, 270), color, -1)
        cv2.line(frame, (30 + self.index % 580, 40), (30 + self.index % 580, 280), (255, 255, 255), 3)
        cv2.putText(frame, "SYNTHETIC - frame {}".format(self.index), (25, 325),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        self.index += 1
        return frame

    def close(self):
        pass


def create_source(camera=None, video=None, synthetic=False, camera_format="MJPG"):
    """Construct a source without importing OpenCV or opening any device."""
    if sum((camera is not None, video is not None, bool(synthetic))) != 1:
        raise ValueError("Select exactly one of camera, video, or synthetic")
    if camera_format not in ("MJPG", "YUYV"):
        raise ValueError("camera_format must be MJPG or YUYV")
    return SyntheticSource() if synthetic else VideoSource(camera=camera, video=video,
                                                           camera_format=camera_format)


def encode_frame(frame, max_width=1280, max_height=720, quality=70):
    cv2 = _load_cv2()
    if (getattr(frame, "ndim", None) != 3 or frame.shape[2] != 3
            or getattr(frame.dtype, "name", None) != "uint8"
            or frame.shape[0] == 0 or frame.shape[1] == 0):
        raise ValueError("Capture must supply a nonempty uint8 HxWx3 BGR frame")
    height, width = frame.shape[:2]
    source_resolution = {"width": width, "height": height}
    scale = min(1.0, max_width / width, max_height / height)
    if scale < 1:
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok or encoded is None or not encoded.size:
        raise RuntimeError("OpenCV could not encode a JPEG")
    return encoded.tobytes(), {"width": width, "height": height}, source_resolution


class VisionState:
    def __init__(self, source, fps=30, stale_after=1.0, max_width=1280,
                 max_height=720, jpeg_quality=70, clock=time.monotonic, yolo_detector=None):
        if not math.isfinite(fps) or not 0 < fps <= 30:
            raise ValueError("fps must be finite and greater than 0, at most 30")
        if not math.isfinite(stale_after) or stale_after <= 0:
            raise ValueError("stale_after must be finite and positive")
        for name, value, maximum in (("max_width", max_width, 1280),
                                     ("max_height", max_height, 720), ("jpeg_quality", jpeg_quality, 100)):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError("{} must be an integer from 1 to {}".format(name, maximum))
        self.source, self.fps, self.stale_after = source, fps, stale_after
        self.max_width, self.max_height, self.jpeg_quality = max_width, max_height, jpeg_quality
        self.clock = clock
        self._lock = threading.Lock()
        self._jpeg = None
        self._sequence = 0
        self._captured_ms = None
        self._resolution = None
        self._source_resolution = None
        self._status = "starting"
        self._error = None
        self.stream_id = uuid.uuid4().hex
        self.started_ms = self.clock() * 1000
        self.boot_id = None
        if sys.platform.startswith("linux"):
            try:
                self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            except OSError:
                pass
        self._capture_failures = 0
        self._last_error = None
        self._last_error_ms = None
        self._invalid_frame_drops = 0
        self._last_invalid_frame = None
        self._last_invalid_frame_ms = None
        self._gray_history = deque(maxlen=10)
        self._jpeg_history = deque(maxlen=30)
        self._trackers = {}
        self._auto_lock = threading.Lock()
        self._auto_tracker = None
        self._auto_cached = None
        self._auto_at = None
        self._capture_generation = 0
        self._auto_thread = None
        self._auto_roi = None
        self._auto_roi_active = False
        self._published_times = deque(maxlen=60)
        self._processing_ms = None
        self.software_jpeg_encoder = "opencv"
        self.yolo_detector = yolo_detector

    def auto_frame_snapshot(self):
        """Serve live preview without waiting for the background detector.

        Never queue jobs or delay preview waiting for recognition. Geometry
        from an older frame is not attached to a moving, newer image.
        """
        with self._lock:
            metadata, jpeg = self._frame_snapshot_locked(None)
            if jpeg is None:
                return metadata, None
            analysis_metadata, analysis_jpeg = dict(metadata), jpeg
            cached = self._auto_cached
            if cached is not None and cached[0] == self._capture_generation:
                result = cached[1]
                # ROI mode shows matched analysis frames at its measured cadence.
                # Clients reject decreasing sequences, so playback cannot rewind.
                age = max(0, self.clock() * 1000 - cached[2]["captured_ms"])
                if age <= min(500 if self._auto_roi_active else 80, self.stale_after * 1000):
                    metadata = dict(cached[2], age_ms=age, fretboard=result)
                    jpeg = cached[3]
                else:
                    reason = ("detector_too_slow" if result["inference_ms"] >
                              self.stale_after * 1000 else "analysis_pending")
                    metadata["fretboard"] = self._pending_auto(metadata, reason)
            else:
                metadata["fretboard"] = self._pending_auto(metadata)
            if (self._auto_thread is None or not self._auto_thread.is_alive()) and (
                    self._auto_at is None or self.clock() - self._auto_at >=
                    (.025 if self._auto_roi_active else .2)):
                gray = (self._gray_history[-1][1] if self._gray_history and
                        self._gray_history[-1][0] == analysis_metadata["sequence"] else None)
                self._auto_thread = threading.Thread(target=self._analyze_auto,
                    args=(gray, analysis_metadata, self._capture_generation, analysis_jpeg),
                    name="vision-analysis", daemon=True)
                self._auto_thread.start()
            return metadata, jpeg

    def _pending_auto(self, metadata, reason="analysis_pending"):
        return {"mode": "auto", "status": "searching", "reason": reason,
                "sequence": metadata["sequence"], "stream_id": self.stream_id,
                "corners": None, "strings": [], "frets": []}

    def set_auto_roi(self, payload):
        """Seed ROI tracking from the exact displayed JPEG, never a newer image."""
        sequence, corners = payload.get("sequence"), payload.get("corners")
        if (isinstance(sequence, bool) or not isinstance(sequence, int) or
                not isinstance(corners, list) or len(corners) != 4):
            raise ValueError("Expected a frame sequence and four normalized corners")
        for point in corners:
            if not isinstance(point, dict) or any(isinstance(point.get(key), bool) or
                    not isinstance(point.get(key), (int, float)) or
                    not math.isfinite(point[key]) or not 0 <= point[key] <= 1 for key in ("x", "y")):
                raise ValueError("Corners must be finite normalized x/y points")
        with self._lock:
            if payload.get("stream_id") != self.stream_id:
                raise ValueError("Camera stream changed; select the neck again")
            metadata, current = self._frame_snapshot_locked(None)
            if current is None:
                raise ValueError("Camera frame is unavailable or stale")
            matched = next((body for seq, body in self._jpeg_history if seq == sequence), None)
            if matched is None:
                raise ValueError("Selection frame expired; select the neck again")
            self._capture_generation += 1
            self._auto_roi = {"jpeg": matched, "corners": [dict(point) for point in corners]}
            self._auto_roi_active = True
            self._auto_cached = None
            self._auto_at = None
            return {"status": "pending", "sequence": sequence, "stream_id": self.stream_id,
                    "backend": "roi_lk_sobel_projection"}

    def _analyze_auto(self, gray, metadata, generation, jpeg):
        with self._auto_lock:
            from auto_fretboard import AutoFretboardTracker
            started = self.clock()
            try:
                if self._auto_tracker is None or getattr(self, "_auto_generation", None) != generation:
                    with self._lock:
                        if generation != self._capture_generation:
                            return
                        roi = self._auto_roi
                    if roi is not None:
                        from roi_fretboard import RoiFretboardTracker
                        self._auto_tracker = RoiFretboardTracker()
                        import numpy as np
                        seed = _load_cv2().imdecode(np.frombuffer(roi["jpeg"], np.uint8),
                                                   _load_cv2().IMREAD_GRAYSCALE)
                        self._auto_tracker.initialize(seed, roi["corners"])
                    elif self.yolo_detector is not None:
                        from yolo_fretboard_pipeline import YoloFretboardPipeline
                        self._auto_tracker = YoloFretboardPipeline(self.yolo_detector)
                    else:
                        self._auto_tracker = AutoFretboardTracker()
                    self._auto_generation = generation
                if gray is None:
                    import numpy as np
                    gray = _load_cv2().imdecode(np.frombuffer(jpeg, np.uint8),
                                               _load_cv2().IMREAD_GRAYSCALE)
                if self.yolo_detector is not None and not self._auto_roi_active:
                    import numpy as np
                    cv2 = _load_cv2()
                    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                    if bgr is not None and gray is not None and bgr.shape[:2] != gray.shape:
                        bgr = cv2.resize(bgr, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_AREA)
                    result = self._auto_tracker.update(gray, bgr)
                else:
                    result = self._auto_tracker.update(gray)
            except Exception:
                result = (self._auto_tracker.invalidate("detection_error") if self._auto_tracker is not None
                          else self._pending_auto(metadata, "detection_error"))
            result = dict(result, sequence=metadata["sequence"], stream_id=self.stream_id,
                          inference_ms=round(max(0, self.clock() - started) * 1000, 3))
            with self._lock:
                if generation == self._capture_generation:
                    self._auto_cached = generation, result, metadata, jpeg
                    self._auto_at = self.clock()

    def publish(self, jpeg, resolution, source_resolution, captured_at, gray=None):
        with self._lock:
            # Calibration can register a tracker while capture prepares this frame.
            if gray is None and self._trackers:
                import numpy as np
                cv2 = _load_cv2()
                gray = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise RuntimeError("Current tracking frame cannot be decoded")
                if self._gray_history:
                    target_height, target_width = self._gray_history[-1][1].shape
                else:
                    scale = min(1.0, min(640, self.max_width) / gray.shape[1])
                    target_width, target_height = round(gray.shape[1] * scale), round(gray.shape[0] * scale)
                if gray.shape != (target_height, target_width):
                    gray = cv2.resize(gray, (target_width, target_height))
            self._jpeg = jpeg
            self._sequence += 1
            self._jpeg_history.append((self._sequence, jpeg))
            self._captured_ms = captured_at * 1000
            self._published_times.append(self.clock())
            self._processing_ms = max(0, self.clock() - captured_at) * 1000
            self._resolution = dict(resolution)
            self._source_resolution = dict(source_resolution)
            self._status, self._error = "streaming", None
            if gray is not None:
                self._gray_history.append((self._sequence, gray))
                for identity, entry in list(self._trackers.items()):
                    if self.clock() - entry["used_at"] > 60:
                        del self._trackers[identity]
                        continue
                    if entry["sequence"] < self._sequence:
                        try:
                            entry["tracker"].update(gray)
                        except Exception:
                            entry["tracker"].invalidate("tracking_error")
                        entry["sequence"] = self._sequence

    def start_tracking(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        corners, sequence = payload.get("corners"), payload.get("sequence")
        if (isinstance(sequence, bool) or not isinstance(sequence, int) or
                not isinstance(corners, list) or len(corners) != 4):
            raise ValueError("Expected a frame sequence and four normalized corners")
        for point in corners:
            if not isinstance(point, dict) or any(isinstance(point.get(key), bool) or
                    not isinstance(point.get(key), (int, float)) or
                    not math.isfinite(point[key]) or not 0 <= point[key] <= 1 for key in ("x", "y")):
                raise ValueError("Corners must be finite normalized x/y points")
        from fretboard_tracker import FretboardTracker
        with self._lock:
            if payload.get("stream_id") != self.stream_id:
                raise ValueError("Camera stream changed; recalibrate")
            if (self._status != "streaming" or self._captured_ms is None or
                    self.clock() * 1000 - self._captured_ms > self.stale_after * 1000):
                raise ValueError("Camera frame is unavailable or stale")
            frames = list(self._gray_history)
            anchor = next((index for index, frame in enumerate(frames) if frame[0] == sequence), None)
            if anchor is None:
                matched = next((body for seq, body in self._jpeg_history if seq == sequence), None)
                if matched is None:
                    raise ValueError("Calibration frame expired; recalibrate")
                import numpy as np
                cv2 = _load_cv2()
                gray = cv2.imdecode(np.frombuffer(matched, np.uint8), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError("Calibration frame cannot be decoded")
                if frames and gray.shape != frames[-1][1].shape:
                    gray = cv2.resize(gray, (frames[-1][1].shape[1], frames[-1][1].shape[0]))
                frames = [(sequence, gray)] + [frame for frame in frames if frame[0] > sequence]
                anchor = 0
            if frames[-1][0] != self._sequence:
                import numpy as np
                cv2 = _load_cv2()
                latest = cv2.imdecode(np.frombuffer(self._jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
                if latest is None:
                    raise ValueError("Current tracking frame cannot be decoded")
                if latest.shape != frames[-1][1].shape:
                    latest = cv2.resize(latest, (frames[-1][1].shape[1], frames[-1][1].shape[0]))
                frames.append((self._sequence, latest))
            # Replace this client's previous tracker without affecting other tabs.
            previous = payload.get("tracker_id")
            if isinstance(previous, str):
                self._trackers.pop(previous, None)
            self._trackers = {key: entry for key, entry in self._trackers.items()
                              if self.clock() - entry["used_at"] <= 60}
            if len(self._trackers) >= 4:
                raise ValueError("Too many active trackers; close another video tab")
            tracker = FretboardTracker()
            initialized = tracker.initialize(frames[anchor][1], corners)
            if initialized["status"] != "tracking":
                raise ValueError("Cannot lock fretboard: {}".format(initialized.get("reason")))
            for _, gray in frames[anchor + 1:]:
                tracker.update(gray)
            if tracker.snapshot()["status"] != "tracking":
                raise ValueError("Fretboard moved too far during calibration; recalibrate")
            identity = uuid.uuid4().hex
            self._trackers[identity] = {"tracker": tracker, "sequence": self._sequence,
                                       "used_at": self.clock()}
            return self._tracking_snapshot(identity)

    def _tracking_snapshot(self, identity):
        entry = self._trackers.get(identity)
        if entry is None:
            return {"tracker_id": identity, "status": "lost", "corners": None,
                    "reason": "tracker_expired", "sequence": self._sequence, "stream_id": self.stream_id}
        entry["used_at"] = self.clock()
        return dict(entry["tracker"].snapshot(), tracker_id=identity,
                    sequence=entry["sequence"], stream_id=self.stream_id)

    def stop_tracking(self, identity):
        if not isinstance(identity, str):
            raise ValueError("Expected tracker_id")
        with self._lock:
            self._trackers.pop(identity, None)

    def drop_invalid_frame(self, reason):
        with self._lock:
            self._invalid_frame_drops += 1
            self._last_invalid_frame = str(reason)
            self._last_invalid_frame_ms = self.clock() * 1000

    def fail(self, error):
        with self._lock:
            self._capture_generation += 1
            self._auto_cached = self._auto_at = None
            self._published_times.clear()
            self._status, self._error, self._jpeg = "error", str(error), None
            self._capture_failures += 1
            self._last_error = str(error)
            self._last_error_ms = self.clock() * 1000
            self._gray_history.clear()
            self._auto_roi = None
            self._auto_roi_active = False
            self._jpeg_history.clear()
            for entry in self._trackers.values():
                entry["tracker"].invalidate("capture_failed")

    def stopped(self):
        with self._lock:
            self._capture_generation += 1
            self._auto_cached = self._auto_at = None
            self._published_times.clear()
            self._status, self._jpeg = "stopped", None
            self._gray_history.clear()
            self._auto_roi = None
            self._auto_roi_active = False
            self._jpeg_history.clear()
            for entry in self._trackers.values():
                entry["tracker"].invalidate("capture_stopped")

    def frame_snapshot(self, tracking_id=None):
        """Metadata and bytes come from the same publication under one lock."""
        with self._lock:
            return self._frame_snapshot_locked(tracking_id)

    def _frame_snapshot_locked(self, tracking_id):
        age_ms = None if self._captured_ms is None else max(0.0, self.clock() * 1000 - self._captured_ms)
        fresh = (self._status == "streaming" and self._jpeg is not None
                 and age_ms is not None and age_ms <= self.stale_after * 1000)
        status = "stale" if self._status == "streaming" and not fresh else self._status
        metadata = {
            "schema_version": 1, "status": status, "source": self.source.kind,
            "stream_id": self.stream_id, "boot_id": self.boot_id,
            "service_started_ms": self.started_ms,
            "capture_failures": self._capture_failures,
            "last_error": self._last_error, "last_error_ms": self._last_error_ms,
            "invalid_frame_drops": self._invalid_frame_drops,
            "last_invalid_frame": self._last_invalid_frame,
            "last_invalid_frame_ms": self._last_invalid_frame_ms,
            "provenance": dict(self.source.provenance),
            "synthetic": bool(self.source.provenance.get("synthetic", False)),
            "sequence": self._sequence, "captured_ms": self._captured_ms,
            "timestamp_clock": "process_monotonic", "age_ms": age_ms,
            "stale": not fresh, "frame_available": fresh, "error": self._error,
            "resolution": None if self._resolution is None else dict(self._resolution),
            "source_resolution": None if self._source_resolution is None else dict(self._source_resolution),
            "max_resolution": {"width": self.max_width, "height": self.max_height},
            "fps": self.fps, "jpeg_quality": self.jpeg_quality,
            "capture_fps": ((len(self._published_times) - 1) /
                            (self._published_times[-1] - self._published_times[0])
                            if len(self._published_times) > 1 and
                            self._published_times[-1] > self._published_times[0] else 0.0),
            "processing_ms": self._processing_ms,
            "software_jpeg_encoder": self.software_jpeg_encoder,
            "timestamp_stage": "after_opencv_read_before_processing",
            "stale_after_ms": self.stale_after * 1000,
            "detection_backend": ("roi_lk_sobel_projection" if self._auto_roi_active else
                                  "yolo_roi_then_observed_lines" if self.yolo_detector is not None
                                  else "opencv_observed_lines"),
        }
        if tracking_id is not None:
            metadata["tracking"] = self._tracking_snapshot(tracking_id)
        return metadata, self._jpeg if fresh else None

    def snapshot(self):
        return self.frame_snapshot()[0]


class CaptureWorker:
    def __init__(self, state, encoder=encode_frame):
        self.state, self.encoder = state, encoder
        backend = getattr(encoder, "backend_name", None)
        state.software_jpeg_encoder = (backend if isinstance(backend, str) else
                                       "opencv" if encoder is encode_frame else "custom")
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="vision-capture", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self, timeout=2.0):
        self._stop.set()
        self.state.stopped()
        if self.thread.ident is not None:
            self.thread.join(timeout)
        return not self.thread.is_alive()

    def _run(self):
        source, state = self.state.source, self.state
        raw_resolution, last_gray = None, -math.inf
        while not self._stop.is_set():
            cleanup_failed = False
            try:
                source.open(state.max_width, state.max_height, state.fps)
                raw_resolution, last_gray = None, -math.inf
                while not self._stop.is_set():
                    started = state.clock()
                    frame = source.read()
                    captured_at = state.clock()
                    if frame is None:
                        raise RuntimeError("{} returned no frame (read failure or end of file)".format(source.kind))
                    gray = None
                    if isinstance(source, VideoSource) and source.raw_mjpeg and frame.ndim == 2:
                        cv2 = _load_cv2()
                        jpeg = frame.tobytes()
                        if not jpeg.startswith(b"\xff\xd8"):
                            state.drop_invalid_frame("Camera raw JPEG is missing SOI")
                            continue
                        end = jpeg.rfind(b"\xff\xd9")
                        if end < 2:
                            state.drop_invalid_frame("Camera raw JPEG is missing EOI")
                            continue
                        # Some capture buffers include padding after the JPEG.
                        if end + 2 < len(jpeg):
                            jpeg = jpeg[:end + 2]
                            frame = frame.reshape(-1)[:end + 2]
                        # Keep calibration history at 10 Hz; live trackers need
                        # every frame. Preview itself never waits for inference.
                        if raw_resolution is None or state._trackers or captured_at - last_gray >= .1:
                            gray = cv2.imdecode(frame, cv2.IMREAD_GRAYSCALE)
                            if gray is None:
                                state.drop_invalid_frame("Camera JPEG grayscale decode returned no frame")
                                continue
                            raw_resolution = {"width": gray.shape[1], "height": gray.shape[0]}
                            last_gray = captured_at
                        resolution = original = raw_resolution
                        if original["width"] > state.max_width or original["height"] > state.max_height:
                            gray_was_requested = gray is not None
                            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                            if frame is None:
                                state.drop_invalid_frame("Camera JPEG color decode returned no frame")
                                continue
                            jpeg, resolution, original = self.encoder(frame, state.max_width,
                                state.max_height, state.jpeg_quality)
                            gray = None
                            if gray_was_requested:
                                last_gray = -math.inf
                    else:
                        jpeg, resolution, original = self.encoder(frame, state.max_width, state.max_height,
                                                                 state.jpeg_quality)
                    if self._stop.is_set():
                        break
                    if (gray is None and getattr(frame, "ndim", None) == 3 and
                            (last_gray == -math.inf or state._trackers or captured_at - last_gray >= .1)):
                        cv2 = _load_cv2()
                        scale = min(1.0, min(640, state.max_width) / frame.shape[1])
                        small = cv2.resize(frame, (round(frame.shape[1] * scale),
                                                  round(frame.shape[0] * scale))) if scale < 1 else frame
                        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                        last_gray = captured_at
                    state.publish(jpeg, resolution, original, captured_at, gray)
                    # V4L2 read is paced by the camera. Keep draining buffers;
                    # an extra sleep here creates a backlog under ARM load.
                    if source.kind != "camera":
                        self._stop.wait(max(0.0, 1 / state.fps - (state.clock() - started)))
            except Exception as exc:
                state.fail("{}: {}".format(type(exc).__name__, exc))
                print("Capture failure: {}: {}".format(type(exc).__name__, exc),
                      file=sys.stderr, flush=True)
            finally:
                try:
                    source.close()
                except Exception as exc:
                    cleanup_failed = True
                    state.fail("Source cleanup failed: {}".format(exc))
            # USB cameras may appear late at boot or disconnect between reads.
            if cleanup_failed or source.kind != "camera" or self._stop.wait(1.0):
                break
        if self._stop.is_set():
            state.stopped()


class CameraWatchdog(threading.Thread):
    """Report a stuck capture so the external supervisor can reset the board."""

    def __init__(self, state, restart_after=8.0, interval=.5,
                 exit_process=None):
        super().__init__(name="vision-camera-watchdog", daemon=True)
        if not math.isfinite(restart_after) or restart_after <= 0:
            raise ValueError("restart_after must be finite and positive")
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval must be finite and positive")
        self.state, self.restart_after, self.interval = state, restart_after, interval
        self._stop_event = threading.Event()
        self._exit_process = exit_process

    def stop(self):
        self._stop_event.set()

    def run(self):
        unhealthy_status = None
        unhealthy_since = None
        while not self._stop_event.wait(self.interval):
            status = self.state.snapshot().get("status")
            now = self.state.clock()
            if status not in ("starting", "stale", "error"):
                unhealthy_status = unhealthy_since = None
                continue
            if status != unhealthy_status:
                unhealthy_status, unhealthy_since = status, now
                continue
            if now - unhealthy_since < self.restart_after:
                continue
            print("Camera watchdog: {} for {:.1f}s; external reset required".format(
                status, now - unhealthy_since), file=sys.stderr, flush=True)
            # A V4L2 read can leave a kernel thread in uninterruptible D state.
            # Exiting the Python process in that case leaves a zombie service
            # and a dead listening socket.  The external root supervisor resets
            # the PS instead; keep this HTTP process alive so status is visible.
            if self._exit_process is not None:
                self._exit_process(1)
            return


def make_server(state, host="127.0.0.1", port=8082):
    """Create HTTP listeners without starting capture or importing OpenCV."""
    class Handler(BaseHTTPRequestHandler):
        # The browser polls small JPEGs continuously. Reusing the TCP session
        # avoids a handshake and a new worker thread for every camera frame.
        protocol_version = "HTTP/1.1"

        def reply(self, code, body, content_type="application/json; charset=utf-8", headers=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(body)

        def json_reply(self, code, payload):
            self.reply(code, json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8"))

        def stream_mjpeg(self):
            """Keep one HTTP connection open and push every fresh camera frame."""
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Pragma", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            last_sequence = None
            try:
                while True:
                    metadata, jpeg = state.frame_snapshot()
                    if jpeg is None:
                        time.sleep(.02)
                        continue
                    sequence = metadata["sequence"]
                    if sequence == last_sequence:
                        time.sleep(.005)
                        continue
                    part_header = (
                        "--frame\r\n"
                        "Content-Type: image/jpeg\r\n"
                        "Content-Length: {length}\r\n"
                        "X-Vision-Sequence: {sequence}\r\n"
                        "X-Vision-Captured-Ms: {captured_ms}\r\n"
                        "X-Vision-Source: {source}\r\n"
                        "X-Vision-Stream: {stream}\r\n\r\n"
                    ).format(length=len(jpeg), sequence=sequence,
                             captured_ms=metadata["captured_ms"], source=metadata["source"],
                             stream=metadata["stream_id"]).encode("ascii")
                    self.wfile.write(part_header)
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                    last_sequence = sequence
            except (BrokenPipeError, ConnectionResetError, OSError):
                self.close_connection = True

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/vision/status":
                self.json_reply(200, state.snapshot())
            elif path == "/vision/stream.mjpeg":
                self.stream_mjpeg()
            elif path == "/vision/frame.jpg":
                from urllib.parse import parse_qs
                query = parse_qs(urlsplit(self.path).query)
                identity = query.get("tracking", [None])[0]
                metadata, jpeg = (state.auto_frame_snapshot() if query.get("auto") == ["1"]
                                  else state.frame_snapshot(identity))
                if jpeg is None:
                    self.json_reply(503, metadata)
                else:
                    headers = {
                        "X-Vision-Sequence": metadata["sequence"],
                        "X-Vision-Captured-Ms": metadata["captured_ms"],
                        "X-Vision-Source": metadata["source"],
                        "X-Vision-Stream": metadata["stream_id"],
                    }
                    if "tracking" in metadata:
                        headers["X-Vision-Tracking"] = json.dumps(metadata["tracking"],
                                                                 allow_nan=False, separators=(",", ":"))
                    if "fretboard" in metadata:
                        headers["X-Vision-Fretboard"] = json.dumps(metadata["fretboard"],
                                                                 allow_nan=False, separators=(",", ":"))
                        # Existing local proxies already forward this geometry header.
                        headers["X-Vision-Tracking"] = headers["X-Vision-Fretboard"]
                    self.reply(200, jpeg, "image/jpeg", headers)
            else:
                self.json_reply(404, {"error": "unknown_endpoint"})

        def do_POST(self):
            path = urlsplit(self.path).path
            if path not in ("/api/vision/tracking", "/api/vision/tracking/stop", "/api/vision/roi"):
                self.json_reply(404, {"error": "unknown_endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("JSON body must be 1..4096 bytes")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                if path == "/api/vision/roi":
                    self.json_reply(200, state.set_auto_roi(payload))
                elif path.endswith("/stop"):
                    state.stop_tracking(payload.get("tracker_id"))
                    self.json_reply(200, {"status": "stopped"})
                else:
                    self.json_reply(200, state.start_tracking(payload))
            except (ValueError, TypeError) as error:
                self.json_reply(400, {"error": str(error)})
            except (ImportError, RuntimeError) as error:
                self.json_reply(503, {"error": str(error)})

        def log_message(self, format_string, *args):
            pass

    return ThreadingHTTPServer((host, port), Handler)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--camera", type=int, help="explicit camera device index; no default camera")
    sources.add_argument("--video", type=Path, help="local video file; stops at EOF without fallback")
    sources.add_argument("--synthetic", action="store_true", help="explicitly labeled synthetic test pattern")
    parser.add_argument("--camera-format", choices=("MJPG", "YUYV"), default="MJPG",
                        help="requested camera pixel format (MJPG default)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--max-height", type=int, default=720)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--jpeg-encoder", choices=("opencv", "native-fast"), default="opencv",
                        help="software JPEG encoder; native-fast requires locally built libfast_jpeg.so")
    parser.add_argument("--stale-after", type=float, default=1.0, help="freshness timeout in seconds")
    parser.add_argument("--yolo-model", type=Path, help="optional custom fretboard YOLO ONNX; never downloaded implicitly")
    parser.add_argument("--yolo-metadata", type=Path, help="required explicit class/output contract for --yolo-model")
    args = parser.parse_args(argv)
    try:
        if (args.yolo_model is None) != (args.yolo_metadata is None):
            raise ValueError("--yolo-model and --yolo-metadata must be supplied together")
        if args.camera_format == "YUYV" and args.camera is None:
            raise ValueError("--camera-format YUYV requires --camera")
        source = create_source(args.camera, args.video, args.synthetic, args.camera_format)
        VisionState(source, args.fps, args.stale_after, args.max_width, args.max_height, args.jpeg_quality)
        if not 0 <= args.port <= 65535:
            raise ValueError("port must be between 0 and 65535")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    source = create_source(args.camera, args.video, args.synthetic, args.camera_format)
    state = VisionState(source, args.fps, args.stale_after, args.max_width, args.max_height, args.jpeg_quality)
    if args.yolo_model is not None:
        from yolo_detector import YoloDetector
        state.yolo_detector = YoloDetector(args.yolo_model, args.yolo_metadata, class_filter=["fretboard"])
    encoder = encode_frame
    if args.jpeg_encoder == "native-fast":
        from fast_jpeg import FastJpegEncoder
        encoder = FastJpegEncoder()
        encoder._load_library()
    server = make_server(state, args.host, args.port)
    server.timeout = .2
    worker = CaptureWorker(state, encoder)
    watchdog = CameraWatchdog(state) if source.kind == "camera" else None
    stop = threading.Event()
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: stop.set())
        worker.start()
        if watchdog is not None:
            watchdog.start()
        print("Vision server: http://{}:{} ({})".format(*server.server_address[:2], source.kind), flush=True)
        while not stop.is_set():
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if watchdog is not None:
            watchdog.stop()
            watchdog.join(1)
        if not worker.stop():
            print("Capture shutdown timed out; native source read is still blocked", file=sys.stderr)
        server.server_close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

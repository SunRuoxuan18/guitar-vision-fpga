"""Independent PL preprocessing preview; consumes the existing camera service."""

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import signal
import threading
import time
from urllib.request import urlopen


class PLPreview:
    def __init__(self, processor, camera_url="http://127.0.0.1:8082", fps=5,
                 stale_after=1.0, clock=time.monotonic):
        if not math.isfinite(fps) or not 0 < fps <= 30 or not 0 < stale_after <= 10:
            raise ValueError("Invalid preview cadence or freshness bound")
        self.processor, self.camera_url = processor, camera_url.rstrip("/")
        self.period, self.stale_after, self.clock = 1 / fps, stale_after, clock
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, name="vision-pl", daemon=True)
        self.jpeg, self.metadata, self.error = None, None, None
        self.reported_error = None
        self.completed, self.failures, self.times = 0, 0, deque(maxlen=30)
        self.hardware = getattr(processor, "fingerprint", {})

    def process_frame(self, jpeg, headers):
        import cv2
        import numpy as np
        captured_ms = float(headers["X-Vision-Captured-Ms"])
        sequence = int(headers["X-Vision-Sequence"])
        stream = headers["X-Vision-Stream"]
        source = headers["X-Vision-Source"]
        if source not in ("camera", "video_file", "synthetic") or not stream or sequence < 0:
            raise ValueError("Invalid camera identity")
        if not 0 <= self.clock() * 1000 - captured_ms <= self.stale_after * 1000:
            raise ValueError("Camera frame is stale or has an invalid timestamp")
        with self.lock:
            if self.metadata and stream == self.metadata["stream_id"] and sequence <= self.metadata["sequence"]:
                return False
        started = self.clock()
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Camera JPEG could not be decoded")
        decode_ms = (self.clock() - started) * 1000
        result = self.processor.process(frame)
        composite = np.concatenate((result["gray"], result["edge"]), axis=1)
        encoded_ok, encoded = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not encoded_ok:
            raise RuntimeError("PL preview JPEG encoding failed")
        if self.clock() * 1000 - captured_ms > self.stale_after * 1000:
            raise RuntimeError("PL result expired before publication")
        metadata = {"backend": "pl", "source": source, "sequence": sequence,
                    "stream_id": stream, "captured_ms": captured_ms,
                    "resolution": {"width": frame.shape[1], "height": frame.shape[0]},
                    "layout": "gray_left_edge_right", "threshold": 64,
                    "sobel_window": "trailing_3x3_first_two_rows_and_columns_zero",
                    "decode_ms": decode_ms, "timings": result.get("timings_ms", {}),
                    "processing_ms": (self.clock() - started) * 1000,
                    "hardware": result.get("hardware_metadata", {})}
        with self.lock:
            self.jpeg, self.metadata, self.error = encoded.tobytes(), metadata, None
            self.completed += 1
            self.times.append(self.clock())
        return True

    def snapshot(self):
        with self.lock:
            payload = dict(self.metadata or {"backend": "pl", "source": None, "hardware": self.hardware})
            age = None if not self.metadata else max(0, self.clock() * 1000 - self.metadata["captured_ms"])
            fresh = self.jpeg is not None and self.error is None and age is not None and age <= self.stale_after * 1000
            payload.update(status="ready" if fresh else "error" if self.error else "starting" if not self.metadata else "stale",
                           stale=not fresh, age_ms=age, error=self.error,
                           completed=self.completed, failures=self.failures,
                           processing_fps=((len(self.times) - 1) / (self.times[-1] - self.times[0])
                                           if len(self.times) > 1 and self.times[-1] > self.times[0] else 0))
            return payload, self.jpeg if fresh else None

    def run(self):
        while not self.stop_event.is_set():
            started = self.clock()
            try:
                with urlopen(self.camera_url + "/vision/frame.jpg", timeout=1) as response:
                    jpeg, headers = response.read(4 * 1024 * 1024), response.headers
                if not self.stop_event.is_set():
                    self.process_frame(jpeg, headers)
                    self.reported_error = None
            except Exception as exc:
                with self.lock:
                    self.failures += 1
                    self.error = "{}: {}".format(type(exc).__name__, exc)
                if self.error != self.reported_error:
                    print("PL preview: {}".format(exc), flush=True)
                    self.reported_error = self.error
                # DMA failure poisons the driver: it must not be retried or
                # replaced by an unlabelled CPU implementation.
                from vision_pl import VisionPLError
                if isinstance(exc, VisionPLError):
                    return
            self.stop_event.wait(max(0, self.period - (self.clock() - started)))


def make_server(state, host="0.0.0.0", port=8083):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            payload, jpeg = self.server.preview_state.snapshot()
            if path not in ("/api/vision/pl/status", "/vision/pl.jpg"):
                self.send_error(404)
                return
            image = path.endswith(".jpg") and jpeg is not None
            body = jpeg if image else json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(200 if image or not path.endswith(".jpg") else 503)
            self.send_header("Content-Type", "image/jpeg" if image else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if image:
                for suffix, key in (("Sequence", "sequence"), ("Stream", "stream_id"),
                                    ("Source", "source"), ("Captured-Ms", "captured_ms"),
                                    ("Backend", "backend"), ("Processing-Ms", "processing_ms")):
                    self.send_header("X-Vision-" + suffix, str(payload[key]))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.preview_state = state
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bitfile", required=True, type=Path)
    parser.add_argument("--load-overlay", action="store_true",
                        help="Explicitly replace PL; stop USB capture before using this option")
    parser.add_argument("--camera-url", default="http://127.0.0.1:8082")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--fps", type=float, default=5)
    args = parser.parse_args(argv)
    if not math.isfinite(args.fps) or not 0 < args.fps <= 30:
        parser.error("--fps must be finite, greater than zero and at most 30")
    if not 0 <= args.port <= 65535:
        parser.error("--port must be between 0 and 65535")
    import cv2
    cv2.setNumThreads(1)
    from vision_pl import VisionPL
    processor, state, server = None, None, None
    try:
        # Bind first: an occupied port must not replace a working PL image.
        server = make_server(None, args.host, args.port)
        processor = VisionPL(args.bitfile, download=args.load_overlay)
        state = PLPreview(processor, args.camera_url, args.fps)
        server.preview_state = state
        server.timeout = .2
        stop = threading.Event()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, lambda *_: stop.set())
        state.thread.start()
        print("PL preview listening on {}:{}".format(args.host, args.port), flush=True)
        while not stop.is_set():
            server.handle_request()
    finally:
        if state is not None:
            state.stop_event.set()
            if state.thread.ident is not None:
                state.thread.join(4)
        if server is not None:
            server.server_close()
        if processor is not None and (state is None or not state.thread.is_alive()):
            processor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

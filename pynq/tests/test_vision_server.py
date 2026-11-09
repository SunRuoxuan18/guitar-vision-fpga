"""Vision service contract tests; network requests stay on localhost."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import http.client
import io
import itertools
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vision_server as vision

try:
    import cv2
    import numpy as np
    CV_AVAILABLE, CV_SKIP_REASON = True, ""
except (ImportError, OSError) as exc:
    CV_AVAILABLE = False
    CV_SKIP_REASON = "OpenCV/numpy unavailable: {}; use the isolated vision venv".format(exc)


JPEG = b"\xff\xd8test-jpeg-bytes\xff\xd9"
RESOLUTION = {"width": 640, "height": 360}


@contextmanager
def http_server(state):
    server = vision.make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        if thread.is_alive():
            raise AssertionError("localhost HTTP server did not stop")


def request(server, path, payload=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2)
    try:
        connection.request("POST" if payload is not None else "GET", path,
                           body=json.dumps(payload) if payload is not None else None,
                           headers={"Content-Type": "application/json"} if payload is not None else {})
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def wait_for_frame(state):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if state.snapshot()["frame_available"]:
            return
        time.sleep(.005)
    raise AssertionError("capture did not publish a frame: {}".format(state.snapshot()))


class SourceAndCliTests(unittest.TestCase):
    def test_import_help_and_source_construction_are_lazy(self):
        code = (
            "import importlib.util, sys, threading; "
            "spec = importlib.util.spec_from_file_location('vision_server', {!r}); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "module.create_source(camera=0); module.create_source(video='unused.avi'); "
            "module.create_source(synthetic=True); "
            "assert not any(x in sys.modules for x in ('cv2', 'numpy', 'audio_fft', 'pynq')); "
            "assert len(threading.enumerate()) == 1; module.parse_args(['--help'])"
        ).format(str(Path(vision.__file__)))
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--camera", result.stdout)

    def test_cli_requires_explicit_exclusive_source(self):
        for argv in ([], ["--camera"], ["--camera", "-1"], ["--camera", "0", "--synthetic"],
                     ["--synthetic", "--video", "file.avi"]):
            with self.subTest(argv=argv), patch.object(vision, "_load_cv2") as loader:
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    vision.parse_args(argv)
                self.assertEqual(error.exception.code, 2)
                loader.assert_not_called()
        args = vision.parse_args(["--synthetic"])
        self.assertIsNone(args.camera)
        self.assertEqual(args.camera_format, "MJPG")
        self.assertEqual(args.jpeg_encoder, "opencv")
        self.assertEqual((args.port, args.fps, args.max_width, args.max_height, args.jpeg_quality),
                         (8082, 30, 1280, 720, 70))

    def test_cli_camera_format_choice_and_noncamera_rejection(self):
        self.assertEqual(vision.parse_args(["--camera", "0", "--camera-format", "YUYV"]).camera_format,
                         "YUYV")
        for argv in (("--camera", "0", "--camera-format", "RGB3"),
                     ("--video", "file.avi", "--camera-format", "YUYV"),
                     ("--synthetic", "--camera-format", "YUYV")):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                vision.parse_args(argv)
            self.assertEqual(error.exception.code, 2)

    def test_configuration_bounds_are_validated(self):
        for option, value in (("--fps", "0"), ("--fps", "31"), ("--fps", "nan"),
                              ("--stale-after", "0"), ("--stale-after", "inf"),
                              ("--max-width", "1281"), ("--max-height", "721"),
                              ("--max-width", "0"), ("--jpeg-quality", "101"),
                              ("--jpeg-quality", "0"), ("--port", "65536")):
            with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    vision.parse_args(["--synthetic", option, value])

    def test_missing_native_encoder_fails_before_listener_or_camera_open(self):
        encoder_module = Mock()
        encoder_module.FastJpegEncoder.return_value._load_library.side_effect = RuntimeError("missing native library")
        with patch.dict(sys.modules, {"fast_jpeg": encoder_module}), \
                patch.object(vision, "make_server") as listener, \
                patch.object(vision, "CaptureWorker") as capture:
            with self.assertRaisesRegex(RuntimeError, "missing native library"):
                vision.main(["--camera", "0", "--camera-format", "YUYV", "--jpeg-encoder", "native-fast"])
        listener.assert_not_called()
        capture.assert_not_called()

    def test_factory_exclusivity_and_provenance(self):
        for kwargs in ({}, {"camera": 0, "video": "a.avi"}, {"camera": 0, "synthetic": True},
                       {"video": "a.avi", "synthetic": True}, {"camera": True}, {"camera": -1},
                       {"synthetic": True, "camera_format": "RGB3"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                vision.create_source(**kwargs)
        with patch.object(vision, "_load_cv2") as loader:
            camera = vision.create_source(camera=2)
            video = vision.create_source(video="a.avi")
            synthetic = vision.create_source(synthetic=True)
        loader.assert_not_called()
        self.assertEqual((camera.kind, camera.target), ("camera", 2))
        self.assertEqual(camera.camera_format, "MJPG")
        self.assertEqual(camera.provenance, {"synthetic": False, "camera_index": 2})
        self.assertEqual(video.kind, "video_file")
        self.assertEqual(Path(video.provenance["path"]), Path("a.avi").resolve())
        self.assertTrue(synthetic.provenance["synthetic"])

    def test_camera_settings_and_release_use_only_mock_capture(self):
        fake_cv = Mock()
        capture = fake_cv.VideoCapture.return_value
        capture.isOpened.return_value = True
        capture.get.return_value = 0
        capture.read.side_effect = [(True, "frame"), (False, None)]
        source = vision.create_source(camera=3)
        with patch.object(vision, "_load_cv2", return_value=fake_cv):
            source.open(1280, 720, 30)
            self.assertEqual(source.read(), "frame")
            self.assertIsNone(source.read())
            source.close()
            source.close()
        if sys.platform.startswith("linux"):
            fake_cv.VideoCapture.assert_called_once_with(3, fake_cv.CAP_V4L2)
        else:
            fake_cv.VideoCapture.assert_called_once_with(3)
        self.assertEqual(capture.set.call_args_list, [
            unittest.mock.call(fake_cv.CAP_PROP_FOURCC, fake_cv.VideoWriter_fourcc.return_value),
            unittest.mock.call(fake_cv.CAP_PROP_FRAME_WIDTH, 1280),
            unittest.mock.call(fake_cv.CAP_PROP_FRAME_HEIGHT, 720),
            unittest.mock.call(fake_cv.CAP_PROP_FPS, 30),
            unittest.mock.call(fake_cv.CAP_PROP_BUFFERSIZE, 4)])
        capture.release.assert_called_once()

    def test_yuyv_requests_format_without_mjpeg_passthrough(self):
        fake_cv = Mock()
        yuyv = sum(ord(char) << (8 * index) for index, char in enumerate("YUYV"))
        fake_cv.VideoWriter_fourcc.side_effect = lambda *chars: sum(
            ord(char) << (8 * index) for index, char in enumerate(chars))
        capture = fake_cv.VideoCapture.return_value
        capture.isOpened.return_value = True
        capture.get.return_value = yuyv
        controls = Mock()
        controls.prefer_fixed_frame_rate.return_value = True
        source = vision.create_source(camera=3, camera_format="YUYV")
        with patch.object(vision, "_load_cv2", return_value=fake_cv), \
                patch.object(vision.sys, "platform", "linux"), \
                patch.dict(sys.modules, {"camera_controls": controls}):
            source.open(1280, 720, 30)
        self.assertEqual(capture.set.call_args_list[:5], [
            unittest.mock.call(fake_cv.CAP_PROP_FOURCC, yuyv),
            unittest.mock.call(fake_cv.CAP_PROP_FRAME_WIDTH, 1280),
            unittest.mock.call(fake_cv.CAP_PROP_FRAME_HEIGHT, 720),
            unittest.mock.call(fake_cv.CAP_PROP_FPS, 30),
            unittest.mock.call(fake_cv.CAP_PROP_BUFFERSIZE, 4)])
        self.assertNotIn(unittest.mock.call(fake_cv.CAP_PROP_CONVERT_RGB, 0), capture.set.call_args_list)
        self.assertFalse(source.raw_mjpeg)
        self.assertEqual(source.provenance["capture_mode"]["fourcc"], "YUYV")
        self.assertEqual(source.provenance["capture_mode"]["requested_fourcc"], "YUYV")

    def test_yuyv_rejects_driver_format_fallback(self):
        fake_cv = Mock()
        fake_cv.VideoWriter_fourcc.side_effect = lambda *chars: sum(
            ord(char) << (8 * index) for index, char in enumerate(chars))
        capture = fake_cv.VideoCapture.return_value
        capture.isOpened.return_value = True
        capture.get.return_value = fake_cv.VideoWriter_fourcc(*"MJPG")
        controls = Mock()
        source = vision.create_source(camera=0, camera_format="YUYV")
        with patch.object(vision, "_load_cv2", return_value=fake_cv), \
                patch.object(vision.sys, "platform", "linux"), \
                patch.dict(sys.modules, {"camera_controls": controls}):
            with self.assertRaisesRegex(RuntimeError, "did not accept YUYV"):
                source.open(640, 480, 30)
            source.close()
        self.assertEqual(source.provenance["capture_mode"]["requested_fourcc"], "YUYV")
        self.assertEqual(source.provenance["capture_mode"]["fourcc"], "MJPG")
        capture.release.assert_called_once()

    def test_noncamera_sources_ignore_camera_format(self):
        for kwargs in ({"video": "a.avi", "camera_format": "YUYV"},
                       {"synthetic": True, "camera_format": "YUYV"}):
            with self.subTest(kwargs=kwargs):
                source = vision.create_source(**kwargs)
                self.assertNotEqual(source.kind, "camera")

    def test_linux_camera_uses_v4l2_instead_of_automatic_gstreamer(self):
        fake_cv = Mock()
        fake_cv.VideoCapture.return_value.get.return_value = 0
        source = vision.create_source(camera=0)
        with patch.object(vision, "_load_cv2", return_value=fake_cv), \
                patch.object(vision.sys, "platform", "linux"):
            source.open(640, 480, 30)
            source.close()
        fake_cv.VideoCapture.assert_called_once_with(0, fake_cv.CAP_V4L2)

    def test_missing_video_never_opens_or_falls_back(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(vision, "_load_cv2") as loader:
            source = vision.create_source(video=Path(directory) / "missing.avi")
            with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
                source.open(1280, 720, 30)
            source.close()
            loader.assert_not_called()

    def test_dependency_import_error_is_actionable(self):
        original = __import__
        def import_without_cv2(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("test missing cv2")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=import_without_cv2):
            with self.assertRaisesRegex(RuntimeError, "opencv-python-headless"):
                vision._load_cv2()

    def test_cli_signal_stops_capture_closes_listener_and_restores_handlers(self):
        server = Mock(server_address=("127.0.0.1", 8082))
        worker = Mock()
        worker.stop.return_value = True
        handlers = {}
        def set_handler(signum, handler):
            handlers[signum] = handler
            return "previous-{}".format(signum)
        server.handle_request.side_effect = lambda: handlers[vision.signal.SIGTERM](None, None)
        with patch.object(vision, "make_server", return_value=server), \
                patch.object(vision, "CaptureWorker", return_value=worker), \
                patch.object(vision.signal, "signal", side_effect=set_handler), \
                patch.object(vision, "_load_cv2") as loader, redirect_stdout(io.StringIO()):
            self.assertEqual(vision.main(["--synthetic"]), 0)
        worker.start.assert_called_once()
        worker.stop.assert_called_once()
        server.server_close.assert_called_once()
        loader.assert_not_called()
        for signum in (vision.signal.SIGINT, vision.signal.SIGTERM):
            self.assertEqual(handlers[signum], "previous-{}".format(signum))


class StateAndHttpTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.state = vision.VisionState(vision.create_source(camera=4), clock=lambda: self.now)

    def publish(self):
        self.state.publish(JPEG, RESOLUTION, {"width": 1920, "height": 1080}, self.now)

    def test_initial_and_fresh_state_metadata(self):
        initial = self.state.snapshot()
        self.assertEqual(initial["status"], "starting")
        self.assertIsNone(initial["captured_ms"])
        self.assertIsNone(initial["resolution"])
        self.assertFalse(initial["frame_available"])
        self.publish()
        self.now += .25
        metadata, jpeg = self.state.frame_snapshot()
        self.assertEqual(jpeg, JPEG)
        self.assertEqual(metadata["sequence"], 1)
        self.assertEqual(metadata["captured_ms"], 10000)
        self.assertEqual(metadata["timestamp_clock"], "process_monotonic")
        self.assertEqual(metadata["age_ms"], 250)
        self.assertEqual(metadata["resolution"], RESOLUTION)
        self.assertEqual(metadata["provenance"]["camera_index"], 4)
        self.assertFalse(metadata["synthetic"])
        metadata["resolution"]["width"] = 1
        metadata["provenance"]["camera_index"] = 99
        self.assertEqual(self.state.snapshot()["resolution"], RESOLUTION)
        self.assertEqual(self.state.snapshot()["provenance"]["camera_index"], 4)
        self.publish()
        self.assertEqual(self.state.snapshot()["sequence"], 2)
        self.assertGreater(self.state.snapshot()["captured_ms"], 10000)

    def test_stale_error_and_stopped_invalidate_bytes(self):
        self.publish()
        self.now += 1.001
        metadata, jpeg = self.state.frame_snapshot()
        self.assertEqual(metadata["status"], "stale")
        self.assertIsNone(jpeg)
        self.publish()
        self.state.fail("capture read failed")
        metadata, jpeg = self.state.frame_snapshot()
        self.assertEqual(metadata["status"], "error")
        self.assertEqual(metadata["error"], "capture read failed")
        self.assertIsNone(jpeg)
        self.publish()
        self.state.stopped()
        self.assertEqual(self.state.snapshot()["status"], "stopped")
        self.assertIsNone(self.state.frame_snapshot()[1])

    def test_recovery_keeps_error_history_and_service_identity(self):
        initial = self.state.snapshot()
        self.assertTrue(initial["stream_id"])
        self.assertEqual(initial["capture_failures"], 0)
        self.assertIsNone(initial["last_error"])
        self.state.fail("USB read failed")
        self.now += .5
        self.publish()
        recovered = self.state.snapshot()
        self.assertEqual(recovered["status"], "streaming")
        self.assertIsNone(recovered["error"])
        self.assertEqual(recovered["capture_failures"], 1)
        self.assertEqual(recovered["last_error"], "USB read failed")
        self.assertEqual(recovered["stream_id"], initial["stream_id"])
        self.assertEqual(recovered["last_error_ms"], 10000)
        other = vision.VisionState(vision.create_source(camera=4))
        self.assertNotEqual(other.snapshot()["stream_id"], initial["stream_id"])

    def test_invalid_frame_drops_preserve_publication_and_expire_normally(self):
        initial = self.state.snapshot()
        self.assertEqual(initial["invalid_frame_drops"], 0)
        self.assertIsNone(initial["last_invalid_frame"])
        self.assertIsNone(initial["last_invalid_frame_ms"])
        self.publish()
        self.now += .25
        self.state.drop_invalid_frame("missing EOI")
        metadata, jpeg = self.state.frame_snapshot()
        self.assertEqual(jpeg, JPEG)
        self.assertEqual(metadata["sequence"], 1)
        self.assertEqual(metadata["captured_ms"], 10000)
        self.assertEqual(metadata["age_ms"], 250)
        self.assertEqual(metadata["status"], "streaming")
        self.assertIsNone(metadata["error"])
        self.assertEqual(metadata["capture_failures"], 0)
        self.assertIsNone(metadata["last_error"])
        self.assertEqual(metadata["invalid_frame_drops"], 1)
        self.assertEqual(metadata["last_invalid_frame"], "missing EOI")
        self.assertEqual(metadata["last_invalid_frame_ms"], 10250)
        self.now += 1
        self.state.drop_invalid_frame("decode failed")
        with http_server(self.state) as server:
            code, _, body = request(server, "/vision/frame.jpg")
        stale = json.loads(body)
        self.assertEqual(code, 503)
        self.assertEqual(stale["status"], "stale")
        self.assertEqual(stale["sequence"], 1)
        self.assertEqual(stale["invalid_frame_drops"], 2)
        self.assertEqual(stale["capture_failures"], 0)
        self.assertIsNone(self.state.frame_snapshot()[1])
        self.publish()
        self.assertEqual(self.state.snapshot()["sequence"], 2)
        self.assertEqual(self.state.snapshot()["invalid_frame_drops"], 2)
        self.assertEqual(self.state.snapshot()["last_invalid_frame"], "decode failed")

    def test_http_status_and_fresh_jpeg_headers_bytes(self):
        self.publish()
        with http_server(self.state) as server:
            code, headers, body = request(server, "/api/vision/status?fresh=1")
            self.assertEqual(code, 200)
            self.assertEqual(headers["Cache-Control"], "no-store")
            status = json.loads(body)
            self.assertEqual(status["source"], "camera")
            self.assertTrue(status["frame_available"])
            code, headers, body = request(server, "/vision/frame.jpg?sequence=1")
        self.assertEqual(code, 200)
        self.assertEqual(body, JPEG)
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        self.assertEqual(int(headers["Content-Length"]), len(JPEG))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Vision-Sequence"], "1")
        self.assertEqual(float(headers["X-Vision-Captured-Ms"]), 10000)
        self.assertEqual(headers["X-Vision-Source"], "camera")
        self.assertEqual(headers["X-Vision-Stream"], self.state.stream_id)

    def test_http_mjpeg_stream_emits_numbered_jpeg_parts(self):
        self.publish()
        with http_server(self.state) as server:
            connection = socket.create_connection(server.server_address, timeout=2)
            try:
                connection.sendall(
                    b"GET /vision/stream.mjpeg HTTP/1.1\r\n"
                    b"Host: 127.0.0.1\r\nConnection: close\r\n\r\n")
                received = b""
                while JPEG not in received:
                    chunk = connection.recv(8192)
                    if not chunk:
                        break
                    received += chunk
            finally:
                connection.close()
        self.assertIn(b"HTTP/1.1 200 OK", received)
        self.assertIn(b"Content-Type: multipart/x-mixed-replace; boundary=frame", received)
        self.assertIn(b"X-Vision-Sequence: 1", received)
        self.assertIn("Content-Length: {}".format(len(JPEG)).encode("ascii"), received)
        self.assertIn(JPEG, received)

    def test_http_unavailable_stale_and_error_return_503(self):
        with http_server(self.state) as server:
            for expected, update in (("starting", lambda: None), ("stale", self.make_stale),
                                     ("error", lambda: self.state.fail("no camera connected"))):
                update()
                code, headers, body = request(server, "/vision/frame.jpg")
                self.assertEqual(code, 503)
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertEqual(json.loads(body)["status"], expected)
                self.assertNotIn("X-Vision-Sequence", headers)
            code, _, body = request(server, "/api/vision/status")
            self.assertEqual(code, 200)
            self.assertEqual(json.loads(body)["error"], "no camera connected")

    def make_stale(self):
        self.publish()
        self.now += 2

    def test_unknown_endpoint_is_json_404(self):
        with http_server(self.state) as server:
            code, headers, body = request(server, "/api/audio")
        self.assertEqual(code, 404)
        self.assertEqual(json.loads(body)["error"], "unknown_endpoint")
        self.assertEqual(headers["Cache-Control"], "no-store")

    def test_video_and_synthetic_headers_have_exact_source_labels(self):
        for source in (vision.create_source(video="unused.avi"), vision.create_source(synthetic=True)):
            state = vision.VisionState(source, clock=lambda: self.now)
            state.publish(JPEG, RESOLUTION, RESOLUTION, self.now)
            with http_server(state) as server:
                code, headers, _ = request(server, "/vision/frame.jpg")
                _, _, body = request(server, "/api/vision/status")
            self.assertEqual(code, 200)
            self.assertEqual(headers["X-Vision-Source"], source.kind)
            self.assertEqual(json.loads(body)["synthetic"], source.kind == "synthetic")


class CaptureThreadTests(unittest.TestCase):
    def source(self):
        source = Mock(kind="video_file", provenance={"path": "test-only", "synthetic": False})
        source.read.side_effect = ["frame1", "frame2", None]
        return source

    def test_dedicated_thread_sequences_closes_and_reports_eof_without_fallback(self):
        source = self.source()
        thread_ids = []
        source.open.side_effect = lambda *_: thread_ids.append(threading.get_ident())
        source.read.side_effect = ["frame1", "frame2", None]
        encoder = Mock(return_value=(JPEG, RESOLUTION, RESOLUTION))
        state = vision.VisionState(source)
        worker = vision.CaptureWorker(state, encoder)
        worker.start()
        worker.thread.join(2)
        self.assertFalse(worker.thread.is_alive())
        self.assertNotEqual(thread_ids[0], threading.get_ident())
        source.open.assert_called_once_with(1280, 720, 30)
        source.close.assert_called_once()
        self.assertEqual(encoder.call_count, 2)
        self.assertEqual(state.snapshot()["sequence"], 2)
        self.assertGreater(state.snapshot()["captured_ms"], 0)
        self.assertEqual(state.snapshot()["status"], "error")
        self.assertIn("end of file", state.snapshot()["error"])
        self.assertIsNone(state.frame_snapshot()[1])

    def test_open_read_encode_and_cleanup_errors_are_visible(self):
        for stage in ("open", "read", "encode", "close"):
            with self.subTest(stage=stage):
                source = self.source()
                encoder = Mock(return_value=(JPEG, RESOLUTION, RESOLUTION))
                if stage == "encode":
                    encoder.side_effect = RuntimeError("encoding failed")
                else:
                    getattr(source, stage).side_effect = RuntimeError(stage + " failed")
                state = vision.VisionState(source)
                worker = vision.CaptureWorker(state, encoder)
                worker.start()
                worker.thread.join(2)
                self.assertFalse(worker.thread.is_alive())
                self.assertEqual(state.snapshot()["status"], "error")
                self.assertIn("failed", state.snapshot()["error"])
                self.assertFalse(state.snapshot()["frame_available"])
                source.close.assert_called_once()

    def test_stop_releases_source_and_stops_publishing(self):
        source = self.source()
        source.read.side_effect = None
        source.read.return_value = "frame"
        state = vision.VisionState(source)
        worker = vision.CaptureWorker(state, lambda *_: (JPEG, RESOLUTION, RESOLUTION))
        worker.start()
        try:
            wait_for_frame(state)
        finally:
            self.assertTrue(worker.stop())
        sequence = state.snapshot()["sequence"]
        self.assertEqual(state.snapshot()["status"], "stopped")
        self.assertIsNone(state.frame_snapshot()[1])
        self.assertEqual(state.snapshot()["sequence"], sequence)
        source.close.assert_called_once()

    def test_camera_recovers_after_open_or_read_failure_without_fake_frames(self):
        for stage in ("open", "read"):
            with self.subTest(stage=stage):
                source = Mock(kind="camera", provenance={"synthetic": False, "camera_index": 0})
                source.open.side_effect = [RuntimeError("not ready"), None] if stage == "open" else None
                source.read.side_effect = itertools.chain([None], itertools.repeat("real frame")) if stage == "read" else None
                source.read.return_value = "real frame"
                encoder = Mock(return_value=(JPEG, RESOLUTION, RESOLUTION))
                state = vision.VisionState(source)
                worker = vision.CaptureWorker(state, encoder)
                worker.start()
                try:
                    deadline = time.monotonic() + 1
                    while state.snapshot()["status"] != "error" and time.monotonic() < deadline:
                        time.sleep(.005)
                    self.assertEqual(state.snapshot()["status"], "error")
                    self.assertIsNone(state.frame_snapshot()[1])
                    encoder.assert_not_called()
                    wait_for_frame(state)
                    self.assertEqual(state.snapshot()["source"], "camera")
                    self.assertFalse(state.snapshot()["synthetic"])
                    self.assertEqual(source.open.call_count, 2)
                finally:
                    self.assertTrue(worker.stop())
                self.assertEqual(source.close.call_count, 2)

    def test_camera_retry_wait_is_interruptible(self):
        source = Mock(kind="camera", provenance={"synthetic": False})
        source.open.side_effect = RuntimeError("not ready")
        state = vision.VisionState(source)
        worker = vision.CaptureWorker(state)
        worker.start()
        deadline = time.monotonic() + 1
        while state.snapshot()["status"] != "error" and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(state.snapshot()["status"], "error")
        self.assertTrue(worker.stop(timeout=.5))
        source.open.assert_called_once()
        source.close.assert_called_once()
        self.assertEqual(state.snapshot()["status"], "stopped")

    def test_camera_does_not_reopen_after_cleanup_failed(self):
        for stage in ("open", "read"):
            with self.subTest(stage=stage):
                source = Mock(kind="camera", provenance={"synthetic": False})
                if stage == "open":
                    source.open.side_effect = RuntimeError("not ready")
                else:
                    source.read.return_value = None
                source.close.side_effect = RuntimeError("device not released")
                encoder = Mock(return_value=(JPEG, RESOLUTION, RESOLUTION))
                state = vision.VisionState(source)
                worker = vision.CaptureWorker(state, encoder)
                worker.start()
                try:
                    worker.thread.join(1.2)
                    self.assertFalse(worker.thread.is_alive(), "Failed cleanup must stop camera retries")
                    source.open.assert_called_once()
                    source.close.assert_called_once()
                    encoder.assert_not_called()
                    self.assertEqual(state.snapshot()["status"], "error")
                    self.assertIn("Source cleanup failed", state.snapshot()["error"])
                    self.assertIsNone(state.frame_snapshot()[1])
                finally:
                    worker.stop()


class CameraWatchdogTests(unittest.TestCase):
    def test_repeated_unhealthy_status_restarts_the_service_process(self):
        state = Mock()
        state.snapshot.side_effect = [{"status": "stale"}, {"status": "stale"}]
        state.clock.side_effect = [10.0, 10.06]
        exit_process = Mock()
        watchdog = vision.CameraWatchdog(
            state, restart_after=.05, interval=.01, exit_process=exit_process)
        watchdog.start()
        watchdog.join(1)
        self.assertFalse(watchdog.is_alive())
        exit_process.assert_called_once_with(1)

    def test_a_fresh_status_resets_the_unhealthy_timer(self):
        state = Mock()
        state.snapshot.side_effect = [
            {"status": "error"}, {"status": "streaming"},
            {"status": "stale"}, {"status": "stale"}
        ]
        state.clock.side_effect = [10.0, 10.2, 20.0, 20.06]
        exit_process = Mock()
        watchdog = vision.CameraWatchdog(
            state, restart_after=.05, interval=.01, exit_process=exit_process)
        watchdog.start()
        watchdog.join(1)
        self.assertFalse(watchdog.is_alive())
        exit_process.assert_called_once_with(1)


@unittest.skipUnless(CV_AVAILABLE, CV_SKIP_REASON)
class TrackingServiceTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.state = vision.VisionState(vision.create_source(camera=0), clock=lambda: self.now)
        self.frame = np.random.default_rng(12).integers(0, 256, (240, 320), dtype=np.uint8)
        self.corners = [{"x": x, "y": y} for x, y in ((.2, .3), (.8, .3), (.8, .7), (.2, .7))]
        self.publish(self.frame)

    def publish(self, gray):
        self.state.publish(JPEG, RESOLUTION, RESOLUTION, self.now, gray)

    def test_tracker_added_after_gray_cadence_decision_receives_current_frame(self):
        tracker = Mock()
        self.state._trackers["new"] = {"tracker": tracker, "sequence": 1, "used_at": self.now}
        large = cv2.resize(self.frame, (640, 480))
        ok, encoded = cv2.imencode(".jpg", large)
        self.assertTrue(ok)
        self.state.publish(encoded.tobytes(), {"width": 640, "height": 480},
                           {"width": 640, "height": 480}, self.now, gray=None)
        tracker.update.assert_called_once()
        self.assertEqual(tracker.update.call_args.args[0].shape, self.frame.shape)
        self.assertEqual(self.state._trackers["new"]["sequence"], 2)

    def payload(self, **updates):
        return dict(corners=self.corners, sequence=1, stream_id=self.state.stream_id, **updates)

    def test_calibration_on_frame_without_gray_history_decodes_matching_jpeg(self):
        ok, encoded = cv2.imencode(".jpg", self.frame)
        self.assertTrue(ok)
        self.state.publish(encoded.tobytes(), {"width": 320, "height": 240},
                           {"width": 320, "height": 240}, self.now)
        self.assertEqual(self.state._gray_history[-1][0], 1)
        tracked = self.state.start_tracking(dict(corners=self.corners, sequence=2,
                                                stream_id=self.state.stream_id))
        self.assertEqual(tracked["status"], "tracking")
        self.assertEqual(tracked["sequence"], 2)

    def test_catchup_per_frame_headers_and_independent_sessions(self):
        moved = cv2.warpAffine(self.frame, np.float32([[1, 0, 4], [0, 1, 2]]), (320, 240))
        self.publish(moved)
        with http_server(self.state) as server:
            code, _, body = request(server, "/api/vision/tracking", self.payload())
            self.assertEqual(code, 200)
            first = json.loads(body)
            self.assertEqual(first["sequence"], 2)
            self.assertEqual(first["status"], "tracking")
            self.assertAlmostEqual(first["corners"][0]["x"], .2 + 4 / 319, places=2)
            _, _, body = request(server, "/api/vision/tracking", self.payload())
            second = json.loads(body)
            self.assertNotEqual(first["tracker_id"], second["tracker_id"])
            code, headers, _ = request(server, "/vision/frame.jpg?tracking=" + first["tracker_id"])
            snapshot = json.loads(headers["X-Vision-Tracking"])
            self.assertEqual(code, 200)
            self.assertEqual(snapshot["sequence"], int(headers["X-Vision-Sequence"]))
            self.assertEqual(snapshot["stream_id"], headers["X-Vision-Stream"])
            request(server, "/api/vision/tracking/stop", {"tracker_id": first["tracker_id"]})
            _, headers, _ = request(server, "/vision/frame.jpg?tracking=" + first["tracker_id"])
            self.assertEqual(json.loads(headers["X-Vision-Tracking"])["status"], "lost")
            self.assertEqual(self.state.frame_snapshot(second["tracker_id"])[0]["tracking"]["status"], "tracking")

    def test_invalid_expired_stale_and_texture_rejection(self):
        with http_server(self.state) as server:
            for changes in ({"sequence": True}, {"stream_id": "wrong"}, {"sequence": -1},
                            {"corners": [{"x": 2, "y": .5}] * 4},
                            {"corners": [{"x": .5, "y": .5}] * 4}):
                payload = self.payload()
                payload.update(changes)
                code, _, _ = request(server, "/api/vision/tracking", payload)
                self.assertEqual(code, 400, changes)
            self.now += 2
            self.assertEqual(request(server, "/api/vision/tracking", self.payload())[0], 400)
            for _ in range(11):
                self.publish(self.frame)
            self.assertEqual(request(server, "/api/vision/tracking", self.payload())[0], 400)
            self.publish(np.zeros_like(self.frame))
            payload = self.payload()
            payload["sequence"] = self.state.snapshot()["sequence"]
            code, _, body = request(server, "/api/vision/tracking", payload)
            self.assertEqual(code, 400)
            self.assertIn("insufficient_texture", json.loads(body)["error"])

    def test_capture_failure_and_stop_cannot_reuse_old_tracking(self):
        for invalidate in (lambda: self.state.fail("usb"), lambda: self.state.stopped()):
            with self.subTest(invalidate=invalidate):
                self.state = vision.VisionState(vision.create_source(camera=0), clock=lambda: self.now)
                self.publish(self.frame)
                started = self.state.start_tracking(self.payload())
                invalidate()
                self.publish(self.frame)
                snapshot = self.state.frame_snapshot(started["tracker_id"])[0]["tracking"]
                self.assertEqual(snapshot["status"], "lost")
                self.assertIsNone(snapshot["corners"])

    def test_session_limit_and_expiry(self):
        for _ in range(4):
            self.state.start_tracking(self.payload())
        with self.assertRaisesRegex(ValueError, "Too many"):
            self.state.start_tracking(self.payload())
        self.now += 61
        self.publish(self.frame)
        payload = self.payload()
        payload["sequence"] = 2
        self.assertEqual(self.state.start_tracking(payload)["status"], "tracking")


@unittest.skipUnless(CV_AVAILABLE, CV_SKIP_REASON)
class OpenCvTests(unittest.TestCase):
    def test_resized_raw_mjpeg_initial_gray_matches_later_tracking_geometry(self):
        ok, encoded = cv2.imencode(".jpg", np.full((900, 1600, 3), (20, 80, 200), np.uint8))
        self.assertTrue(ok)
        gray_shapes = []
        original_publish = vision.VisionState.publish

        def record_gray(state, *args):
            gray_shapes.append(None if args[-1] is None else args[-1].shape)
            return original_publish(state, *args)

        with patch.object(vision.VisionState, "publish", new=record_gray):
            state, snapshots = self.capture_raw_frames([(10.0, encoded.tobytes())],
                encoder=vision.encode_frame, max_width=1280, max_height=720)
        self.assertEqual(gray_shapes, [(360, 640)])
        self.assertEqual(snapshots[0][0]["resolution"], {"width": 1280, "height": 720})

    def test_yuyv_gray_history_is_throttled_but_manual_tracking_keeps_each_frame(self):
        for tracking in (False, True):
            with self.subTest(tracking=tracking):
                source = vision.create_source(camera=0, camera_format="YUYV")
                now = [10.0]
                state = vision.VisionState(source, clock=lambda: now[0])
                tracker = Mock()
                if tracking:
                    state._trackers["manual"] = {"tracker": tracker, "sequence": 0, "used_at": now[0]}
                worker = vision.CaptureWorker(state, lambda *_: (JPEG, RESOLUTION, RESOLUTION))
                remaining = iter((10.0, 10.04, 10.08, 10.12))
                observations = []

                def read():
                    now[0] = next(remaining)
                    return np.zeros((48, 64, 3), np.uint8)

                original_publish = state.publish
                def publish(*args):
                    original_publish(*args)
                    observations.append(args[-1] is not None)
                    if len(observations) == 4:
                        worker._stop.set()

                with patch.object(source, "open"), patch.object(source, "close"), \
                        patch.object(source, "read", side_effect=read), \
                        patch.object(state, "publish", side_effect=publish):
                    worker.start()
                    worker.thread.join(2)
                    self.assertFalse(worker.thread.is_alive())
                self.assertEqual(observations, [True] * 4 if tracking else [True, False, False, True])
                if tracking:
                    self.assertEqual(tracker.update.call_count, 4)

    def capture_raw_frames(self, frames, encoder=None, **state_options):
        source = vision.create_source(camera=0)
        source.raw_mjpeg = True
        now = [10.0]
        state = vision.VisionState(source, clock=lambda: now[0], **state_options)
        encoder = encoder if encoder is not None else Mock(
            side_effect=AssertionError("JPEG must not be re-encoded"))
        worker = vision.CaptureWorker(state, encoder)
        remaining = iter(frames)
        snapshots, lifecycle = [], []

        def read():
            now[0], jpeg = next(remaining)
            return np.frombuffer(jpeg, np.uint8).reshape(1, -1)

        def record(operation):
            def apply(*args, **kwargs):
                result = operation(*args, **kwargs)
                snapshots.append(state.frame_snapshot())
                lifecycle.append((opened.call_count, closed.call_count))
                if len(snapshots) == len(frames):
                    worker._stop.set()
                return result
            return apply

        with patch.object(source, "open") as opened, patch.object(source, "close") as closed, \
                patch.object(source, "read", side_effect=read), \
                patch.object(state, "publish", side_effect=record(state.publish)), \
                patch.object(state, "drop_invalid_frame", side_effect=record(state.drop_invalid_frame)), \
                patch.object(state, "fail", wraps=state.fail) as failed:
            worker.start()
            try:
                worker.thread.join(2)
                self.assertFalse(worker.thread.is_alive(), "Raw capture did not finish its frame sequence")
            finally:
                self.assertTrue(worker.stop())
            opened.assert_called_once()
            closed.assert_called_once()
            failed.assert_not_called()
        self.assertEqual(lifecycle, [(1, 0)] * len(frames), "Invalid frames must not close or reopen capture")
        self.assertEqual(state.snapshot()["status"], "stopped")
        return state, snapshots

    def test_raw_camera_rejects_missing_markers_between_gray_decodes_and_recovers(self):
        ok, encoded = cv2.imencode(".jpg", np.full((48, 64, 3), (40, 80, 160), np.uint8))
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        actual_decode = cv2.imdecode
        with patch.object(cv2, "imdecode", wraps=actual_decode) as decode:
            state, snapshots = self.capture_raw_frames([
                (10.0, jpeg), (10.02, jpeg[:-2]), (10.04, jpeg[2:]), (10.06, jpeg),
            ])
        self.assertEqual(decode.call_count, 1, "The truncated frame arrives inside the 100 ms gray cadence")
        self.assertEqual([metadata["sequence"] for metadata, _ in snapshots], [1, 1, 1, 2])
        self.assertEqual([metadata["invalid_frame_drops"] for metadata, _ in snapshots], [0, 1, 2, 2])
        self.assertIn("EOI", snapshots[1][0]["last_invalid_frame"])
        self.assertIn("SOI", snapshots[2][0]["last_invalid_frame"])
        self.assertEqual([body for _, body in snapshots], [jpeg] * 4)
        self.assertEqual(snapshots[1][0]["captured_ms"], 10000)
        self.assertEqual(snapshots[-1][0]["captured_ms"], 10060)
        self.assertEqual(state.snapshot()["capture_failures"], 0)

    def test_raw_camera_all_bad_after_valid_becomes_stale_without_reopening(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        _, snapshots = self.capture_raw_frames([
            (10.0, jpeg), (10.05, jpeg[:-2]), (10.2, jpeg[:-2]),
        ], stale_after=.1)
        metadata, body = snapshots[-1]
        self.assertEqual(metadata["status"], "stale")
        self.assertTrue(metadata["stale"])
        self.assertFalse(metadata["frame_available"])
        self.assertIsNone(body)
        self.assertEqual(metadata["sequence"], 1)
        self.assertEqual(metadata["captured_ms"], 10000)
        self.assertAlmostEqual(metadata["age_ms"], 200)
        self.assertEqual(metadata["invalid_frame_drops"], 2)
        self.assertEqual(metadata["capture_failures"], 0)
        self.assertIsNone(metadata["error"])

    def test_raw_camera_decode_none_drops_and_next_valid_frame_recovers(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        malformed = jpeg[:2] + jpeg[-2:]
        self.assertIsNone(cv2.imdecode(np.frombuffer(malformed, np.uint8), cv2.IMREAD_GRAYSCALE))
        _, snapshots = self.capture_raw_frames([
            (10.0, jpeg), (10.2, malformed), (10.4, jpeg),
        ])
        self.assertEqual([metadata["sequence"] for metadata, _ in snapshots], [1, 1, 2])
        self.assertEqual(snapshots[-1][0]["invalid_frame_drops"], 1)
        self.assertIn("grayscale decode", snapshots[1][0]["last_invalid_frame"])
        self.assertEqual(snapshots[1][0]["captured_ms"], 10000)
        self.assertEqual(snapshots[-1][0]["captured_ms"], 10400)
        self.assertEqual([body for _, body in snapshots], [jpeg] * 3)

    def test_raw_camera_oversize_color_decode_none_is_dropped(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        actual_decode = cv2.imdecode
        color_calls = []

        def decode(frame, mode):
            if mode == cv2.IMREAD_COLOR:
                color_calls.append(mode)
                if len(color_calls) == 2:
                    return None
            return actual_decode(frame, mode)

        encoder = Mock(wraps=vision.encode_frame)
        with patch.object(cv2, "imdecode", side_effect=decode):
            _, snapshots = self.capture_raw_frames([
                (10.0, jpeg), (10.02, jpeg), (10.04, jpeg),
            ], encoder=encoder, max_width=32, max_height=24)
        self.assertEqual(len(color_calls), 3)
        self.assertEqual(encoder.call_count, 2)
        self.assertEqual([metadata["sequence"] for metadata, _ in snapshots], [1, 1, 2])
        self.assertEqual(snapshots[-1][0]["invalid_frame_drops"], 1)
        self.assertIn("color decode", snapshots[1][0]["last_invalid_frame"])
        self.assertEqual(snapshots[1][0]["captured_ms"], 10000)
        self.assertEqual(snapshots[-1][0]["resolution"], {"width": 32, "height": 24})
        self.assertEqual(snapshots[-1][0]["source_resolution"], {"width": 64, "height": 48})

    def test_raw_camera_trims_padding_after_last_eoi(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        _, snapshots = self.capture_raw_frames([(10.0, jpeg + b"\x00" * 64)])
        metadata, body = snapshots[0]
        self.assertEqual(body, jpeg)
        self.assertEqual(metadata["invalid_frame_drops"], 0)
        self.assertEqual(metadata["resolution"], {"width": 64, "height": 48})

    def test_camera_jpeg_passthrough_preserves_bytes_and_skips_reencoding(self):
        frame = np.random.default_rng(8).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        raw = encoded.reshape(1, -1)
        source = vision.create_source(camera=0)
        source.raw_mjpeg = True
        state = vision.VisionState(source)
        encoder = Mock(side_effect=AssertionError("JPEG must not be re-encoded"))
        worker = vision.CaptureWorker(state, encoder)
        with patch.object(source, "open"), patch.object(source, "close"), \
                patch.object(source, "read", return_value=raw):
            worker.start()
            try:
                wait_for_frame(state)
                metadata, jpeg = state.frame_snapshot()
                self.assertEqual(jpeg, encoded.tobytes())
                self.assertEqual(metadata["resolution"], {"width": 64, "height": 48})
                self.assertEqual(state._gray_history[-1][1].shape, (48, 64))
                encoder.assert_not_called()
            finally:
                self.assertTrue(worker.stop())

    def test_encoder_bounds_aspect_ratio_quality_and_real_jpeg(self):
        for shape, expected in (((1080, 1920, 3), (720, 1280, 3)),
                                ((1920, 1080, 3), (720, 405, 3)),
                                ((180, 320, 3), (180, 320, 3))):
            with self.subTest(shape=shape):
                frame = np.full(shape, (40, 100, 200), dtype=np.uint8)
                original = frame.copy()
                actual_imencode = cv2.imencode
                with patch.object(cv2, "imencode", wraps=actual_imencode) as encode:
                    jpeg, resolution, source_resolution = vision.encode_frame(frame)
                self.assertEqual(encode.call_args[0][2], [cv2.IMWRITE_JPEG_QUALITY, 70])
                decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, expected)
                self.assertEqual(resolution, {"width": expected[1], "height": expected[0]})
                self.assertEqual(source_resolution, {"width": shape[1], "height": shape[0]})
                np.testing.assert_array_equal(frame, original)

    def test_bad_frames_and_failed_jpeg_encoding(self):
        for frame in (None, np.zeros((3, 3), np.uint8), np.zeros((3, 3, 4), np.uint8),
                      np.zeros((0, 3, 3), np.uint8), np.zeros((3, 3, 3), np.float32)):
            with self.assertRaisesRegex(ValueError, "BGR frame"):
                vision.encode_frame(frame)
        with patch.object(cv2, "imencode", return_value=(False, None)):
            with self.assertRaisesRegex(RuntimeError, "encode a JPEG"):
                vision.encode_frame(np.zeros((3, 3, 3), np.uint8))

    def test_synthetic_capture_serves_real_labeled_jpeg_without_camera(self):
        with patch.object(cv2, "VideoCapture", side_effect=AssertionError("physical capture forbidden")):
            source = vision.create_source(synthetic=True)
            source.open(1280, 720, 30)
            frame = source.read()
            self.assertGreater(np.count_nonzero(frame[300:345] != 24), 0)
            self.assertFalse(np.array_equal(frame, source.read()))
            state = vision.VisionState(source)
            worker = vision.CaptureWorker(state)
            worker.start()
            try:
                wait_for_frame(state)
                with http_server(state) as server:
                    code, headers, body = request(server, "/vision/frame.jpg")
                    _, _, status_body = request(server, "/api/vision/status")
                self.assertEqual(code, 200)
                self.assertEqual(headers["X-Vision-Source"], "synthetic")
                decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, (360, 640, 3))
                status = json.loads(status_body)
                self.assertTrue(status["synthetic"])
                self.assertTrue(status["provenance"]["synthetic"])
                self.assertEqual(status["resolution"], RESOLUTION)
            finally:
                self.assertTrue(worker.stop())

    def test_real_video_file_is_read_and_closed_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
            if not writer.isOpened():
                writer.release()
                self.skipTest("OpenCV lacks the MJPG encoder for an offline video fixture")
            try:
                writer.write(np.full((48, 64, 3), (40, 80, 160), np.uint8))
            finally:
                writer.release()
            actual_capture = cv2.VideoCapture
            def file_only(target):
                self.assertEqual(target, str(path))
                return actual_capture(target)
            source = vision.create_source(video=path)
            state = vision.VisionState(source)
            worker = vision.CaptureWorker(state)
            with patch.object(cv2, "VideoCapture", side_effect=file_only):
                worker.start()
                worker.thread.join(2)
            self.assertFalse(worker.thread.is_alive())
            self.assertEqual(state.snapshot()["sequence"], 1)
            self.assertEqual(state.snapshot()["source"], "video_file")
            self.assertEqual(state.snapshot()["resolution"], {"width": 64, "height": 48})
            self.assertEqual(state.snapshot()["status"], "error")
            self.assertIsNone(source.capture)


if __name__ == "__main__":
    unittest.main()

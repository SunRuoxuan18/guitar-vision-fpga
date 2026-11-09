"""Matched-frame automatic geometry, cache expiry and capture recovery."""

import json
from pathlib import Path
import sys
import types
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_server import VisionState, create_source
from test_vision_server import http_server, request


class FakeDetector:
    instances = []

    def __init__(self):
        self.calls = []
        self.instances.append(self)

    def update(self, gray):
        self.calls.append(gray)
        return {"mode": "auto", "status": "searching", "reason": "no_lattice",
                "corners": None, "strings": [], "frets": []}

    def invalidate(self, reason):
        return {"mode": "auto", "status": "searching", "reason": reason,
                "corners": None, "strings": [], "frets": []}


class AutoVisionApiTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        FakeDetector.instances = []
        self.patch = patch.dict(sys.modules, {"auto_fretboard": types.SimpleNamespace(
            AutoFretboardTracker=FakeDetector)})
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.state = VisionState(create_source(camera=0), clock=lambda: self.now)
        self.resolution = {"width": 640, "height": 480}

    def publish(self, gray, jpeg=b"first"):
        self.state.publish(jpeg, self.resolution, self.resolution, self.now, gray)

    def analyze(self):
        metadata, jpeg = self.state.auto_frame_snapshot()
        self.state._auto_thread.join(2)
        self.assertFalse(self.state._auto_thread.is_alive())
        return self.state.auto_frame_snapshot()

    def test_cached_geometry_never_delays_or_replaces_newest_preview(self):
        self.publish("gray1")
        first, jpeg = self.analyze()
        self.assertEqual(jpeg, b"first")
        self.now += .1
        self.publish("gray2", b"second")
        cached, jpeg = self.state.auto_frame_snapshot()
        self.assertEqual(jpeg, b"second")
        self.assertEqual(cached["sequence"], first["sequence"] + 1)
        self.assertEqual(cached["fretboard"]["reason"], "analysis_pending")
        self.assertIsNone(cached["fretboard"]["corners"])
        self.assertEqual(FakeDetector.instances[0].calls, ["gray1"])
        self.now += .11
        fresh, jpeg = self.analyze()
        self.assertEqual(jpeg, b"second")
        self.assertEqual(fresh["fretboard"]["sequence"], 2)
        self.assertEqual(FakeDetector.instances[0].calls, ["gray1", "gray2"])

    def test_roi_request_uses_displayed_jpeg_and_invalidates_old_analysis(self):
        self.publish("gray1", b"displayed")
        self.analyze()
        self.now += .02
        self.publish("gray2", b"newer")
        previous = self.state._capture_generation
        corners = [{"x": x, "y": y} for x, y in ((.1, .2), (.9, .2), (.9, .8), (.1, .8))]
        reply = self.state.set_auto_roi(dict(sequence=1, stream_id=self.state.stream_id, corners=corners))
        self.assertEqual(reply["status"], "pending")
        self.assertEqual(self.state._auto_roi["jpeg"], b"displayed")
        self.assertIsNone(self.state._auto_cached)
        self.assertGreater(self.state._capture_generation, previous)
        corners[0]["x"] = .5
        self.assertEqual(self.state._auto_roi["corners"][0]["x"], .1)
        self.state.fail("camera disconnected")
        self.assertIsNone(self.state._auto_roi)
        self.assertFalse(self.state._auto_roi_active)

    def test_roi_endpoint_rejects_invalid_stale_and_expired_selections(self):
        self.publish("gray1")
        corners = [{"x": x, "y": y} for x, y in ((.1, .2), (.9, .2), (.9, .8), (.1, .8))]
        payload = dict(sequence=1, stream_id=self.state.stream_id, corners=corners)
        with http_server(self.state) as server:
            for changes in ({"sequence": True}, {"sequence": -1}, {"stream_id": "wrong"},
                            {"corners": [{"x": 2, "y": .5}] * 4}, {"corners": []}):
                code, _, _ = request(server, "/api/vision/roi", dict(payload, **changes))
                self.assertEqual(code, 400, changes)
            self.now += 1.1
            self.assertEqual(request(server, "/api/vision/roi", payload)[0], 400)

    def test_roi_configuration_cannot_publish_superseded_background_results(self):
        entered, release = threading.Event(), threading.Event()
        class Blocked(FakeDetector):
            def update(self, gray):
                entered.set()
                release.wait(2)
                return super().update(gray)
        self.publish("gray1")
        with patch.dict(sys.modules, {"auto_fretboard": types.SimpleNamespace(AutoFretboardTracker=Blocked)}):
            try:
                self.state.auto_frame_snapshot()
                self.assertTrue(entered.wait(1))
                self.state.set_auto_roi(dict(sequence=1, stream_id=self.state.stream_id,
                    corners=[{"x": x, "y": y} for x, y in ((.1, .2), (.9, .2), (.9, .8), (.1, .8))]))
            finally:
                release.set()
                self.state._auto_thread.join(2)
        self.assertIsNone(self.state._auto_cached)

    def test_capture_failure_then_recovery_discards_cached_detector(self):
        self.publish("gray1")
        self.analyze()
        self.state.fail("USB disconnected")
        self.assertIsNone(self.state.auto_frame_snapshot()[1])
        self.now += .01
        self.publish("gray2", b"recovered")
        metadata, jpeg = self.analyze()
        self.assertEqual(jpeg, b"recovered")
        self.assertEqual(len(FakeDetector.instances), 2)
        self.assertEqual(metadata["fretboard"]["sequence"], 2)

    def test_completed_result_is_only_served_with_its_own_bounded_age_jpeg(self):
        self.publish("gray1")
        self.analyze()
        self.now += .03
        self.publish("gray2", b"second")
        metadata, jpeg = self.state.auto_frame_snapshot()
        self.assertEqual(jpeg, b"first")
        self.assertEqual(metadata["fretboard"]["sequence"], metadata["sequence"])
        self.now += .06
        metadata, jpeg = self.state.auto_frame_snapshot()
        self.assertEqual(jpeg, b"second")
        self.assertIsNone(metadata["fretboard"]["corners"])

    def test_slow_inference_does_not_stop_capture_or_publish_old_geometry(self):
        state = self.state
        outer = self
        class Slow(FakeDetector):
            def update(self, gray):
                # Publishing here would deadlock if inference held capture lock.
                outer.now += 1.1
                outer.publish("gray2", b"new")
                return super().update(gray)
        self.publish("gray1")
        with patch.dict(sys.modules, {"auto_fretboard": types.SimpleNamespace(AutoFretboardTracker=Slow)}):
            metadata, jpeg = self.analyze()
        self.assertEqual(jpeg, b"new")
        self.assertFalse(metadata["stale"])
        self.assertEqual(metadata["fretboard"]["reason"], "detector_too_slow")
        self.assertEqual(metadata["fretboard"]["sequence"], metadata["sequence"])
        self.assertIsNone(metadata["fretboard"]["corners"])

    def test_blocked_detection_does_not_block_http_or_queue_work(self):
        entered, release = threading.Event(), threading.Event()
        class Blocked(FakeDetector):
            def update(self, gray):
                entered.set()
                release.wait(2)
                return super().update(gray)
        self.publish("gray1")
        with patch.dict(sys.modules, {"auto_fretboard": types.SimpleNamespace(AutoFretboardTracker=Blocked)}):
            try:
                self.state.auto_frame_snapshot()
                self.assertTrue(entered.wait(1))
                with http_server(self.state) as server:
                    for index in range(8):
                        self.now += .04
                        self.publish("new", str(index).encode())
                        started = time.monotonic()
                        code, headers, jpeg = request(server, "/vision/frame.jpg?auto=1")
                        self.assertLess(time.monotonic() - started, .25)
                        self.assertEqual(code, 200)
                        self.assertEqual(jpeg, str(index).encode())
                        geometry = json.loads(headers["X-Vision-Fretboard"])
                        self.assertEqual(geometry["sequence"], int(headers["X-Vision-Sequence"]))
                        self.assertIsNone(geometry["corners"])
                self.assertEqual(len(FakeDetector.instances), 1)
            finally:
                release.set()
                self.state._auto_thread.join(2)

    def test_http_geometry_header_matches_jpeg_and_plain_preview_stays_lazy(self):
        self.publish("gray1")
        with http_server(self.state) as server:
            code, headers, jpeg = request(server, "/vision/frame.jpg")
            self.assertEqual(code, 200)
            self.assertNotIn("X-Vision-Fretboard", headers)
            self.assertEqual(FakeDetector.instances, [])
            code, headers, jpeg = request(server, "/vision/frame.jpg?auto=1")
            self.assertEqual(code, 200)
            geometry = json.loads(headers["X-Vision-Fretboard"])
            self.assertEqual(geometry["sequence"], int(headers["X-Vision-Sequence"]))
            self.assertEqual(geometry["stream_id"], headers["X-Vision-Stream"])
            self.assertEqual(jpeg, b"first")
            self.assertEqual(headers["X-Vision-Tracking"], headers["X-Vision-Fretboard"])
            self.now += 1.1
            code, headers, body = request(server, "/vision/frame.jpg?auto=1")
            self.assertEqual(code, 503)
            self.assertNotIn("X-Vision-Fretboard", headers)


if __name__ == "__main__":
    unittest.main()

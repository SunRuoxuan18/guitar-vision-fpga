"""Event-controlled worker scheduling, provenance, history, and shutdown tests."""

import base64
import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fingertip_worker
from fingertip_heatmap import POINT_NAMES
from fingertip_worker import FingertipWorker


class Clock:
    def __init__(self):
        self.value = 100.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def advance(self, seconds):
        with self.lock:
            self.value += seconds


class ControlledModel:
    def __init__(self, count=4):
        self.entered = [threading.Event() for _ in range(count)]
        self.release = [threading.Event() for _ in range(count)]
        self.calls = []
        self.failures = {}
        self.close = mock.Mock()

    def infer_rgba(self, crop, roi):
        index = len(self.calls)
        self.calls.append((crop, roi, threading.get_ident()))
        self.entered[index].set()
        if not self.release[index].wait(5):
            raise RuntimeError("Test did not release model")
        if index in self.failures:
            raise self.failures[index]
        return {"points": {name: {"x": float(i), "y": float(i + 1), "raw_score": -3.0 + i}
                           for i, name in enumerate(POINT_NAMES)}, "roi": roi}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.model = ControlledModel()
        self.encoder_threads = []

        def encode(crop):
            self.encoder_threads.append(threading.get_ident())
            return {"jpeg_base64": "test-jpeg", "mime_type": "image/jpeg", "width": 208, "height": 208}

        self.encoder = mock.patch.object(fingertip_worker, "_encode_preview", side_effect=encode).start()
        self.addCleanup(mock.patch.stopall)
        self.worker = FingertipWorker(self.model, clock=self.clock)
        self.addCleanup(self.cleanup_worker)
        self.frame = np.zeros((90, 120, 4), np.uint8)
        self.frame[7:28, 11:52] = [201, 57, 11, 255]
        self.roi = (11.25, 7.5, 40.5, 20.25)

    def cleanup_worker(self):
        self.worker.request_stop()
        for event in self.model.release:
            event.set()
        self.assertTrue(self.worker.shutdown(timeout=5))

    def submit(self, sequence, stream="camera", generation=1, **kwargs):
        return self.worker.submit(self.frame, self.roi, stream, generation, sequence, **kwargs)

    def entered(self, index):
        self.assertTrue(self.model.entered[index].wait(3), "Model did not start")

    def completed(self, sequence):
        with self.worker._condition:
            self.assertTrue(self.worker._condition.wait_for(
                lambda: self.worker._last_outcome is not None
                and self.worker._last_outcome["sequence"] == sequence
                and not self.worker._running, timeout=3), "Result was not published")

    def test_only_pending_slot_is_overwritten_and_model_is_serial(self):
        self.submit(1)
        self.entered(0)
        self.submit(2)
        self.submit(3)
        snap = self.worker.snapshot("camera", 1, 3)
        self.assertTrue(snap["running"])
        self.assertTrue(snap["pending"])
        self.assertEqual(snap["dropped"], 1)
        self.assertEqual(len(self.model.calls), 1)
        self.model.release[0].set()
        self.entered(1)
        self.model.release[1].set()
        self.completed(3)
        snap = self.worker.snapshot("camera", 1, 3)
        self.assertEqual(snap["sequence"], 3)
        self.assertEqual(len(self.model.calls), 2)
        self.assertEqual(self.encoder.call_count, 2)
        self.assertEqual(self.model.calls[0][2], self.model.calls[1][2])
        self.assertEqual(self.encoder_threads, [self.model.calls[0][2]] * 2)
        self.assertNotEqual(self.model.calls[0][2], threading.get_ident())

    def test_copies_only_effective_roi_and_restores_fullframe_coordinates(self):
        self.submit(4)
        self.entered(0)
        crop, local_roi, _ = self.model.calls[0]
        self.assertEqual(crop.shape, (21, 41, 4))
        self.assertEqual(crop.nbytes, 21 * 41 * 4)
        self.assertFalse(np.shares_memory(crop, self.frame))
        self.assertFalse(crop.flags.writeable)
        self.assertEqual(local_roi, (0, 0, 41, 21))
        self.frame[:] = 0
        np.testing.assert_array_equal(crop[0, 0], [201, 57, 11, 255])
        self.model.release[0].set()
        self.completed(4)
        snap = self.worker.snapshot("camera", 1, 4)
        self.assertEqual(snap["status"], "ok")
        self.assertEqual(snap["roi"], (11, 7, 41, 21))
        self.assertEqual(snap["points"]["index"], {"x": 11, "y": 8, "raw_score": -3})
        self.assertEqual(snap["last_result"]["preview"]["jpeg_base64"], "test-jpeg")
        snap["last_result"]["points"]["index"]["x"] = 999
        snap["points"]["index"]["y"] = 999
        self.assertEqual(self.worker.snapshot("camera", 1, 4)["points"]["index"]["y"], 8)
        self.assertEqual(self.worker.snapshot("camera", 1, 4)["last_result"]["points"]["index"]["x"], 11)
        json.dumps(self.worker.snapshot("camera", 1, 4), allow_nan=False)
        self.assertEqual(self.encoder.call_count, 1)

    def test_capture_age_includes_wait_and_936ms_inference_is_stale(self):
        self.submit(1, capture_monotonic=99.9)
        self.entered(0)
        self.clock.advance(0.936)
        self.model.release[0].set()
        self.completed(1)
        snap = self.worker.snapshot("camera", 1, 1)
        self.assertEqual(snap["status"], "stale")
        self.assertIsNone(snap["points"])
        self.assertAlmostEqual(snap["inference_ms"], 936)
        self.assertAlmostEqual(snap["age_ms"], 1036)
        self.assertEqual(snap["capture_monotonic"], 99.9)
        self.assertAlmostEqual(snap["completed_monotonic"], 100.936)
        self.assertEqual(snap["last_result"]["stale_reasons"], ["age"])
        self.assertIsNotNone(snap["last_result"]["points"])
        self.clock.advance(0.1)
        self.assertAlmostEqual(self.worker.snapshot("camera", 1, 1)["age_ms"], 1136)
        self.assertEqual(self.encoder.call_count, 1)

    def test_configurable_ttl_never_allows_overlay_on_a_different_sequence(self):
        self.worker.max_age_ms = 1500
        self.submit(1)
        self.entered(0)
        self.clock.advance(0.936)
        self.model.release[0].set()
        self.completed(1)
        self.assertEqual(self.worker.snapshot("camera", 1, 1)["status"], "ok")
        snap = self.worker.snapshot("camera", 1, 2)
        self.assertEqual(snap["status"], "stale")
        self.assertIsNone(snap["points"])
        self.assertEqual(snap["last_result"]["stale_reasons"], ["sequence"])
        self.assertEqual(snap["last_result"]["sequence"], 1)
        future = self.worker.snapshot("camera", 1, 0)
        self.assertEqual(future["status"], "sequence_mismatch")
        self.assertIsNone(future["last_result"])

    def test_pending_queue_wait_counts_toward_age_but_not_inference_ms(self):
        self.submit(1)
        self.entered(0)
        self.submit(2)
        self.clock.advance(0.5)
        self.model.release[0].set()
        self.entered(1)
        self.clock.advance(0.1)
        self.model.release[1].set()
        self.completed(2)
        snap = self.worker.snapshot("camera", 1, 2)
        self.assertAlmostEqual(snap["age_ms"], 600)
        self.assertAlmostEqual(snap["inference_ms"], 100)
        self.assertEqual(snap["status"], "stale")
        self.assertIsNone(snap["points"])

    def test_other_stream_or_generation_hides_points_and_history(self):
        self.submit(1)
        self.entered(0)
        self.model.release[0].set()
        self.completed(1)
        for stream, generation in (("other", 1), ("camera", 2)):
            snap = self.worker.snapshot(stream, generation, 1)
            self.assertEqual(snap["status"], "source_mismatch")
            self.assertIsNone(snap["points"])
            self.assertIsNone(snap["last_result"])
            self.assertIsNone(snap["sequence"])
            self.assertIsNone(snap["fault"])

    def test_identity_switch_discards_old_inflight_publication_including_aba(self):
        self.submit(1)
        self.entered(0)
        self.submit(1, stream="other")
        self.submit(1)
        self.model.release[0].set()
        self.entered(1)
        snap = self.worker.snapshot("camera", 1, 1)
        self.assertEqual(snap["status"], "pending")
        self.assertIsNone(snap["last_result"])
        self.assertEqual(snap["dropped"], 1)
        self.model.release[1].set()
        self.completed(1)
        self.assertEqual(self.worker.snapshot("camera", 1, 1)["status"], "ok")

    def test_old_or_duplicate_submissions_do_not_replace_pending(self):
        self.submit(2)
        self.entered(0)
        self.assertTrue(self.submit(4))
        self.assertFalse(self.submit(3))
        self.assertFalse(self.submit(4))
        self.assertEqual(self.worker.snapshot("camera", 1, 4)["dropped"], 2)
        self.model.release[0].set()
        self.entered(1)
        self.model.release[1].set()
        self.completed(4)
        self.assertEqual(self.worker.snapshot("camera", 1, 4)["sequence"], 4)

    def test_exception_publishes_source_fault_retains_history_and_recovers(self):
        self.submit(1)
        self.entered(0)
        self.model.release[0].set()
        self.completed(1)
        self.model.failures[1] = RuntimeError("PL failed")
        self.submit(2)
        self.entered(1)
        self.clock.advance(0.05)
        self.model.release[1].set()
        self.completed(2)
        snap = self.worker.snapshot("camera", 1, 2)
        self.assertEqual(snap["status"], "fault")
        self.assertEqual(snap["fault"], {"type": "RuntimeError", "message": "PL failed"})
        self.assertEqual(snap["sequence"], 2)
        self.assertEqual(snap["last_result"]["sequence"], 1)
        self.assertTrue(snap["last_result"]["stale"])
        self.assertIsNone(snap["points"])
        self.assertFalse(snap["running"])
        self.assertEqual(self.encoder.call_count, 1)
        self.submit(3)
        self.entered(2)
        self.model.release[2].set()
        self.completed(3)
        self.assertIsNone(self.worker.snapshot("camera", 1, 3)["fault"])

    def test_preview_encoding_error_is_a_fault_and_never_current_points(self):
        self.encoder.side_effect = RuntimeError("JPEG failed")
        self.submit(1)
        self.entered(0)
        self.model.release[0].set()
        self.completed(1)
        snap = self.worker.snapshot("camera", 1, 1)
        self.assertEqual(snap["status"], "fault")
        self.assertEqual(snap["fault"]["message"], "JPEG failed")
        self.assertIsNone(snap["points"])
        self.assertIsNone(snap["last_result"])

    def test_shutdown_drops_pending_waits_for_inflight_and_never_closes_model(self):
        self.submit(1)
        self.entered(0)
        self.submit(2)
        self.assertFalse(self.worker.shutdown(wait=False))
        self.assertFalse(self.worker.shutdown(timeout=0))
        snap = self.worker.snapshot("camera", 1, 2)
        self.assertEqual(snap["status"], "stopping")
        self.assertTrue(snap["running"])
        self.assertFalse(snap["pending"])
        self.assertEqual(snap["dropped"], 1)
        self.model.close.assert_not_called()
        with self.assertRaises(RuntimeError):
            self.submit(3)
        self.model.release[0].set()
        self.assertTrue(self.worker.shutdown(timeout=3))
        snap = self.worker.snapshot("camera", 1, 2)
        self.assertEqual(snap["status"], "stopped")
        self.assertFalse(snap["running"])
        self.assertIsNone(snap["points"])
        self.assertEqual(len(self.model.calls), 1)
        self.model.close.assert_not_called()
        self.assertTrue(self.worker.shutdown(timeout=0))

    def test_shutdown_wait_true_joins_before_returning(self):
        self.submit(1)
        self.entered(0)
        returned = threading.Event()

        def close():
            self.worker.shutdown()
            returned.set()

        closer = threading.Thread(target=close)
        closer.start()
        try:
            with self.worker._condition:
                self.assertTrue(self.worker._condition.wait_for(lambda: self.worker._stopping, timeout=3))
            self.assertFalse(returned.is_set())
            self.model.release[0].set()
            self.assertTrue(returned.wait(3))
        finally:
            self.model.release[0].set()
            closer.join(3)
        self.assertFalse(closer.is_alive())

    def test_shutdown_waits_for_encoding_and_completed_timestamp_includes_it(self):
        encoding = threading.Event()
        release = threading.Event()

        def encode(crop):
            encoding.set()
            if not release.wait(3):
                raise RuntimeError("Test did not release JPEG encoder")
            return {"jpeg_base64": "jpeg", "mime_type": "image/jpeg", "width": 208, "height": 208}

        self.encoder.side_effect = encode
        try:
            self.submit(1)
            self.entered(0)
            self.clock.advance(0.1)
            self.model.release[0].set()
            self.assertTrue(encoding.wait(3))
            self.assertFalse(self.worker.shutdown(timeout=0))
            self.assertTrue(self.worker.snapshot("camera", 1, 1)["running"])
            self.clock.advance(0.2)
        finally:
            release.set()
        self.assertTrue(self.worker.shutdown(timeout=3))
        snap = self.worker.snapshot("camera", 1, 1)
        self.assertAlmostEqual(snap["inference_ms"], 100)
        self.assertAlmostEqual(snap["completed_monotonic"], 100.3)
        self.assertAlmostEqual(snap["last_result"]["age_ms"], 300)
        self.assertTrue(snap["last_result"]["stale"])
        self.model.close.assert_not_called()

    def test_reject_invalid_input_before_model_or_pending_slot(self):
        for frame, roi in ((None, self.roi), (self.frame.astype(np.float32), self.roi),
                           (self.frame[:, :, :3], self.roi), (self.frame, (-1, 0, 10, 10)),
                           (self.frame, (0, 0, np.nan, 10)), (self.frame, (0, 0, 0, 10)),
                           (self.frame, (0, 0, 121, 10))):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                self.worker.submit(frame, roi, "camera", 1, 1)
        for stream, generation, sequence in (("", 1, 1), (None, 1, 1), ("x", True, 1),
                                              ("x", 1, -1), ("x", 1, 1.5)):
            with self.assertRaises(ValueError):
                self.worker.submit(self.frame, self.roi, stream, generation, sequence)
        for captured in (np.nan, np.inf, 101, -1, True):
            with self.assertRaises(ValueError):
                self.submit(1, capture_monotonic=captured)
        self.assertEqual(self.worker.snapshot("camera", 1, 1)["status"], "idle")
        self.assertEqual(len(self.model.calls), 0)

    def test_invalid_model_points_fail_closed(self):
        with mock.patch.object(self.model, "infer_rgba", return_value={
                "points": {name: {"x": 0, "y": 0, "raw_score": np.nan} for name in POINT_NAMES}}):
            self.submit(1)
            self.completed(1)
        snap = self.worker.snapshot("camera", 1, 1)
        self.assertEqual(snap["status"], "fault")
        self.assertIsNone(snap["points"])
        self.encoder.assert_not_called()


class PreviewTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for JPEG preview")
    def test_real_preview_is_source_roi_208_rgb_correct_encoded_once_in_worker(self):
        import cv2
        model = ControlledModel(count=1)
        worker = FingertipWorker(model, max_age_ms=1500)
        frame = np.zeros((90, 120, 4), np.uint8)
        frame[7:28, 11:52] = [220, 40, 10, 127]
        try:
            with mock.patch.object(cv2, "imencode", wraps=cv2.imencode) as encoder:
                worker.submit(frame, (11, 7, 41, 21), "camera", 1, 1)
                self.assertTrue(model.entered[0].wait(3))
                frame[:] = [0, 255, 0, 255]
                model.release[0].set()
                with worker._condition:
                    self.assertTrue(worker._condition.wait_for(lambda: worker._last_result is not None, timeout=3))
                snap = worker.snapshot("camera", 1, 2)
                preview = snap["last_result"]["preview"]
                jpeg = base64.b64decode(preview["jpeg_base64"], validate=True)
                decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, (208, 208, 3))
                np.testing.assert_allclose(decoded[100, 100], [10, 40, 220], atol=3)
                self.assertEqual(snap["last_result"]["roi"], (11, 7, 41, 21))
                self.assertIsNone(snap["points"])
                self.assertTrue(snap["last_result"]["stale"])
                for _ in range(5):
                    worker.snapshot("camera", 1, 3)
                self.assertEqual(encoder.call_count, 1)
                json.dumps(snap, allow_nan=False)
        finally:
            model.release[0].set()
            self.assertTrue(worker.shutdown(timeout=3))

    def test_invalid_constructor_options_never_start_a_thread(self):
        for model, kwargs in ((object(), {}), (ControlledModel(), {"max_age_ms": 0}),
                              (ControlledModel(), {"max_age_ms": np.nan}),
                              (ControlledModel(), {"max_age_ms": True}),
                              (ControlledModel(), {"clock": 4})):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FingertipWorker(model, **kwargs)


if __name__ == "__main__":
    unittest.main()

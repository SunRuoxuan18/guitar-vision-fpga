"""Host-only optical flow, provenance, loss and bounded replay checks."""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fingertip_heatmap import POINT_NAMES
from fingertip_tracking import FourTipTracker


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds=.03):
        self.value += seconds


class Flow:
    backend_name = "test_cpu_opencv"

    def __init__(self):
        self.calls = []
        self.fault = False

    def __call__(self, previous, current, points):
        self.calls.append((previous, current, points.copy()))
        return cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(21, 21), maxLevel=3,
                                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))


class TrackerTests(unittest.TestCase):
    def setUp(self):
        threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        self.addCleanup(cv2.setNumThreads, threads)
        # Correctness must not depend on ARM scheduling or real LK duration.
        # Budget tests override this timer with explicit elapsed times.
        timer = mock.patch("fingertip_tracking.time.perf_counter", return_value=0.0)
        timer.start()
        self.addCleanup(timer.stop)
        self.clock = Clock()
        self.tracker = FourTipTracker(clock=self.clock)
        self.centers = [(100, 120), (240, 120), (380, 300), (520, 300)]
        rng = np.random.default_rng(14)
        self.patches = [rng.integers(20, 235, (64, 64), dtype=np.uint8) for _ in POINT_NAMES]
        self.frame = self.image()
        self.source = self.result(0)

    def image(self, offsets=None, hidden=()):
        offsets = offsets or [(0, 0)] * 4
        frame = np.full((480, 640), 100, np.uint8)
        for i, ((x, y), (dx, dy)) in enumerate(zip(self.centers, offsets)):
            if i not in hidden:
                frame[y + dy - 32:y + dy + 32, x + dx - 32:x + dx + 32] = self.patches[i]
        return frame

    def result(self, sequence, captured=None):
        return {"stream_id": "camera", "generation": 1, "sequence": sequence,
                "capture_monotonic": self.clock() if captured is None else captured,
                "roi": (40, 60, 550, 330),
                "points": {name: {"x": x, "y": y, "raw_score": -10.0 + i}
                           for i, (name, (x, y)) in enumerate(zip(POINT_NAMES, self.centers))}}

    def update(self, sequence, frame=None, result=None, stream="camera", generation=1):
        return self.tracker.update(self.frame if frame is None else frame,
                                   None if result is None else {"last_result": result},
                                   {"stream": stream, "generation": generation, "sequence": sequence})

    def assert_points(self, output, offsets=None):
        self.assertEqual(output["status"], "tracking", output)
        offsets = offsets or [(0, 0)] * 4
        self.assertEqual(set(output["points"]), set(POINT_NAMES))
        for name, (x, y), (dx, dy) in zip(POINT_NAMES, self.centers, offsets):
            self.assertAlmostEqual(output["points"][name]["x"], x + dx, delta=1)
            self.assertAlmostEqual(output["points"][name]["y"], y + dy, delta=1)

    def fast_history(self, backend=None, count=2, gap=.25):
        self.tracker = FourTipTracker(backend, self.clock, fast_catchup=True)
        source = self.result(0)
        for index in range(count):
            self.update(index * 3)
            self.clock.advance(gap)
        return source, count * 3

    def test_independent_translation_and_raw_negative_scores(self):
        seeded = self.update(0, result=self.source)
        self.assert_points(seeded)
        self.assertEqual(seeded["keypoint_source"], "neural")
        offsets = [(4, 0), (0, 0), (0, 12), (-6, 0)]
        self.clock.advance()
        output = self.update(1, self.image(offsets), self.source)
        self.assert_points(output, offsets)
        self.assertEqual(output["source_sequence"], 0)
        self.assertEqual(output["sequence"], 1)
        self.assertEqual(output["keypoint_source"], "optical_flow")
        self.assertEqual(output["method"], "ps_opencv_lk_fb_patch")
        self.assertEqual(output["points"]["index"]["raw_score"], -10)
        json.dumps(output, allow_nan=False)

    def test_patch_quality_is_batched_and_preserves_independent_motion(self):
        quality = mock.Mock()
        def evaluate(previous, current, anchors, original, returned):
            self.assertEqual(previous.shape, (4, 11, 11))
            self.assertEqual(original.shape, (4, 2))
            return np.array([[np.linalg.norm(q-p), self.tracker._correlation(old,new),
                self.tracker._correlation(anchor,new), new.std(), old.std()]
                for old,new,anchor,p,q in zip(previous,current,anchors,original,returned)])
        quality.evaluate.side_effect = evaluate
        self.tracker = FourTipTracker(clock=self.clock, patch_quality=quality)
        self.update(0, result=self.source)
        offsets = [(4,0),(0,0),(0,12),(-6,0)]
        self.clock.advance()
        output = self.update(1, self.image(offsets))
        self.assert_points(output, offsets)
        quality.evaluate.assert_called_once()
        self.assertEqual(output['diagnostics']['patch_backend'], 'native_batch')

    def test_native_rounding_near_threshold_uses_existing_float32_decision(self):
        for column, value in ((0,1.50000001),(1,.87999999),(2,.87999999),
                              (3,4.99999999),(4,4.99999999)):
            with self.subTest(column=column):
                metrics = np.tile([0.,1.,1.,50.,50.], (4,1))
                metrics[:,column] = value
                quality = mock.Mock()
                quality.evaluate.return_value = metrics
                self.tracker = FourTipTracker(clock=self.clock, patch_quality=quality)
                self.update(0, result=self.source)
                self.clock.advance()
                self.assert_points(self.update(1))
                quality.evaluate.assert_called_once()

    def test_invalid_native_quality_clears_tips(self):
        quality = mock.Mock()
        quality.evaluate.side_effect = ValueError('nonfinite patches')
        self.tracker = FourTipTracker(clock=self.clock, patch_quality=quality)
        self.update(0, result=self.source)
        self.clock.advance()
        output = self.update(1)
        self.assertEqual(output['points'], {})
        self.assertEqual(output['status'], 'lost')
        with self.assertRaises(ValueError):
            FourTipTracker(patch_quality=object())

    def test_sparse_camera_ids_track_at_analysis_rate(self):
        self.assert_points(self.update(40, result=self.result(40)))
        for sequence, offsets in [(44, [(4, 0), (0, 0), (0, 12), (-6, 0)]),
                                  (49, [(4, 0), (8, 12), (0, 12), (-6, 0)])]:
            self.clock.advance(.25)
            output = self.update(sequence, self.image(offsets))
            self.assert_points(output, offsets)
            self.assertEqual(output["source_sequence"], 40)
            self.assertEqual(output["sequence"], sequence)
            self.assertEqual(output["keypoint_source"], "optical_flow")
        self.assertEqual(output["diagnostics"]["history_size"], 3)

    def test_worker_snapshot_contract_with_submit_before_tracker(self):
        self.clock.advance(.012)
        source = self.result(40)
        self.clock.advance(.008)
        options = {"stream": "camera", "generation": 1, "sequence": 40,
                   "fast_tracking": True, "transport_roi": None}
        pending = {"stream_id": None, "generation": None, "sequence": None,
                   "capture_monotonic": None, "completed_monotonic": None,
                   "age_ms": None, "inference_ms": None, "roi": None,
                   "status": "pending", "points": None, "last_result": None,
                   "current_source": {"stream_id": "camera", "generation": 1, "sequence": 40},
                   "fault": None, "running": True, "pending": False}
        output = self.tracker.update(self.frame, pending, options, {"status": "tracking"})
        self.assertEqual(output["status"], "pending")
        self.assertEqual(output["points"], {})
        self.clock.advance(.25)
        options["sequence"] = 44
        history = dict(source, status="stale", stale=True, stale_reasons=["sequence", "age"],
                       age_ms=258., completed_monotonic=self.clock(), inference_ms=250.,
                       preview={"jpeg_base64": "unused", "mime_type": "image/jpeg"}, fault=None)
        snapshot = dict(history, points=None, last_result=history,
                        current_source={"stream_id": "camera", "generation": 1, "sequence": 44},
                        running=False, pending=False)
        output = self.tracker.update(self.image([(4, 0)] * 4), snapshot, options,
                                     {"status": "tracking"})
        self.assert_points(output, [(4, 0)] * 4)
        self.assertEqual((output["stream_id"], output["generation"], output["sequence"]),
                         ("camera", 1, 44))
        self.assertEqual(output["source_sequence"], 40)
        self.assertEqual(output["keypoint_source"], "optical_flow")
        self.assertAlmostEqual(output["detector_age_ms"], 258.)
        self.assertTrue({"status", "points", "source_sequence", "sequence", "stream_id", "generation",
                         "keypoint_source", "method", "detector_age_ms", "processing_ms", "diagnostics"}
                        <= output.keys())
        self.assertNotIn("preview", output)
        json.dumps(output, allow_nan=False)

    def test_one_finger_motion_does_not_move_other_fingers(self):
        self.update(0, result=self.source)
        self.clock.advance()
        offsets = [(0, 0), (8, 12), (0, 0), (0, 0)]
        self.assert_points(self.update(1, self.image(offsets)), offsets)

    def test_occlusion_loses_only_one_tip_and_same_source_cannot_resurrect(self):
        self.update(0, result=self.source)
        self.clock.advance()
        output = self.update(1, self.image(hidden=(1,)), self.source)
        self.assertEqual(output["status"], "tracking")
        self.assertEqual(set(output["points"]), set(POINT_NAMES) - {"middle"})
        self.assertIn("middle", output["diagnostics"]["lost_tips"])
        self.clock.advance()
        output = self.update(2, self.frame, self.source)
        self.assertNotIn("middle", output["points"])
        self.clock.advance()
        output = self.update(3, result=self.result(3))
        self.assert_points(output)
        self.assertEqual(output["keypoint_source"], "neural")

    def test_all_occluded_and_no_hand_affine_fallback(self):
        self.update(0, result=self.source)
        self.clock.advance()
        output = self.tracker.update(self.image(hidden=range(4)), {"last_result": self.source},
                                     {"stream": "camera", "generation": 1, "sequence": 1},
                                     hand_region={"points": self.source["points"]})
        self.assertEqual(output["status"], "lost")
        self.assertEqual(output["points"], {})

    def test_delayed_result_replays_every_pair(self):
        backend = Flow()
        self.tracker = FourTipTracker(backend, self.clock)
        for index, sequence in enumerate((0, 3, 6, 10, 13)):
            offsets = [(2 * index, 0)] * 4
            output = self.update(sequence, self.image(offsets), self.source if index == 4 else None)
            self.clock.advance(.25)
        self.assert_points(output, [(8, 0)] * 4)
        self.assertEqual(output["diagnostics"]["catchup_steps"], 4)
        self.assertEqual(output["sequence"], 13)
        self.assertEqual(output["keypoint_source"], "optical_flow")
        self.assertEqual(len(backend.calls), 8)
        for forward, reverse in zip(backend.calls[::2], backend.calls[1::2]):
            self.assertIs(forward[0], reverse[1])
            self.assertIs(forward[1], reverse[0])
            self.assertEqual(forward[0].shape, (200, 320))
        self.assertEqual(output["method"], "test_cpu_opencv_fb_patch")

    def test_step_bound_retains_partial_replay_until_exact_current_frame(self):
        self.assertEqual(self.tracker.BUDGET_SECONDS, .080)
        self.assertEqual(self.tracker.MAX_STEPS, 20)
        for sequence in range(26):
            self.update(sequence)
            self.clock.advance()
        output = self.update(26, result=self.source)
        self.assertEqual(output["status"], "pending")
        self.assertEqual(output["points"], {})
        self.assertEqual(output["diagnostics"]["catchup_steps"], 20)
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 20)
        self.assertEqual(output["diagnostics"]["catchup_remaining"], 6)
        for sequence in range(27, 32):
            self.clock.advance()
            output = self.update(sequence, result=self.source)
            if output["status"] == "tracking":
                break
        self.assert_points(output)
        self.assertEqual(output["diagnostics"]["tracked_sequence"], output["sequence"])
        self.assertEqual(output["source_sequence"], 0)

    def test_time_budget_defers_before_next_pair(self):
        for elapsed, expected_steps in ((.079, 3), (.080, 0), (.081, 0)):
            with self.subTest(elapsed=elapsed):
                self.tracker = FourTipTracker(clock=self.clock)
                self.assertEqual(self.tracker.BUDGET_SECONDS, .080)
                for sequence in range(3):
                    self.update(sequence)
                    self.clock.advance()
                timer = iter([0.0])
                with mock.patch("fingertip_tracking.time.perf_counter",
                                side_effect=lambda: next(timer, elapsed)):
                    output = self.update(3, result=self.source)
                self.assertEqual(output["diagnostics"]["catchup_steps"], expected_steps)
                if expected_steps:
                    self.assert_points(output)
                    self.assertFalse(output["diagnostics"]["budget_exhausted"])
                else:
                    self.assertEqual(output["status"], "pending")
                    self.assertEqual(output["points"], {})
                    self.assertTrue(output["diagnostics"]["budget_exhausted"])
                self.clock.advance()
                self.assert_points(self.update(4, result=self.source))

    def test_time_budget_stops_between_pairs_and_retains_progress(self):
        self.assertEqual(self.tracker.BUDGET_SECONDS, .080)
        for sequence in range(3):
            self.update(sequence)
            self.clock.advance()
        with mock.patch("fingertip_tracking.time.perf_counter", side_effect=[0, .079, .080, .081]):
            output = self.update(3, result=self.source)
        self.assertEqual(output["status"], "pending")
        self.assertEqual(output["points"], {})
        self.assertEqual(output["diagnostics"]["catchup_steps"], 1)
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 1)
        self.assertEqual(output["diagnostics"]["catchup_remaining"], 2)
        self.assertTrue(output["diagnostics"]["budget_exhausted"])
        self.clock.advance()
        output = self.update(4, result=self.source)
        self.assert_points(output)
        self.assertEqual(output["diagnostics"]["catchup_steps"], 3)
        self.assertEqual(output["source_sequence"], 0)

    def test_delayed_correction_does_not_rewind_pending_replay(self):
        self.tracker.MAX_STEPS = 2
        sources = {}
        for sequence in range(6):
            sources[sequence] = self.result(sequence)
            self.update(sequence)
            self.clock.advance(.1)
        output = self.update(6, result=sources[0])
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 2)
        self.clock.advance(.1)
        output = self.update(7, result=sources[0])
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 4)
        self.clock.advance(.1)
        output = self.update(8, result=sources[3])
        self.assertEqual(output["diagnostics"]["reason"], "detector_correction_deferred")
        self.assertEqual(output["source_sequence"], 0)
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 6)
        self.assertEqual(self.tracker._attempted_sequence, 0)
        self.clock.advance(.1)
        output = self.update(9, result=sources[3])
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 8)
        self.clock.advance(.1)
        output = self.update(10, result=sources[3])
        self.assert_points(output)
        self.assertEqual(output["source_sequence"], 0)

    def test_newer_source_can_advance_pending_replay(self):
        self.tracker.MAX_STEPS = 1
        sources = {}
        for sequence in range(5):
            sources[sequence] = self.result(sequence)
            self.update(sequence)
            self.clock.advance(.1)
        output = self.update(5, result=sources[0])
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 1)
        self.clock.advance(.1)
        output = self.update(6, result=sources[4])
        self.assertEqual(output["source_sequence"], 4)
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 5)

    def test_fast_catchup_delayed_independent_motion_uses_one_direct_pair(self):
        backend = Flow()
        backend.timings = {"kernel_ms": 5.0}
        source, sequence = self.fast_history(backend, count=6)
        offsets = [(8, 0), (0, 0), (0, 12), (-6, 0)]
        output = self.update(sequence, self.image(offsets), source)
        self.assert_points(output, offsets)
        self.assertTrue(output["diagnostics"]["direct_attempt"])
        self.assertTrue(output["diagnostics"]["direct_accepted"])
        self.assertEqual(output["diagnostics"]["catchup_steps"], 1)
        self.assertEqual(output["diagnostics"]["catchup_remaining"], 0)
        self.assertEqual(output["source_sequence"], 0)
        self.assertEqual(output["keypoint_source"], "optical_flow")
        self.assertEqual(len(backend.calls), 2)
        self.assertIs(backend.calls[0][0], self.tracker._history[0][2])
        self.assertIs(backend.calls[0][1], self.tracker._history[-1][2])
        self.assertIs(backend.calls[0][0], backend.calls[1][1])
        self.assertIs(backend.calls[0][1], backend.calls[1][0])
        self.assertEqual(output["diagnostics"]["flow_calls"], [{"kernel_ms": 5.0}] * 2)
        self.clock.advance()
        continued = self.update(sequence + 4, self.image(offsets), source)
        self.assert_points(continued, offsets)
        self.assertFalse(continued["diagnostics"]["direct_attempt"])
        self.assertFalse(continued["diagnostics"]["direct_accepted"])
        self.assertEqual(len(backend.calls), 4)
        json.dumps(output, allow_nan=False)

    def test_current_tracker_accepts_newer_delayed_detector_correction(self):
        backend = Flow()
        source, sequence = self.fast_history(backend, count=6)
        self.assert_points(self.update(sequence, result=source))
        captured = self.tracker._history[-2][1]
        correction = self.result(sequence-3, captured)
        self.clock.advance(.2)
        output = self.update(sequence+3, result=correction)
        self.assert_points(output)
        self.assertEqual(output['source_sequence'], sequence-3)
        self.assertTrue(output['diagnostics']['direct_accepted'])
        self.assertNotEqual(output['diagnostics']['reason'], 'detector_correction_deferred')

    def test_fast_catchup_single_finger_and_partial_occlusion_stay_independent(self):
        for hidden in ((), (0,)):
            with self.subTest(hidden=hidden):
                backend = Flow()
                source, sequence = self.fast_history(backend)
                offsets = [(0, 0), (8, 12), (0, 0), (0, 0)]
                output = self.update(sequence, self.image(offsets, hidden), source)
                self.assertEqual(output["status"], "tracking")
                expected = set(POINT_NAMES) - ({"index"} if hidden else set())
                self.assertEqual(set(output["points"]), expected)
                for name, (x, y), (dx, dy) in zip(POINT_NAMES, self.centers, offsets):
                    if name in expected:
                        self.assertAlmostEqual(output["points"][name]["x"], x + dx, delta=1)
                        self.assertAlmostEqual(output["points"][name]["y"], y + dy, delta=1)
                self.assertTrue(output["diagnostics"]["direct_accepted"])
                self.assertEqual(len(backend.calls), 2)
                if hidden:
                    self.assertIn("index", output["diagnostics"]["lost_tips"])

    def test_fast_catchup_zero_or_one_pair_uses_normal_path(self):
        for count in (0, 1):
            with self.subTest(count=count):
                backend = Flow()
                source, sequence = self.fast_history(backend, count=count)
                output = self.update(sequence, result=source)
                self.assert_points(output)
                self.assertFalse(output["diagnostics"]["direct_attempt"])
                self.assertFalse(output["diagnostics"]["direct_accepted"])
                self.assertEqual(output["diagnostics"]["catchup_steps"], count)
                self.assertEqual(len(backend.calls), count * 2)
                self.assertEqual(output["keypoint_source"], "optical_flow" if count else "neural")

    def test_fast_catchup_capture_age_bounds_include_zero_and_2500ms(self):
        for age, count, attempted in ((0., 2, True), (2.5, 5, True),
                                      (2.5001, 5, False), (3.5, 7, False), (4.5, 9, False)):
            with self.subTest(age=age):
                backend = Flow()
                source, sequence = self.fast_history(backend, count=count, gap=age / count)
                output = self.update(sequence, result=source)
                self.assertEqual(output["diagnostics"]["direct_attempt"], attempted)
                self.assertEqual(output["diagnostics"]["direct_accepted"], attempted)
                if age <= 4:
                    self.assert_points(output)
                    self.assertEqual(output["diagnostics"]["catchup_steps"], 1 if attempted else count)
                    self.assertEqual(len(backend.calls), 2 if attempted else count * 2)
                else:
                    self.assertEqual(output["diagnostics"]["reason"], "detector_expired")
                    self.assertEqual(output["points"], {})
                    self.assertEqual(backend.calls, [])

    def test_fast_catchup_invalid_source_and_snapshot_cannot_dispatch(self):
        for change in ({"stream_id": "other"}, {"generation": 2}, {"sequence": 99},
                       {"capture_monotonic": float("nan")}, {"capture_monotonic": 999.},
                       {"roi": (0, 0, 641, 480)}, {"fault": True}, {"points": {}}):
            with self.subTest(change=change):
                backend = Flow()
                source, sequence = self.fast_history(backend)
                output = self.update(sequence, result=dict(source, **change))
                self.assertEqual(output["points"], {})
                self.assertFalse(output["diagnostics"]["direct_attempt"])
                self.assertEqual(backend.calls, [])
        for changes in ({"current_source": {"stream_id": "camera", "generation": 1, "sequence": 0}},
                        {"status": "fault"}, {"fault": {"type": "RuntimeError"}}):
            with self.subTest(snapshot=changes):
                backend = Flow()
                source, sequence = self.fast_history(backend)
                snapshot = dict(last_result=source, **changes)
                output = self.tracker.update(self.frame, snapshot,
                    {"stream": "camera", "generation": 1, "sequence": sequence})
                self.assertEqual(output["points"], {})
                self.assertFalse(output["diagnostics"]["direct_attempt"])
                self.assertEqual(backend.calls, [])

    def test_fast_catchup_frame_gap_or_generation_change_cannot_bridge_history(self):
        for change in ("gap", "generation", "order"):
            with self.subTest(change=change):
                backend = Flow()
                source, sequence = self.fast_history(backend)
                if change == "gap":
                    self.clock.advance(.601)
                output = self.update(0 if change == "order" else sequence, result=source,
                                     generation=2 if change == "generation" else 1)
                self.assertEqual(output["points"], {})
                self.assertFalse(output["diagnostics"]["direct_attempt"])
                self.assertEqual(backend.calls, [])

    def test_fast_catchup_all_fail_restores_seed_losses_and_quality_before_replay(self):
        backend = Flow()
        source, sequence = self.fast_history(backend)
        source["points"]["little"]["visible"] = False
        offsets = [(8, 0), (0, 0), (0, 12), (-6, 0)]
        correlation = self.tracker._correlation
        correlations = []

        def fail_after_first_valid_tip(first, second):
            correlations.append(None)
            if len(correlations) == 3:
                raise ValueError("direct patch failed")
            return correlation(first, second)

        step = self.tracker._step
        attempts = []

        def inspected_step(previous, current):
            if not attempts:
                attempts.append(copy.deepcopy((self.tracker._tips, self.tracker._losses,
                                               self.tracker._quality)))
            elif len(attempts) == 1:
                tips, losses, quality = attempts[0]
                self.assertEqual(self.tracker._losses, losses)
                self.assertEqual(self.tracker._quality, quality)
                self.assertEqual(set(self.tracker._tips), set(tips))
                for name, tip in tips.items():
                    np.testing.assert_array_equal(self.tracker._tips[name]["point"], tip["point"])
                    np.testing.assert_array_equal(self.tracker._tips[name]["anchor"], tip["anchor"])
                attempts.append(None)
            step(previous, current)

        with mock.patch.object(self.tracker, "_correlation", side_effect=fail_after_first_valid_tip), \
                mock.patch.object(self.tracker, "_step", side_effect=inspected_step):
            output = self.update(sequence, self.image(offsets), source)
        self.assertEqual(output["status"], "tracking")
        self.assertEqual(set(output["points"]), set(POINT_NAMES) - {"little"})
        for name, (x, y), (dx, dy) in zip(POINT_NAMES[:3], self.centers, offsets):
            self.assertAlmostEqual(output["points"][name]["x"], x + dx, delta=1)
            self.assertAlmostEqual(output["points"][name]["y"], y + dy, delta=1)
        self.assertEqual(output["diagnostics"]["lost_tips"], {"little": "not_visible"})
        self.assertTrue(output["diagnostics"]["direct_attempt"])
        self.assertFalse(output["diagnostics"]["direct_accepted"])
        self.assertEqual(output["diagnostics"]["catchup_steps"], 3)
        self.assertEqual(len(backend.calls), 6)

    def test_fast_catchup_all_occluded_never_uses_hand_affine_points(self):
        backend = Flow()
        source, sequence = self.fast_history(backend)
        output = self.tracker.update(self.image(hidden=range(4)), {"last_result": source},
            {"stream": "camera", "generation": 1, "sequence": sequence},
            hand_region={"status": "tracking", "points": source["points"]})
        self.assertEqual(output["status"], "lost")
        self.assertEqual(output["points"], {})
        self.assertTrue(output["diagnostics"]["direct_attempt"])
        self.assertFalse(output["diagnostics"]["direct_accepted"])

    def test_fast_catchup_faults_propagate_without_restoration_or_replay(self):
        for error_type in (OSError, RuntimeError):
            for failure_call in (1, 2):
                with self.subTest(error=error_type, failure_call=failure_call):
                    flow = Flow()
                    calls = []
                    failure = error_type("direct PL fault")

                    def backend(previous, current, points):
                        calls.append(None)
                        if len(calls) == failure_call:
                            backend.fault = True
                            raise failure
                        return flow(previous, current, points)

                    backend.fault = False
                    source, sequence = self.fast_history(backend)
                    with self.assertRaises(error_type) as raised:
                        self.update(sequence, result=source)
                    self.assertIs(raised.exception, failure)
                    self.assertEqual(self.tracker._tips, {})
                    self.assertTrue(self.tracker._direct_attempt)
                    self.assertFalse(self.tracker._direct_accepted)
                    self.assertEqual(len(calls), failure_call)
                    self.clock.advance()
                    output = self.update(sequence + 3, result=source)
                    self.assertEqual(output["status"], "lost")
                    self.assertEqual(len(calls), failure_call)

    def test_fast_catchup_exhausted_budget_skips_direct_attempt(self):
        for elapsed, expected in ((.079, True), (.080, False), (.081, False)):
            with self.subTest(elapsed=elapsed):
                backend = Flow()
                source, sequence = self.fast_history(backend)
                timer = iter([0.])
                with mock.patch("fingertip_tracking.time.perf_counter",
                                side_effect=lambda: next(timer, elapsed)):
                    output = self.update(sequence, result=source)
                self.assertEqual(output["diagnostics"]["direct_attempt"], expected)
                self.assertEqual(output["diagnostics"]["direct_accepted"], expected)
                self.assertEqual(len(backend.calls), 2 if expected else 0)
                self.assertEqual(output["status"], "tracking" if expected else "pending")

    def test_fast_catchup_backend_fault_flags_never_restore_or_dispatch_reverse(self):
        for fault_before in (False, True):
            with self.subTest(fault_before=fault_before):
                flow = Flow()

                def backend(previous, current, points):
                    result = flow(previous, current, points)
                    backend.fault = True
                    return result

                backend.fault = fault_before
                source, sequence = self.fast_history(backend)
                with self.assertRaises(RuntimeError):
                    self.update(sequence, result=source)
                self.assertEqual(self.tracker._tips, {})
                self.assertTrue(self.tracker._direct_attempt)
                self.assertFalse(self.tracker._direct_accepted)
                self.assertEqual(len(flow.calls), 0 if fault_before else 1)

    def test_fast_catchup_rewind_guard_keeps_pending_progress(self):
        flow = Flow()
        elapsed = [0.]

        def backend(previous, current, points):
            moved, valid, error = flow(previous, current, points)
            if len(flow.calls) == 2:
                valid[:] = 0
                elapsed[0] = .081
            return moved, valid, error

        source, sequence = self.fast_history(backend, count=5, gap=.1)
        correction = self.result(3, captured=self.tracker._history[1][1])
        with mock.patch("fingertip_tracking.time.perf_counter", side_effect=lambda: elapsed[0]):
            output = self.update(sequence, result=source)
        self.assertEqual(output["status"], "pending")
        self.tracker.MAX_STEPS = 2
        self.clock.advance(.1)
        output = self.update(sequence + 3, result=source)
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 6)
        self.clock.advance(.1)
        output = self.update(sequence + 6, result=correction)
        self.assertEqual(output["source_sequence"], 0)
        self.assertEqual(output["diagnostics"]["reason"], "detector_correction_deferred")
        self.assertFalse(output["diagnostics"]["direct_attempt"])
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 12)

    def test_fast_catchup_failed_direct_time_counts_and_same_seed_is_not_retried(self):
        flow = Flow()
        elapsed = [0.]

        def backend(previous, current, points):
            moved, valid, error = flow(previous, current, points)
            if len(flow.calls) == 2:
                elapsed[0] = .081
                valid[:] = 0
            return moved, valid, error

        source, sequence = self.fast_history(backend)
        with mock.patch("fingertip_tracking.time.perf_counter", side_effect=lambda: elapsed[0]):
            output = self.update(sequence, result=source)
        self.assertEqual(output["status"], "pending")
        self.assertEqual(output["points"], {})
        self.assertEqual(output["diagnostics"]["catchup_steps"], 1)
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 0)
        self.assertTrue(output["diagnostics"]["budget_exhausted"])
        self.assertEqual(output["diagnostics"]["lost_tips"], {})
        self.assertEqual(output["diagnostics"]["quality"], dict.fromkeys(POINT_NAMES, {}))
        self.assertEqual(len(flow.calls), 2)
        self.assertAlmostEqual(output["processing_ms"], 81.)
        self.clock.advance()
        output = self.update(sequence + 3, result=source)
        self.assert_points(output)
        self.assertFalse(output["diagnostics"]["direct_attempt"])
        self.assertEqual(output["diagnostics"]["catchup_steps"], 3)
        self.assertEqual(len(flow.calls), 8)

    def test_fast_catchup_failed_direct_counts_toward_twenty_step_limit(self):
        flow = Flow()

        def backend(previous, current, points):
            moved, valid, error = flow(previous, current, points)
            if len(flow.calls) == 2:
                valid[:] = 0
            return moved, valid, error

        source, sequence = self.fast_history(backend, count=25, gap=.03)
        output = self.update(sequence, result=source)
        self.assertEqual(self.tracker.MAX_STEPS, 20)
        self.assertEqual(output["status"], "pending")
        self.assertEqual(output["diagnostics"]["catchup_steps"], 20)
        self.assertEqual(output["diagnostics"]["tracked_sequence"], 19 * 3)
        self.assertEqual(output["diagnostics"]["catchup_remaining"], 6)
        self.assertEqual(len(flow.calls), 40)

    def test_fast_catchup_refreshes_expiry_after_direct_flow(self):
        flow = Flow()

        def backend(previous, current, points):
            result = flow(previous, current, points)
            if len(flow.calls) == 2:
                self.clock.advance(4.)
            return result

        source, sequence = self.fast_history(backend)
        output = self.update(sequence, result=source)
        self.assertEqual(output["status"], "lost")
        self.assertEqual(output["points"], {})
        self.assertEqual(output["diagnostics"]["reason"], "detector_expired")
        self.assertTrue(output["diagnostics"]["direct_attempt"])
        self.assertFalse(output["diagnostics"]["direct_accepted"])
        self.assertEqual(output["detector_age_ms"], 4500.)
        self.assertEqual(len(flow.calls), 2)

    def test_generation_stream_and_snapshot_identity(self):
        self.update(0, result=self.source)
        self.clock.advance()
        output = self.update(0, result=self.source, generation=2)
        self.assertEqual(output["points"], {})
        self.assertEqual(output["diagnostics"]["reason"], "detector_identity")
        self.assertIsNone(output["source_sequence"])
        self.clock.advance()
        output = self.update(0, result=self.source, stream="other")
        self.assertEqual(output["points"], {})
        self.clock.advance()
        output = self.tracker.update(self.frame, {"last_result": self.source,
                                   "current_source": {"stream_id": "camera", "generation": 1, "sequence": 9}},
                                   {"stream": "camera", "generation": 1, "sequence": 0})
        self.assertEqual(output["points"], {})
        self.assertEqual(output["diagnostics"]["reason"], "snapshot_identity_or_order")

    def test_duplicate_reversed_and_future_sequence(self):
        self.update(0, result=self.source)
        output = self.update(0, result=self.source)
        self.assertEqual(output["status"], "lost")
        self.assertEqual(output["points"], {})
        self.clock.advance()
        output = self.update(1, result=self.result(4))
        self.assertEqual(output["diagnostics"]["reason"], "detector_future_sequence")
        self.clock.advance()
        output = self.update(3, result=self.source)
        self.assertEqual(output["points"], {})
        self.assertEqual(output["diagnostics"]["history_size"], 3)
        self.assertEqual(self.update(2)["diagnostics"]["reason"], "frame_order")

    def test_time_gap_and_expiry(self):
        self.update(0, result=self.source)
        self.clock.advance(.601)
        output = self.update(1, result=self.source)
        self.assertEqual(output["status"], "lost")
        self.assertEqual(output["points"], {})
        self.clock.advance()
        self.update(2, result=self.result(2))
        for sequence in range(3, 12):
            self.clock.advance(.5)
            output = self.update(sequence)
        self.assertEqual(output["status"], "lost")
        self.assertEqual(output["diagnostics"]["reason"], "detector_expired")
        self.assertGreater(output["detector_age_ms"], 4000)

    def test_evicted_frame_and_bounded_detached_history(self):
        for sequence in range(66):
            self.update(sequence)
            self.clock.advance()
        output = self.update(66, result=self.source)
        self.assertEqual(output["points"], {})
        self.assertEqual(output["diagnostics"]["reason"], "source_not_retained")
        self.assertEqual(len(self.tracker._history), 64)
        self.assertEqual(self.tracker._history[-1][2].shape, (200, 320))
        self.assertFalse(np.shares_memory(self.tracker._history[-1][2], self.frame))
        self.assertEqual(sum(item[2].nbytes for item in self.tracker._history), 64 * 64000)

    def test_invalid_roi_points_texture_and_future_timestamp(self):
        for roi in [(0, 0, 641, 480), (0, 0, -1, 10), (0, float("nan"), 10, 10), None]:
            with self.subTest(roi=roi):
                self.tracker = FourTipTracker(clock=self.clock)
                result = copy.deepcopy(self.source)
                result["roi"] = roi
                self.assertEqual(self.update(0, result=result)["points"], {})
        self.tracker = FourTipTracker(clock=self.clock)
        result = copy.deepcopy(self.source)
        result["points"]["index"]["x"] = float("inf")
        result["points"]["middle"]["x"] = 639
        output = self.update(0, result=result)
        self.assertEqual(set(output["points"]), {"ring", "little"})
        self.tracker = FourTipTracker(clock=self.clock)
        self.assertEqual(self.update(0, np.zeros_like(self.frame), self.source)["status"], "lost")
        self.tracker = FourTipTracker(clock=self.clock)
        result = self.result(0, self.clock() + 1)
        self.assertEqual(self.update(0, result=result)["diagnostics"]["reason"], "detector_timestamp")

    def test_backend_failure_does_not_fall_back_and_failed_result_is_not_retried(self):
        backend = mock.Mock(side_effect=RuntimeError("no hardware"))
        backend.fault = False
        self.tracker = FourTipTracker(backend, self.clock)
        self.update(0, result=self.source)
        self.clock.advance()
        output = self.update(1)
        self.assertEqual(output["status"], "lost")
        self.assertEqual(backend.call_count, 1)
        self.clock.advance()
        output = self.update(2, result=self.source)
        self.assertEqual(output["status"], "lost")
        self.assertEqual(backend.call_count, 1)

    def test_faulted_backend_propagates_original_error_and_never_runs_again(self):
        for error_type in (OSError, RuntimeError):
            for failure_call in (1, 2):
                with self.subTest(error=error_type, failure_call=failure_call):
                    flow = Flow()
                    calls = []
                    failure = error_type("PL fault")

                    def backend(previous, current, points):
                        calls.append(None)
                        if len(calls) == failure_call:
                            backend.fault = True
                            raise failure
                        return flow(previous, current, points)

                    backend.fault = False
                    self.tracker = FourTipTracker(backend, self.clock)
                    self.update(0, result=self.result(0))
                    self.clock.advance(.25)
                    with self.assertRaises(error_type) as raised:
                        self.update(3)
                    self.assertIs(raised.exception, failure)
                    self.assertEqual(self.tracker._tips, {})
                    self.assertEqual(len(calls), failure_call)
                    self.clock.advance(.25)
                    self.update(6, result=self.result(6))
                    self.clock.advance(.25)
                    with self.assertRaises(RuntimeError):
                        self.update(9)
                    self.assertEqual(len(calls), failure_call)

    def test_backend_fault_flag_after_return_stops_before_reverse(self):
        flow = Flow()
        calls = []

        def backend(previous, current, points):
            calls.append(None)
            result = flow(previous, current, points)
            backend.fault = True
            return result

        backend.fault = False
        self.tracker = FourTipTracker(backend, self.clock)
        self.update(0, result=self.source)
        self.clock.advance(.25)
        with self.assertRaises(RuntimeError):
            self.update(3)
        self.assertEqual(len(calls), 1)

    def test_detector_fault_and_explicit_invisible_points_are_skipped(self):
        result = copy.deepcopy(self.source)
        result["fault"] = {"type": "RuntimeError", "message": "inference failed"}
        output = self.update(0, result=result)
        self.assertEqual(output["points"], {})
        self.assertEqual(output["diagnostics"]["reason"], "detector_fault")
        self.clock.advance(.25)
        result = self.result(3)
        result["points"]["middle"]["visible"] = False
        result["points"]["ring"]["visible"] = np.bool_(False)
        result["points"]["little"]["visible"] = True
        output = self.update(3, result=result)
        self.assertEqual(set(output["points"]), {"index", "little"})
        self.assertEqual(output["keypoint_source"], "neural")
        self.assertEqual(output["diagnostics"]["lost_tips"]["middle"], "not_visible")
        self.assertEqual(output["diagnostics"]["lost_tips"]["ring"], "not_visible")
        self.assertEqual(output["points"]["index"]["raw_score"], -10)

    def test_snapshot_fault_clears_tips_and_blocks_historical_revival(self):
        for seeded in (False, True):
            for fault_fields in ({"status": "fault", "fault": None},
                                 {"status": "stale", "fault": {"type": "RuntimeError"}}):
                with self.subTest(seeded=seeded, fault_fields=fault_fields):
                    backend = Flow()
                    self.tracker = FourTipTracker(backend, self.clock)
                    source = self.result(0)
                    self.update(0, result=source if seeded else None)
                    self.clock.advance(.25)
                    snapshot = dict(self.result(3), **fault_fields)
                    snapshot["points"] = None
                    snapshot["last_result"] = source
                    snapshot["current_source"] = {"stream_id": "camera", "generation": 1, "sequence": 3}
                    output = self.tracker.update(self.frame, snapshot,
                                                 {"stream": "camera", "generation": 1, "sequence": 3})
                    self.assertEqual(output["status"], "lost")
                    self.assertEqual(output["points"], {})
                    self.assertEqual(output["diagnostics"]["reason"], "snapshot_fault")
                    self.assertEqual(output["diagnostics"]["catchup_steps"], 0)
                    self.assertEqual(self.tracker._tips, {})
                    self.assertEqual(backend.calls, [])
                    if seeded:
                        self.assertEqual(set(output["diagnostics"]["lost_tips"]), set(POINT_NAMES))
                    self.clock.advance(.25)
                    output = self.update(6, result=source)
                    self.assertEqual(output["status"], "lost")
                    self.assertEqual(output["points"], {})
                    self.assertEqual(backend.calls, [])
                    self.clock.advance(.25)
                    self.update(9)
                    newer = self.result(9)
                    self.clock.advance(.25)
                    output = self.update(12, result=newer)
                    self.assert_points(output)
                    self.assertEqual(output["source_sequence"], 9)
                    self.assertEqual(output["keypoint_source"], "optical_flow")
                    self.assertEqual(len(backend.calls), 2)

    def test_snapshot_fault_cancels_pending_catchup(self):
        backend = Flow()
        self.tracker = FourTipTracker(backend, self.clock)
        self.tracker.MAX_STEPS = 1
        for sequence in (0, 3, 6):
            self.update(sequence)
            self.clock.advance(.25)
        output = self.update(9, result=self.source)
        self.assertEqual(output["status"], "pending")
        self.assertEqual(output["points"], {})
        self.assertEqual(len(backend.calls), 2)
        self.clock.advance(.25)
        snapshot = {"status": "fault", "last_result": self.source, "fault": {"type": "RuntimeError"}}
        output = self.tracker.update(self.frame, snapshot,
                                     {"stream": "camera", "generation": 1, "sequence": 12})
        self.assertEqual(output["status"], "lost")
        self.assertEqual(output["points"], {})
        self.assertEqual(output["diagnostics"]["catchup_steps"], 0)
        self.assertEqual(output["diagnostics"]["catchup_remaining"], 0)
        self.assertEqual(len(backend.calls), 2)
        self.clock.advance(.25)
        self.assertEqual(self.update(15, result=self.source)["status"], "lost")
        self.assertEqual(len(backend.calls), 2)

    def test_age_is_refreshed_and_expired_after_expensive_flow(self):
        flow = Flow()

        def backend(previous, current, points):
            result = flow(previous, current, points)
            if len(flow.calls) == 2:
                self.clock.advance(4)
            return result

        self.tracker = FourTipTracker(backend, self.clock)
        self.update(0, result=self.source)
        self.clock.advance(.25)
        output = self.update(3)
        self.assertEqual(output["status"], "lost")
        self.assertEqual(output["points"], {})
        self.assertEqual(output["detector_age_ms"], 4250)
        self.assertEqual(output["diagnostics"]["reason"], "detector_expired")
        self.assertEqual(output["diagnostics"]["catchup_steps"], 1)

    def test_expired_delayed_result_is_rejected_without_flow(self):
        backend = Flow()
        self.tracker = FourTipTracker(backend, self.clock)
        for sequence in range(10):
            self.update(sequence)
            self.clock.advance(.5)
        output = self.update(10, result=self.source)
        self.assertEqual(output["diagnostics"]["reason"], "detector_expired")
        self.assertEqual(output["points"], {})
        self.assertEqual(backend.calls, [])

    def test_nonfinite_reverse_loses_only_affected_tip(self):
        backend = Flow()
        calls = []

        def damaged(previous, current, points):
            moved, status, error = backend(previous, current, points)
            calls.append(None)
            if len(calls) == 2:
                moved[0] = float("nan")
            return moved, status, error

        self.tracker = FourTipTracker(damaged, self.clock)
        self.update(0, result=self.source)
        self.clock.advance()
        output = self.update(1)
        self.assertEqual(output["status"], "tracking")
        self.assertEqual(set(output["points"]), set(POINT_NAMES) - {"index"})
        self.assertEqual(output["diagnostics"]["lost_tips"]["index"], "forward_backward")
        json.dumps(output, allow_nan=False)

    def test_invalid_frame_and_options(self):
        for frame in [np.zeros((200, 320), np.uint8), self.frame.astype(np.float32)]:
            with self.assertRaises(ValueError):
                self.update(0, frame)
        for sequence in [-1, True, 1.2]:
            with self.assertRaises(ValueError):
                self.update(sequence)


if __name__ == "__main__":
    unittest.main()

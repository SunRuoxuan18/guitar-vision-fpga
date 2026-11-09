import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neck_model import NeckModel
from vision_pc_bridge import LatticeSearch


class NeckModelTests(unittest.TestCase):
    @staticmethod
    def model_with_detector(detector):
        model = NeckModel.__new__(NeckModel)
        model.detector = detector
        model.confidence_threshold = .5
        model.lock = __import__('threading').Lock()
        model.last = {}
        model.last_detection = None
        model.last_error = None
        model.load_ms = None
        return model

    def test_full_frame_geometry_candidate_avoids_slow_refinement(self):
        detector = Mock()
        detector.detect.return_value = {
            "status": "proposal", "confidence": .9, "bbox": [.1, .3, .9, .7],
            "inference_ms": 4.0}
        model = self.model_with_detector(detector)
        candidate = [[{"x": .12, "y": .36}, {"x": .88, "y": .36},
                      {"x": .88, "y": .64}, {"x": .12, "y": .64}]]
        with patch("neck_model.find_neck_regions") as refine:
            result = model.detect(np.zeros((100, 200, 3), np.uint8), proposals=candidate)
        self.assertEqual(result["status"], "tracking")
        self.assertEqual(result["reason"], "model_bbox_geometry_gated")
        self.assertEqual(result["acquisition"], "neck_model_geometry_gated")
        self.assertEqual(result["geometry_gated_candidates"], 1)
        self.assertEqual(refine.call_count, 0)
        np.testing.assert_allclose([[point["x"], point["y"]] for point in result["corners"]],
                                   [[point["x"], point["y"]] for point in candidate[0]], atol=1e-6)

    def test_short_or_invalid_geometry_candidate_uses_existing_refinement(self):
        detector = Mock()
        detector.detect.return_value = {
            "status": "proposal", "confidence": .9, "bbox": [.1, .3, .9, .7],
            "inference_ms": 4.0}
        model = self.model_with_detector(detector)
        short = [[{"x": .45, "y": .42}, {"x": .55, "y": .42},
                  {"x": .55, "y": .58}, {"x": .45, "y": .58}]]
        refined = [[{"x": .12, "y": .4}, {"x": .88, "y": .4},
                    {"x": .88, "y": .6}, {"x": .12, "y": .6}]]
        with patch("neck_model.find_neck_regions", side_effect=[refined, []]) as refine:
            result = model.detect(np.zeros((100, 200, 3), np.uint8), proposals=short)
        self.assertEqual(result["status"], "tracking")
        self.assertEqual(result["acquisition"], "neck_model_geometry_refined")
        self.assertEqual(refine.call_count, 2)

    def test_bbox_is_refined_in_crop_and_mapped_to_original(self):
        model = NeckModel.__new__(NeckModel)
        model.detector = Mock()
        model.detector.detect.return_value = {
            "status": "proposal", "confidence": .9, "bbox": [.25, .25, .75, .75],
            "inference_ms": 4.0}
        model.confidence_threshold = .5
        model.lock = __import__('threading').Lock()
        model.last = {}
        model.last_detection = None
        model.last_error = None
        model.load_ms = None
        image = np.zeros((100, 200, 3), np.uint8)
        refined = [[{"x": .25, "y": .2}, {"x": .45, "y": .2},
                    {"x": .45, "y": .8}, {"x": .25, "y": .8}],
                   [{"x": .1, "y": .2}, {"x": .9, "y": .2},
                    {"x": .9, "y": .8}, {"x": .1, "y": .8}]]
        with patch("neck_model.find_neck_regions", side_effect=[refined, []]) as refine:
            result = model.detect(image)
        self.assertEqual(refine.call_count, 2)
        self.assertEqual(result["status"], "tracking")
        self.assertNotEqual(result["corners"], [{"x": .25, "y": .25}])
        self.assertLess(result["corners"][0]["x"], .3)
        self.assertGreater(result["corners"][2]["x"], .7)

    def test_model_gate_rejects_unrelated_borders_and_prefers_long_tight_extent(self):
        short = [dict(x=.2, y=.4), dict(x=.5, y=.4),
                 dict(x=.5, y=.6), dict(x=.2, y=.6)]
        full = [dict(x=.1, y=.4), dict(x=.9, y=.4),
                dict(x=.9, y=.6), dict(x=.1, y=.6)]
        expanded = [dict(x=.1, y=.3), dict(x=.9, y=.3),
                    dict(x=.9, y=.7), dict(x=.1, y=.7)]
        background = [dict(x=0, y=0), dict(x=1, y=0),
                      dict(x=1, y=.1), dict(x=0, y=.1)]
        eligible = NeckModel._model_supported([short, full, expanded, background], [0, .25, 1, .75])
        self.assertNotIn(background, eligible)
        self.assertEqual(NeckModel._select_proposal(eligible, (480, 640)), full)

    def test_no_detection_is_explicit(self):
        model = NeckModel.__new__(NeckModel)
        model.detector = Mock()
        model.detector.detect.return_value = {"status": "no_detection", "reason": "no_matching_detection"}
        model.confidence_threshold = .5
        model.lock = __import__('threading').Lock()
        model.last = {}
        model.last_detection = None
        model.last_error = None
        model.load_ms = None
        result = model.detect(np.zeros((20, 20, 3), np.uint8))
        self.assertEqual(result["status"], "no_detection")
        self.assertEqual(result["reason"], "no_matching_detection")

    def test_yolo_box_remains_valid_neck_candidate_when_geometry_refinement_fails(self):
        model = self.model_with_detector(Mock())
        model.detector.detect.return_value = {
            "status": "proposal", "confidence": .9, "bbox": [.1, .2, .9, .8],
            "inference_ms": 8.0}
        with patch("neck_model.find_neck_regions", return_value=[]):
            result = model.detect(np.zeros((100, 200, 3), np.uint8))
        self.assertEqual(result["status"], "tracking")
        self.assertEqual(result["reason"], "model_bbox_only")
        self.assertEqual(result["acquisition"], "yolo_bbox")
        self.assertAlmostEqual(result["corners"][0]["x"], .1)
        self.assertAlmostEqual(result["corners"][2]["y"], .8)

    def test_no_detection_keeps_detector_confidence_diagnostics(self):
        detector = Mock()
        detector.detect.return_value = {"status": "no_detection", "reason": "no_matching_detection",
                                        "max_candidate_confidence": .41,
                                        "candidates_considered": 2100, "inference_ms": 4.0}
        model = self.model_with_detector(detector)
        result = model.detect(np.zeros((20, 20, 3), np.uint8), proposals=[])
        self.assertEqual(result["status"], "no_detection")
        self.assertEqual(result["model_diagnostics"], {
            "max_candidate_confidence": .41, "candidates_considered": 2100,
            "inference_ms": 4.0})

    def test_status_does_not_wait_for_inference_lock(self):
        import threading
        model = NeckModel.__new__(NeckModel)
        model.detector = Mock()
        entered = threading.Event()
        release = threading.Event()
        def slow_detect(image):
            entered.set()
            release.wait(1)
            return {"status": "no_detection", "reason": "done"}
        model.detector.detect.side_effect = slow_detect
        model.confidence_threshold = .5
        model.lock = __import__('threading').Lock()
        model.last = {"status": "ready"}
        model.last_detection = None
        model.last_error = None
        model.load_ms = None
        worker = threading.Thread(target=model.detect, args=(np.zeros((20, 20, 3), np.uint8),))
        worker.start()
        self.assertTrue(entered.wait(1))
        status = model.status()
        release.set()
        worker.join(1)
        self.assertEqual(status["status"], "ready")

    def test_lattice_search_uses_color_model_only_outside_lattice_only(self):
        detector = Mock()
        detector.detect.return_value = {"status": "no_detection", "reason": "no_matching_detection"}
        detector.status.return_value = {"status": "ready"}
        worker = LatticeSearch(detector)
        gray = np.zeros((20, 30), np.uint8)
        bgr = np.zeros((20, 30, 3), np.uint8)
        with patch("vision_pc_bridge.threading.Thread") as thread, \
             patch("vision_pc_bridge.find_neck_regions_fast", return_value=[]), \
             patch("vision_pc_bridge.find_neck_regions", return_value=[]), \
             patch("vision_pc_bridge.AutoFretboardTracker"):
            self.assertTrue(worker.submit(gray, None, ("s", 1), bgr=bgr))
            thread.return_value.start.assert_called_once()
            detector.detect.assert_not_called()
            # Execute the captured target to verify the model route.
            thread.return_value.start.reset_mock()
            target = thread.call_args.kwargs["target"]
            target()
            self.assertFalse(worker.busy)
            self.assertTrue(worker.model_busy)
            # Neural fallback runs independently from the fast acquisition job.
            thread.call_args.kwargs["target"]()
        detector.detect.assert_called_once()
        np.testing.assert_array_equal(detector.detect.call_args.args[0], bgr)
        worker.stopping = True

    def test_lattice_only_skips_model_and_fast_negative_queues_neural_fallback(self):
        detector = Mock()
        detector.status.return_value = {"status": "ready"}
        worker = LatticeSearch(detector)
        gray = np.zeros((20, 30), np.uint8)
        with patch("vision_pc_bridge.threading.Thread") as thread, \
             patch("vision_pc_bridge.find_neck_regions_fast", return_value=[]), \
             patch("vision_pc_bridge.find_neck_regions", return_value=[]), \
             patch("vision_pc_bridge.AutoFretboardTracker") as factory:
            factory.return_value.update.return_value = {"status": "searching", "reason": "fallback"}
            self.assertTrue(worker.submit(gray, None, ("s", 1), lattice_only=True))
            thread.call_args.kwargs["target"]()
        detector.detect.assert_not_called()
        factory.return_value.update.assert_called_once()

        detector.detect.reset_mock()
        with patch("vision_pc_bridge.threading.Thread") as thread, \
             patch("vision_pc_bridge.find_neck_regions", return_value=[]), \
             patch("vision_pc_bridge.AutoFretboardTracker") as factory:
            factory.return_value.update.return_value = {"status": "searching", "reason": "fallback"}
            self.assertTrue(worker.submit(gray, None, ("s", 2), bgr=np.zeros((20, 30, 3), np.uint8)))
            thread.call_args.kwargs["target"]()
            self.assertFalse(worker.busy)
            self.assertTrue(worker.model_busy)
            thread.call_args.kwargs["target"]()
        detector.detect.assert_called_once()
        factory.return_value.update.assert_not_called()


if __name__ == "__main__":
    unittest.main()

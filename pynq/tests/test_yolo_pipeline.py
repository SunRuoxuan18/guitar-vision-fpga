"""Object boxes do not become a fret lattice without observed line evidence."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_fretboard_pipeline import YoloFretboardPipeline
from vision_server import parse_args
from vision_server import VisionState, create_source


class Detector:
    def __init__(self, result):
        self.result = result

    def detect(self, image):
        self.image = image
        return self.result


class Geometry:
    def __init__(self, tracking=True):
        self.tracking = tracking

    def update(self, gray):
        self.shape = gray.shape
        line = {"start": {"x": 0, "y": 0}, "end": {"x": 1, "y": 1}, "observed": True}
        return {"status": "tracking" if self.tracking else "searching", "confidence": .8,
                "corners": [{"x": x, "y": y} for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))],
                "strings": [dict(line) for _ in range(6)], "frets": [dict(line) for _ in range(8)]}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.zeros((100, 200), np.uint8)

    def test_missing_model_yields_no_geometry(self):
        pipe = YoloFretboardPipeline(Detector({"status": "not_ready", "reason": "weights_missing"}), Geometry())
        result = pipe.update(self.gray)
        self.assertEqual(result["reason"], "yolo_not_ready")
        self.assertIsNone(result["corners"])

    def test_proposal_alone_cannot_become_a_lattice(self):
        pipe = YoloFretboardPipeline(Detector({"status": "proposal", "bbox": [.2, .3, .8, .7]}), Geometry(False))
        result = pipe.update(self.gray)
        self.assertEqual(result["status"], "searching")
        self.assertEqual(result["strings"], [])
        self.assertEqual(result["proposal_bbox"], [.2, .3, .8, .7])

    def test_geometry_is_mapped_back_to_original_image(self):
        geometry = Geometry()
        pipe = YoloFretboardPipeline(Detector({"status": "proposal", "bbox": [.2, .3, .8, .7]}), geometry)
        result = pipe.update(self.gray)
        self.assertEqual(result["status"], "tracking")
        self.assertFalse(result["absolute_frets_known"])
        self.assertEqual(len(result["strings"]), 6)
        x0, y0 = result["corners"][0]["x"], result["corners"][0]["y"]
        x1, y1 = result["corners"][2]["x"], result["corners"][2]["y"]
        self.assertAlmostEqual(x0, 33 / 199)
        self.assertAlmostEqual(y0, 25 / 99)
        self.assertAlmostEqual((x1 - x0) * 199, geometry.shape[1] - 1)
        self.assertAlmostEqual((y1 - y0) * 99, geometry.shape[0] - 1)
        result["corners"][0]["x"] = 0
        self.assertNotEqual(pipe._snapshot["corners"][0]["x"], 0)

    def test_invalid_boxes_do_not_publish_a_crop(self):
        for box in ([1, 0, 0, 1], [0, 0, float("nan"), 1], [-.1, 0, 1, 1], [True, 0, 1, 1]):
            with self.subTest(box=box):
                result = YoloFretboardPipeline(Detector({"status": "proposal", "bbox": box}), Geometry()).update(self.gray)
                self.assertEqual(result["reason"], "invalid_proposal")

    def test_model_cli_requires_both_paths(self):
        with self.assertRaises(SystemExit):
            parse_args(["--synthetic", "--yolo-model", "absent.onnx"])

    def test_service_aligns_large_jpeg_with_downscaled_tracking_frame(self):
        import cv2
        detector = Detector({"status": "proposal", "bbox": [.2, .3, .8, .7]})
        state = VisionState(create_source(camera=0), clock=lambda: 10, yolo_detector=detector)
        geometry = Geometry()
        state._auto_tracker = YoloFretboardPipeline(detector, geometry)
        state._auto_generation = state._capture_generation
        bgr = np.zeros((720, 1280, 3), np.uint8)
        bgr[:, :, 2] = 200
        success, jpeg = cv2.imencode(".jpg", bgr)
        self.assertTrue(success)
        state._analyze_auto(np.zeros((360, 640), np.uint8), {"sequence": 1},
                            state._capture_generation, jpeg.tobytes())
        self.assertEqual(detector.image.shape, (360, 640, 3))
        self.assertGreater(detector.image[0, 0, 2], 190)
        self.assertEqual(state._auto_cached[1]["status"], "tracking")


if __name__ == "__main__":
    unittest.main()

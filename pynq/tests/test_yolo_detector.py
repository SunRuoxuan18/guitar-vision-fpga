"""Synthetic decoding and real tiny-graph DNN checks; no trained weights."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_detector import (LetterboxTransform, YoloDetector, decode_predictions,
                           letterbox, load_metadata, non_max_suppression)

try:
    import cv2
    import numpy as np
    HAS_CV = True
except (ImportError, OSError):
    HAS_CV = False
try:
    import onnx
    HAS_ONNX = True
except (ImportError, OSError):
    HAS_ONNX = False


def metadata(layout="BCN", size=(320, 320)):
    return {"schema_version": 1, "model_family": "yolov8", "task": "detect",
            "classes": ["fretboard", "fingertip"], "input_size": list(size),
            "output_layout": layout, "box_format": "cxcywh",
            "coordinate_units": "input_pixels", "color_order": "RGB",
            "has_objectness": False, "embedded_nms": False}


class ReadinessTests(unittest.TestCase):
    def test_missing_weights_is_not_ready_and_json_safe(self):
        detector = YoloDetector()
        result = detector.detect(None)
        self.assertEqual(result["status"], "not_ready")
        self.assertEqual(result["reason"], "weights_missing")
        self.assertIsNone(result["bbox"])
        self.assertEqual(result["confidence"], 0)
        self.assertEqual(result["detections"], [])
        json.dumps(result, allow_nan=False)
        result["detections"].append("mutated")
        self.assertEqual(detector.snapshot()["detections"], [])

    def test_import_and_missing_model_never_import_dependencies(self):
        code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cv2', 'numpy', 'torch', 'ultralytics', 'onnx'):
            raise AssertionError('eager import: ' + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
from yolo_detector import YoloDetector
assert YoloDetector().detect(None)['reason'] == 'weights_missing'
"""
        process = subprocess.run([sys.executable, "-I", "-B", "-c", code,
                                  str(Path(__file__).resolve().parents[1])],
                                 capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_missing_invalid_metadata_and_corrupt_model_are_not_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "broken.onnx"
            model.write_bytes(b"not an ONNX model")
            self.assertEqual(YoloDetector(model).status()["reason"], "metadata_missing")
            self.assertEqual(YoloDetector(model, {}).status()["reason"], "invalid_metadata")
            self.assertEqual(YoloDetector(model, Path(directory) / "missing.json").status()["reason"],
                             "invalid_metadata")
            if HAS_CV:
                self.assertEqual(YoloDetector(model, metadata()).status()["reason"], "model_load_failed")

    def test_options_reject_nan_infinity_and_boolean(self):
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                YoloDetector(confidence_threshold=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                YoloDetector(iou_threshold=value)
        for value in (True, 0, -1, 2.5):
            with self.assertRaises(ValueError):
                YoloDetector(max_detections=value)

    def test_metadata_requires_explicit_supported_contract(self):
        self.assertEqual(load_metadata(metadata())["classes"], ["fretboard", "fingertip"])
        for key, value in (("classes", []), ("classes", ["x", "x"]), ("classes", {0: "x"}),
                           ("classes", [" fretboard"]), ("schema_version", True),
                           ("model_family", "yolov5"), ("task", "segment"),
                           ("input_size", [321, 320]), ("input_size", [True, 320]),
                           ("output_layout", "auto"), ("has_objectness", True),
                           ("has_objectness", 0), ("embedded_nms", True),
                           ("coordinate_units", "normalized"), ("color_order", "BGR"),
                           ("onnx_sha256", "no")):
            data = metadata()
            data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                load_metadata(data)
        for key in metadata():
            data = metadata()
            del data[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                load_metadata(data)

    def test_cli_missing_weights_has_json_exit_two(self):
        cli = Path(__file__).resolve().parents[1] / "yolo_cli.py"
        process = subprocess.run([sys.executable, "-B", str(cli), "status"],
                                 capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 2, process.stderr)
        self.assertEqual(json.loads(process.stdout)["reason"], "weights_missing")


@unittest.skipUnless(HAS_CV, "OpenCV/numpy required")
class DecodeTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)
        _, self.transform = letterbox(self.image, (320, 320))
        self.rows = np.float32([
            [160, 160, 160, 80, .9, .1],
            [160, 160, 150, 70, .8, .2],
            [160, 160, 160, 80, .1, .85],
            [35, 110, 10, 10, .4, .2]])

    def decode(self, rows=None, layout="BCN", **kwargs):
        rows = self.rows if rows is None else rows
        output = rows.T[None] if layout == "BCN" else rows[None]
        return decode_predictions(output, metadata(layout), self.transform, **kwargs)

    def test_both_layouts_2d_and_single_output_list_are_equivalent(self):
        expected = self.decode()
        self.assertEqual(expected, self.decode(layout="BNC"))
        for output in (self.rows.T, [self.rows.T[None]], (self.rows.T[None],)):
            self.assertEqual(expected, decode_predictions(output, metadata(), self.transform))

    def test_remap_to_original_normalized_and_class_aware_nms(self):
        detections = self.decode()
        self.assertEqual(len(detections), 3)
        np.testing.assert_allclose(detections[0]["bbox"], [.25, .25, .75, .75])
        self.assertEqual([item["class_name"] for item in detections], ["fretboard", "fingertip", "fretboard"])
        self.assertNotIn("strings", detections[0])
        self.assertNotIn("corners", detections[0])
        json.dumps(detections, allow_nan=False)

    def test_odd_rounding_padding_and_portrait(self):
        for shape, size in (((101, 203, 3), (320, 320)), ((203, 101, 3), (320, 640))):
            image = np.zeros(shape, dtype=np.uint8)
            padded, transform = letterbox(image, size)
            rw, rh = transform.resized_size
            row = np.float32([[transform.pad_left + rw * .5, transform.pad_top + rh * .5,
                               rw * .8, rh * .6, .9, .1]])
            result = decode_predictions(row.T[None], metadata(size=size), transform)
            np.testing.assert_allclose(result[0]["bbox"], [.1, .2, .9, .8], atol=1e-6)
            self.assertEqual(padded.shape, (size[1], size[0], 3))
            self.assertEqual(transform.original_size, (shape[1], shape[0]))

    def test_padding_clipping_degenerate_and_empty_outputs(self):
        rows = np.float32([[160, 160, 640, 640, .9, .1], [20, 20, 10, 10, .8, .2],
                           [160, 160, 0, 10, .7, .1]])
        result = self.decode(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["bbox"], [0, 0, 1, 1])
        self.assertEqual(self.decode(np.empty((0, 6), dtype=np.float32)), [])
        self.assertEqual(self.decode(confidence_threshold=.99), [])

    def test_class_filters_and_limits(self):
        for filters in (["fingertip"], [1]):
            result = self.decode(class_filter=filters)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["class_id"], 1)
        # A lower-scoring allowed class does not relabel a disallowed winner.
        self.assertEqual(self.decode(self.rows[:1], class_filter=[1]), [])
        self.assertEqual(len(self.decode(max_detections=1)), 1)
        self.assertEqual(len(self.decode(max_candidates=1)), 1)
        for value in ([], ["guitar"], [3], [-1], [True], "fretboard"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.decode(class_filter=value)

    def test_nms_deterministic_ties_and_cross_class_overlap(self):
        boxes = np.float64([[.1, .1, .8, .8], [.1, .1, .8, .8], [.1, .1, .8, .8], [0, 0, .05, .05]])
        self.assertEqual(non_max_suppression(boxes, [.8, .8, .8, .7], [0, 0, 1, 0]), [0, 2, 3])
        self.assertEqual(non_max_suppression(boxes, [.8, .8, .8, .7], [0, 0, 1, 0],
                                             iou_threshold=1), [0, 1, 2, 3])
        for bad_boxes, scores, classes in (([[0, 0, float("nan"), 1]], [.5], [0]),
                                           ([[0, 0, 0, 1]], [.5], [0]),
                                           ([[-1, 0, 1, 1]], [.5], [0]),
                                           ([[0, 0, 1, 1]], [float("inf")], [0]),
                                           ([[0, 0, 1, 1]], [.5], [.1])):
            with self.assertRaises(ValueError):
                non_max_suppression(bad_boxes, scores, classes)

    def test_nonfinite_probability_negative_size_and_overflow_rejected(self):
        for column, value in ((0, np.nan), (1, np.inf), (4, np.nan), (5, -np.inf),
                              (4, 1.1), (4, -.1), (2, -1), (3, -1)):
            rows = self.rows.copy()
            rows[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                self.decode(rows)
        rows = self.rows.astype(np.float64)
        rows[0, 0] = np.finfo(np.float64).max
        rows[0, 2] = np.finfo(np.float64).max
        with self.assertRaisesRegex(ValueError, "overflowed"):
            self.decode(rows)

    def test_incompatible_heads_batches_and_layouts_rejected(self):
        for output in (np.zeros((2, 6, 4)), np.zeros((1, 7, 4)), np.zeros((1, 4, 6)),
                       np.zeros((1, 6, 4, 1)), np.zeros(6), np.full((1, 6, 4), "bad"),
                       [self.rows.T[None], self.rows.T[None]]):
            with self.subTest(shape=getattr(output, "shape", None)), self.assertRaises(ValueError):
                decode_predictions(output, metadata(), self.transform)
        with self.assertRaises(ValueError):
            decode_predictions(self.rows.T[None], metadata(size=(640, 640)), self.transform)

    def test_bad_images_and_transforms_rejected(self):
        for image in (None, [], np.zeros((0, 20, 3), np.uint8), np.zeros((20, 20), np.uint8),
                      np.zeros((20, 20, 4), np.uint8), np.zeros((20, 20, 3), np.float32)):
            with self.assertRaises(ValueError):
                letterbox(image, (320, 320))
        with self.assertRaises(ValueError):
            LetterboxTransform((100, 100), (320, 320), (320, 320), 1, 0)


def build_tiny_onnx(path, layout="BCN"):
    """A real Conv+pool graph, with synthetic boxes and an RGB-dependent score."""
    from onnx import TensorProto, helper, numpy_helper

    rows = np.float32([[16, 16, 16, 8, .70, .10],
                       [16, 16, 14, 6, .60, .10],
                       [8, 8, 4, 4, .10, .85]])
    output = rows.T[None] if layout == "BCN" else rows[None]
    weights = np.zeros((18, 3, 1, 1), dtype=np.float32)
    score_index = 12 if layout == "BCN" else 4
    weights[score_index, 0, 0, 0] = .1
    graph = helper.make_graph([
        helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
        helper.make_node("Conv", ["pooled", "weights", "bias"], ["values"], kernel_shape=[1, 1]),
        helper.make_node("Reshape", ["values", "shape"], ["predictions"])],
        "synthetic_yolo_head_not_trained",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, list(output.shape))],
        initializer=[numpy_helper.from_array(weights, "weights"),
                     numpy_helper.from_array(output.reshape(-1), "bias"),
                     numpy_helper.from_array(np.array(output.shape, dtype=np.int64), "shape")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@unittest.skipUnless(HAS_CV and HAS_ONNX, "real inference test needs optional ONNX authoring package")
class RealOnnxTests(unittest.TestCase):
    def test_forward_both_layouts_rgb_normalization_nms_remap_and_error_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            for layout in ("BCN", "BNC"):
                with self.subTest(layout=layout):
                    path = Path(directory) / (layout + ".onnx")
                    build_tiny_onnx(path, layout)
                    detector = YoloDetector(path, metadata(layout, (32, 32)))
                    self.assertEqual(detector.status()["status"], "ready")
                    image = np.zeros((32, 64, 3), dtype=np.uint8)
                    result = detector.detect(image)
                    self.assertEqual(result["status"], "proposal", result)
                    self.assertEqual(result["class_name"], "fingertip")
                    self.assertEqual(len(result["detections"]), 2)
                    board = next(item for item in result["detections"] if item["class_name"] == "fretboard")
                    np.testing.assert_allclose(board["bbox"], [.25, .25, .75, .75], atol=1e-6)
                    expected = .70 + .1 * (114 / 255) * .5
                    self.assertAlmostEqual(board["confidence"], expected, places=5)
                    image[:, :, 2] = 255
                    red = detector.detect(image)
                    red_score = next(item["confidence"] for item in red["detections"]
                                     if item["class_name"] == "fretboard")
                    self.assertAlmostEqual(red_score - board["confidence"], .05, places=5)
                    self.assertGreater(result["inference_ms"], 0)
                    self.assertEqual(result["image_size"], [64, 32])
                    self.assertEqual(detector.detect(None)["status"], "error")
                    self.assertIsNone(detector.snapshot()["bbox"])
                    self.assertEqual(detector.detect(image)["status"], "proposal")
                    json.dumps(result, allow_nan=False)

    def test_checksum_layout_filter_mismatch_and_cli_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "tiny.onnx"
            build_tiny_onnx(path)
            data = metadata(size=(32, 32))
            data["onnx_sha256"] = "0" * 64
            self.assertEqual(YoloDetector(path, data).status()["reason"], "weights_checksum_mismatch")
            data.pop("onnx_sha256")
            image = np.zeros((32, 64, 3), np.uint8)
            self.assertEqual(YoloDetector(path, data, class_filter=["guitar"]).status()["reason"], "invalid_metadata")
            mismatch = metadata("BNC", (32, 32))
            self.assertEqual(YoloDetector(path, mismatch).detect(image)["status"], "error")
            no_detection = YoloDetector(path, data, confidence_threshold=.99).detect(image)
            self.assertEqual(no_detection["status"], "no_detection")
            self.assertIsNone(no_detection["bbox"])
            metadata_path = directory / "tiny.json"
            metadata_path.write_text(json.dumps(data), encoding="utf-8")
            image_path = directory / "image.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            cli = Path(__file__).resolve().parents[1] / "yolo_cli.py"
            for command in ("image", "benchmark"):
                process = subprocess.run([sys.executable, "-B", str(cli), command,
                                          "--weights", str(path), "--metadata", str(metadata_path),
                                          "--image", str(image_path), "--iterations", "3", "--warmup", "1"],
                                         capture_output=True, text=True, timeout=30)
                self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
                result = json.loads(process.stdout)
                self.assertEqual(result["status"], "proposal")
                if command == "benchmark":
                    self.assertEqual(result["benchmark"]["iterations"], 3)
                    self.assertFalse(result["benchmark"]["accuracy_measured"])
                    self.assertFalse(result["benchmark"]["pl_acceleration"])


if __name__ == "__main__":
    unittest.main()

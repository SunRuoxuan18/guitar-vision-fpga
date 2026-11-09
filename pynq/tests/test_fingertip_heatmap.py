"""Focused fingertip wrapper contracts; no board, overlay, or trained weights."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fingertip_heatmap import FingertipHeatmap, POINT_NAMES, decode_heatmaps


class FakeGraph:
    def __init__(self, outputs=None):
        self.outputs = [np.zeros((1, 4, 26, 26), np.float32)] if outputs is None else outputs
        self.calls = 0
        self.input = None

    def setInput(self, blob):
        self.input = blob

    def forward(self):
        self.calls += 1
        return self.outputs


class DecodeTests(unittest.TestCase):
    def test_raw_peaks_channel_order_mapping_and_no_half_pixel(self):
        heatmaps = np.full((1, 4, 26, 26), -20.0, np.float32)
        locations = [(0, 0), (25, 25), (7, 13), (19, 2)]
        scores = [-2.0, 9.0, 0.0, 2.5]
        for channel, ((x, y), score) in enumerate(zip(locations, scores)):
            heatmaps[0, channel, y, x] = score
        coords, peaks = decode_heatmaps(heatmaps, (10, 20, 260, 130))
        np.testing.assert_array_equal(coords, [[10, 20], [260, 145], [80, 85], [200, 30]])
        np.testing.assert_array_equal(peaks, scores)
        unbatched = decode_heatmaps(heatmaps[0], {"x": 10, "y": 20, "width": 260, "height": 130})
        np.testing.assert_array_equal(unbatched[0], coords)
        self.assertEqual(POINT_NAMES, ("index", "middle", "ring", "little"))

    def test_ties_use_first_row_major_peak_and_flat_maps_are_not_visibility(self):
        maps = np.full((4, 26, 26), -3, np.float32)
        maps[0, 2, 4] = maps[0, 3, 1] = 5
        coords, peaks = decode_heatmaps(maps, (0, 0, 26, 26))
        np.testing.assert_array_equal(coords, [[4, 2], [0, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(peaks, [5, -3, -3, -3])

    def test_reject_shapes_dtypes_and_any_nonfinite_value(self):
        for shape in ((1, 21, 26, 26), (2, 4, 26, 26), (1, 4, 32, 32), (4, 676), (0,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                decode_heatmaps(np.zeros(shape, np.float32), (0, 0, 26, 26))
        for dtype in (np.int32, np.complex64, object, bool):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                decode_heatmaps(np.zeros((4, 26, 26), dtype), (0, 0, 26, 26))
        for value in (np.nan, np.inf, -np.inf):
            maps = np.zeros((4, 26, 26), np.float32)
            maps[3, 25, 25] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_heatmaps(maps, (0, 0, 26, 26))


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.graph = FakeGraph()
        self.wrapper = FingertipHeatmap(graph=self.graph)
        self.frame = np.zeros((220, 230, 3), np.uint8)
        self.roi = (10, 5, 208, 208)

    def test_bgr_rgb_normalization_contiguous_nchw_and_result(self):
        self.frame[:] = [11, 57, 201]
        self.graph.outputs[0][0, :, 13, 7] = [-2, 3, 4, 9]
        self.graph.outputs[0][0, 0, :, :] = -9
        self.graph.outputs[0][0, 0, 13, 7] = -2
        result = self.wrapper.infer_bgr(self.frame, self.roi)
        expected = (np.float32([201, 57, 11]) - np.float32([123.675, 116.28, 103.53])) / np.float32([58.395, 57.12, 57.375])
        np.testing.assert_array_equal(self.graph.input[0, :, 0, 0], expected)
        self.assertEqual(self.graph.input.shape, (1, 3, 208, 208))
        self.assertEqual(self.graph.input.dtype, np.float32)
        self.assertTrue(self.graph.input.flags.c_contiguous)
        self.assertEqual(result["roi"], self.roi)
        self.assertEqual(tuple(result["points"]), POINT_NAMES)
        self.assertEqual(result["points"]["index"], {"x": 66.0, "y": 109.0, "raw_score": -2.0})
        self.assertEqual(set(result["timings_ms"]), {"preprocess", "graph", "decode", "total"})
        self.assertTrue(all(value >= 0 for value in result["timings_ms"].values()))
        json.dumps(result, allow_nan=False)
        self.assertEqual(self.graph.calls, 1)

    def test_rgba_matches_bgr_and_ignores_alpha(self):
        rng = np.random.default_rng(4)
        bgr = rng.integers(0, 256, self.frame.shape, dtype=np.uint8)
        rgba = np.concatenate((bgr[:, :, ::-1], np.zeros(bgr.shape[:2] + (1,), np.uint8)), axis=2)
        self.wrapper.infer_bgr(bgr, self.roi)
        expected = self.graph.input.copy()
        self.wrapper.infer_rgba(rgba, self.roi)
        np.testing.assert_array_equal(self.graph.input, expected)
        rgba[:, :, 3] = 255
        self.wrapper.infer_rgba(rgba, self.roi)
        np.testing.assert_array_equal(self.graph.input, expected)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for resize parity")
    def test_fractional_roi_and_resize_match_training_preprocessing(self):
        import cv2
        rng = np.random.default_rng(8)
        frame = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
        roi = {"x": 13.25, "y": 7.75, "width": 260.5, "height": 218.5}
        self.graph.outputs[0][0, 1, 25, 25] = 5
        result = self.wrapper.infer_bgr(frame, roi)
        crop = cv2.resize(frame[7:227, 13:274], (208, 208), interpolation=cv2.INTER_AREA)
        rgb = crop[:, :, ::-1].astype(np.float32)
        expected = (rgb - np.float32([123.675, 116.28, 103.53])) / np.float32([58.395, 57.12, 57.375])
        np.testing.assert_array_equal(self.graph.input, expected.transpose(2, 0, 1)[None])
        self.assertEqual(result["roi"], (13, 7, 261, 220))
        self.assertEqual(result["points"]["index"]["x"], 13)
        self.assertEqual(result["points"]["index"]["y"], 7)
        self.assertAlmostEqual(result["points"]["middle"]["x"], 13 + 25 / 26 * 261)
        self.assertAlmostEqual(result["points"]["middle"]["y"], 7 + 25 / 26 * 220)
        rgba = np.concatenate((frame[:, :, ::-1], np.full((300, 400, 1), 97, np.uint8)), axis=2)
        self.wrapper.infer_rgba(rgba, roi)
        np.testing.assert_array_equal(self.graph.input, expected.transpose(2, 0, 1)[None])

    def test_invalid_rois_never_execute_graph(self):
        for roi in (None, (), (0, 0, 1), (0, 0, 1, 2, 3), {},
                    (0, 0, 0, 1), (0, 0, 1, -1), (-1, 0, 20, 20),
                    (229, 0, 2, 1), (0, 220, 1, 1), (0, 0, np.nan, 1),
                    (np.inf, 0, 1, 1), (0, 0, 1, -np.inf),
                    (True, 0, 1, 1), (0, 0, "20", 20), (0, 0, 1e308, 1)):
            with self.subTest(roi=roi), self.assertRaises(ValueError):
                self.wrapper.infer_bgr(self.frame, roi)
        self.assertEqual(self.graph.calls, 0)
        self.assertIsNone(self.graph.input)

    def test_invalid_frames_and_explicit_color_api_never_execute_graph(self):
        for frame in (None, [], np.zeros((208, 208), np.uint8),
                      np.zeros((0, 208, 3), np.uint8), np.zeros((208, 208, 4), np.uint8),
                      np.zeros((208, 208, 3), np.float32), np.full((208, 208, 3), np.nan)):
            with self.subTest(shape=np.shape(frame)), self.assertRaises(ValueError):
                self.wrapper.infer_bgr(frame, (0, 0, 208, 208))
        with self.assertRaises(ValueError):
            self.wrapper.infer_rgba(self.frame, self.roi)
        self.assertEqual(self.graph.calls, 0)

    def test_output_contract_and_nonfinite_fail_without_stale_result(self):
        self.wrapper.infer_bgr(self.frame, self.roi)
        for outputs in ([], [np.zeros((4, 26, 26), np.float32)],
                        [np.zeros((1, 4, 26, 26), np.float32)] * 2,
                        np.zeros((1, 4, 26, 26), np.float32),
                        [np.full((1, 4, 26, 26), np.inf, np.float32)]):
            self.graph.outputs = outputs
            with self.subTest(output_type=type(outputs)), self.assertRaises(ValueError):
                self.wrapper.infer_bgr(self.frame, self.roi)

    def test_engine_errors_propagate_without_cpu_fallback(self):
        with mock.patch.object(self.graph, "forward", side_effect=RuntimeError("PL failure")):
            with self.assertRaisesRegex(RuntimeError, "PL failure"):
                self.wrapper.infer_bgr(self.frame, self.roi)

    def test_constructor_requires_unambiguous_graph_or_engine(self):
        for kwargs in ({}, {"directory": "unused"}, {"graph": object()},
                       {"graph": self.graph, "engine": object()},
                       {"graph": self.graph, "model_path": "unused"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                FingertipHeatmap(**kwargs)

    def test_production_constructs_hand_graph_with_passed_engine_and_model(self):
        from types import SimpleNamespace
        engine = SimpleNamespace(convolve=lambda *args: None)
        self.graph.nodes = [{"op": "PLConvolution"}, {"op": "Clip"}, {"op": "Add"}]
        self.graph.outputs = ["fingertip_heatmaps"]
        with mock.patch("hand_graph.HandGraph", return_value=self.graph) as constructor:
            wrapper = FingertipHeatmap("compiled", engine, model_path="selected.onnx")
            constructor.assert_called_once_with("compiled", engine, model_path="selected.onnx")
            self.assertIs(wrapper.graph, self.graph)
            self.graph.nodes.append({"op": "Sigmoid"})
            with self.assertRaises(ValueError):
                FingertipHeatmap("compiled", engine)
            self.graph.nodes = [{"op": "PLConvolution"}]
            self.graph.outputs = ["other"]
            with self.assertRaises(ValueError):
                FingertipHeatmap("compiled", engine)

    def test_supplied_graph_can_infer_without_opencv_or_pynq_imports(self):
        code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cv2', 'pynq', 'torch', 'onnxruntime', 'hand_graph'):
            raise AssertionError('Unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
import numpy as np
from fingertip_heatmap import FingertipHeatmap
class Graph:
    def setInput(self, blob):
        assert blob.shape == (1, 3, 208, 208)
    def forward(self):
        return [np.zeros((1, 4, 26, 26), np.float32)]
wrapper = FingertipHeatmap(graph=Graph())
result = wrapper.infer_bgr(np.zeros((208, 208, 3), np.uint8), (0, 0, 208, 208))
assert len(result['points']) == 4
"""
        process = subprocess.run([sys.executable, "-I", "-B", "-c", code,
                                  str(Path(__file__).resolve().parents[1])],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()

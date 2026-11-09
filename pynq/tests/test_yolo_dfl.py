"""Tests for the dependency-light raw YOLOv8 DFL decoder."""

from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_dfl import decode_dfl

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cv2
    import onnx
    import onnxruntime
    HAS_ONNX_RUNTIME = HAS_NUMPY
except (ImportError, OSError):
    HAS_ONNX_RUNTIME = False


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for DFL decoding")
class DflDecoderTests(unittest.TestCase):
    def raw(self, input_size=32, classes_count=2, reg_max=16, strides=(8, 16, 32)):
        count = sum((input_size // stride) ** 2 for stride in strides)
        return np.zeros((1, 4 * reg_max + classes_count, count), dtype=np.float32)

    def test_dfl_ltrb_anchors_and_sigmoid_classes(self):
        raw = np.full((1, 66, 21), -100.0, dtype=np.float32)
        bins = (1, 2, 3, 4)
        for side, bin_index in enumerate(bins):
            raw[0, side * 16 + bin_index, :] = 100.0
        raw[0, 64, :] = 0.0
        raw[0, 65, :] = np.log(3.0)

        decoded = decode_dfl(raw, 32, 2)

        self.assertEqual(decoded.shape, (1, 6, 21))
        # Candidate zero is the stride-8 anchor at (0.5, 0.5), or (4, 4) px.
        np.testing.assert_allclose(decoded[0, :4, 0], (12.0, 12.0, 32.0, 48.0), atol=1e-12)
        np.testing.assert_allclose(decoded[0, 4:, 0], (0.5, 0.75), atol=1e-12)
        # Candidate 16 is the first stride-16 anchor, proving scale concatenation order.
        np.testing.assert_allclose(decoded[0, :4, 16], (24.0, 24.0, 64.0, 96.0), atol=1e-12)

    def test_strict_input_validation(self):
        raw = self.raw()
        invalid_outputs = (
            raw[0], np.zeros((2, 66, 21), np.float32), np.zeros((1, 65, 21), np.float32),
            np.zeros((1, 66, 20), np.float32), np.zeros((1, 66, 21, 1), np.float32),
            np.full((1, 66, 21), "invalid"),
        )
        for output in invalid_outputs:
            with self.subTest(shape=getattr(output, "shape", None)), self.assertRaises(ValueError):
                decode_dfl(output, 32, 2)
        for value in (np.nan, np.inf, -np.inf):
            invalid = raw.copy()
            invalid[0, 0, 0] = value
            with self.subTest(nonfinite=value), self.assertRaises(ValueError):
                decode_dfl(invalid, 32, 2)
        for input_size in (True, 0, -1, 32.0, (32, 32, 32), [32], [30, 32],
                           [32, 30], ("32", 32)):
            with self.subTest(input_size=input_size), self.assertRaises(ValueError):
                decode_dfl(raw, input_size, 2)
        for classes_count, reg_max, strides in ((True, 16, (8, 16, 32)), (0, 16, (8, 16, 32)),
                                                (2, True, (8, 16, 32)), (2, 0, (8, 16, 32)),
                                                (2, 16, ()), (2, 16, (8, 16, 0)),
                                                (2, 16, (8, 8, 32)), (2, 16, (8, 16, True))):
            with self.subTest(classes_count=classes_count, reg_max=reg_max, strides=strides), self.assertRaises(ValueError):
                decode_dfl(raw, 32, classes_count, reg_max, strides)

    def test_rectangular_input_uses_width_height_grid_order(self):
        input_size = [32, 16]
        raw = np.full((1, 66, 10), -100.0, dtype=np.float32)
        for side in range(4):
            raw[0, side * 16, :] = 100.0
        decoded = decode_dfl(raw, input_size, 2, strides=(8, 16))

        self.assertEqual(decoded.shape, (1, 6, 10))
        self.assertEqual(decoded.dtype, np.float32)
        # 4x2 stride-8 cells, then the first 2x1 stride-16 cell.
        np.testing.assert_allclose(decoded[0, :4, 8], (8.0, 8.0, 0.0, 0.0), atol=1e-5)
        np.testing.assert_allclose(decoded[0, :4, 9], (24.0, 8.0, 0.0, 0.0), atol=1e-5)


@unittest.skipUnless(HAS_ONNX_RUNTIME, "real equivalence test requires OpenCV, ONNX, and ONNX Runtime")
class RealModelEquivalenceTests(unittest.TestCase):
    def test_raw_head_matches_full_neck320_output(self):
        root = Path(__file__).resolve().parents[2]
        model_path = root / "models" / "neck" / "candidates" / "neck320.onnx"
        image_path = root / "diagnostics" / "current_neck_frame.jpg"
        self.assertTrue(model_path.is_file())
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_tensor = np.transpose(image, (2, 0, 1))[None]

        raw_name = "/model.22/Concat_3_output_0"
        model = onnx.load(str(model_path))
        model.graph.output.append(onnx.helper.make_tensor_value_info(
            raw_name, onnx.TensorProto.FLOAT, [1, 66, 2100]))
        with tempfile.TemporaryDirectory() as directory:
            cropped_path = Path(directory) / "neck320_with_raw_head.onnx"
            onnx.save(model, str(cropped_path))
            session = onnxruntime.InferenceSession(
                str(cropped_path), providers=["CPUExecutionProvider"])
            full, raw = session.run(["output0", raw_name], {"images": input_tensor})

        self.assertEqual(raw.shape, (1, 66, 2100))
        self.assertEqual(full.shape, (1, 6, 2100))
        np.testing.assert_allclose(decode_dfl(raw, 320, 2), full, rtol=1e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()

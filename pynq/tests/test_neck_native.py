"""Native neck proposal equivalence. Build libneck_region.so on the PYNQ board."""
import ctypes
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import neck_region


def _native_library_path():
    configured = os.environ.get("FRETLENS_NECK_NATIVE_LIBRARY")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "libneck_region.so"


@unittest.skipUnless(_native_library_path().is_file(), "native neck library is not built")
class NativeNeckEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = ctypes.CDLL(str(_native_library_path()))
        array = np.ctypeslib.ndpointer
        cls.function = cls.library.neck_border_candidates
        cls.function.argtypes = [array(np.float32, flags="C_CONTIGUOUS"), ctypes.c_int,
                                 ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                 array(np.float32, flags="C_CONTIGUOUS")]
        cls.function.restype = ctypes.c_int

    @staticmethod
    def _segments_fixture():
        image = np.full((480, 640), 35, np.uint8)
        for y in (190, 260):
            cv2.line(image, (60, y), (340, y), 230, 2)
        for x in (65, 85, 107, 132, 160, 191, 226, 265, 309, 358, 413, 475):
            cv2.line(image, (x, 185), (x, 265), 180, 2)
        edges = cv2.Canny(cv2.GaussianBlur(image, (5, 5), 1.0), 30, 80)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 30, minLineLength=22, maxLineGap=9)
        return image, np.ascontiguousarray(lines[:, 0].astype(np.float32).reshape(-1, 4))

    def test_fixture_matches_python_geometry(self):
        image, segments = self._segments_fixture()
        lines = segments.reshape(-1, 1, 4).astype(np.int32)
        with patch.object(neck_region.cv2, "HoughLinesP", return_value=lines):
            expected = neck_region._find_neck_regions(image, 6, np.empty((1, 1), np.uint8),
                                                       require_frets=True)
        output = np.full(6 * 8, np.nan, np.float32)
        count = self.function(segments, len(segments), image.shape[1], image.shape[0], 6, 1, output)
        actual = output[:count * 8].reshape(count, 4, 2)
        expected_array = np.array([[[point["x"], point["y"]] for point in quad]
                                   for quad in expected], np.float32)
        self.assertEqual(count, len(expected))
        np.testing.assert_allclose(actual, expected_array, rtol=0, atol=2e-5)

    def test_rejects_invalid_bounds_without_writing(self):
        segments = np.zeros((1, 4), np.float32)
        output = np.full(48, -1, np.float32)
        for args in ((4097, 640, 480, 6, 1), (1, 1, 480, 6, 1), (1, 640, 1, 6, 1),
                     (1, 640, 480, 0, 1), (1, 640, 480, 7, 1), (1, 640, 480, 6, 2)):
            with self.subTest(args=args):
                self.assertEqual(self.function(segments, *args, output), 0)
                np.testing.assert_array_equal(output, np.full(48, -1, np.float32))


if __name__ == "__main__":
    unittest.main()

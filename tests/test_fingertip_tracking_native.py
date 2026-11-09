"""Host contracts plus compiled-C checks; no board or tracker imports.

Use FINGERTIP_PATCH_QUALITY_LIBRARY for a prebuilt library, or compile with
CC/gcc/clang/cc into a temporary directory. Native cases skip without a compiler.
Run: python -B -m unittest discover -s tests -p test_fingertip_tracking_native.py
"""

import ctypes
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pynq"))
from fingertip_tracking_native import NativePatchQuality


def inputs(count=4):
    rng = np.random.default_rng(2710)
    patches = [rng.uniform(0, 255, (count, 11, 11)).astype(np.float32) for _ in range(3)]
    points = rng.uniform(10, 190, (count, 2)).astype(np.float32)
    returned = points + rng.uniform(-2, 2, points.shape).astype(np.float32)
    return patches + [points, returned]


def reference(arrays, dtype=np.float64):
    previous, current, anchors, original, returned = [a.astype(dtype) for a in arrays]
    def correlation(a, b):
        a, b = a - a.mean(), b - b.mean()
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.sum(a * b) / norm) if norm > 0 else -1.
    return np.array([[np.linalg.norm(q - p), correlation(old, new),
                      correlation(anchor, new), new.std(), old.std()]
                     for old, new, anchor, p, q in
                     zip(previous, current, anchors, original, returned)], dtype=np.float64)


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.library = mock.Mock()
        self.library.fingertip_patch_quality.return_value = 0
        with mock.patch("fingertip_tracking_native.ctypes.CDLL", return_value=self.library):
            self.quality = NativePatchQuality("test-library")

    def test_default_path_uses_cdll(self):
        with mock.patch("fingertip_tracking_native.ctypes.CDLL", return_value=self.library) as load:
            NativePatchQuality()
        self.assertEqual(Path(load.call_args.args[0]), ROOT / "pynq/libfingertip_patch_quality.so")

    def test_one_call_with_original_pointers_and_five_double_columns(self):
        arrays = inputs()
        for array in arrays:
            array.flags.writeable = False
        def fill(*args):
            dst = ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_double))
            for i in range(args[-2] * 5):
                dst[i] = i + .25
            return 0
        function = self.library.fingertip_patch_quality
        function.side_effect = fill
        output = self.quality.evaluate(*arrays)
        function.assert_called_once()
        self.assertEqual(function.call_args.args[:5], tuple(a.ctypes.data for a in arrays))
        self.assertEqual(function.call_args.args[5], 4)
        self.assertEqual(output.dtype, np.float64)
        np.testing.assert_array_equal(output, np.arange(20).reshape(4, 5) + .25)

    def test_bad_shapes_counts_dtype_layout_and_alignment_reject_before_call(self):
        cases = [inputs(0), inputs(5)]
        for index in range(5):
            for replacement in (inputs()[index].astype(np.float64),
                                inputs()[index].astype(">f4"),
                                inputs()[index][..., ::-1],
                                inputs()[index].tolist(),
                                inputs()[index][0]):
                arrays = inputs()
                arrays[index] = replacement
                cases.append(arrays)
        for index in range(3):
            arrays = inputs()
            arrays[index] = np.zeros((4, 10, 11), np.float32)
            cases.append(arrays)
        arrays = inputs()
        arrays[3] = np.zeros((4, 1, 2), np.float32)
        cases.append(arrays)
        arrays = inputs()
        arrays[0] = np.ndarray((4, 11, 11), np.float32, buffer=bytearray(4 * 121 * 4 + 1), offset=1)
        cases.append(arrays)
        for arrays in cases:
            with self.subTest(shapes=[getattr(a, "shape", None) for a in arrays]):
                with self.assertRaises(ValueError):
                    self.quality.evaluate(*arrays)
        self.library.fingertip_patch_quality.assert_not_called()

    def test_native_invalid_status_raises(self):
        self.library.fingertip_patch_quality.return_value = -1
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.quality.evaluate(*inputs())


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        library = os.environ.get("FINGERTIP_PATCH_QUALITY_LIBRARY")
        cls.directory = None
        if not library:
            configured = os.environ.get("CC")
            compiler = shlex.split(configured) if configured else next(
                ([path] for name in ("gcc", "clang", "cc") if (path := shutil.which(name))), None)
            if not compiler:
                raise unittest.SkipTest("No C compiler; run native cases on board or set FINGERTIP_PATCH_QUALITY_LIBRARY")
            cls.directory = tempfile.TemporaryDirectory(prefix="fingertip-patch-quality-")
            cls.addClassCleanup(cls.directory.cleanup)
            library = Path(cls.directory.name) / ("quality.dll" if os.name == "nt" else "quality.so")
            command = compiler + ["-O3", "-shared", "-fPIC", "-std=c99",
                str(ROOT / "pynq/native/fingertip_patch_quality.c"), "-lm", "-o", str(library)]
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        cls.quality = NativePatchQuality(library)
        if cls.directory is not None and os.name == "nt":
            cls.addClassCleanup(cls.unload_windows_library)

    @classmethod
    def unload_windows_library(cls):
        import _ctypes
        _ctypes.FreeLibrary(cls.quality.library._handle)

    def assert_reference(self, arrays):
        before = [a.copy() for a in arrays]
        actual = self.quality.evaluate(*arrays)
        self.assertEqual(actual.shape, (len(arrays[0]), 5))
        np.testing.assert_allclose(actual, reference(arrays), rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(actual, reference(arrays, np.float32), rtol=2e-6, atol=2e-6)
        for original, after in zip(before, arrays):
            np.testing.assert_array_equal(original, after)

    def test_fractional_random_batches_one_to_four(self):
        for count in range(1, 5):
            with self.subTest(count=count):
                self.assert_reference(inputs(count))

    def test_correlated_and_anticorrelated_fractional_patches(self):
        arrays = inputs()
        arrays[1] = arrays[0] * np.float32(.75) + np.float32(3.125)
        arrays[2] = np.float32(255.25) - arrays[0]
        self.assert_reference(arrays)

    def test_constant_patches_zero_norm_and_identical_points(self):
        arrays = inputs()
        arrays[0][0].fill(17.25)
        arrays[1][1].fill(0)
        arrays[2][2].fill(-4.5)
        for array in arrays[:3]:
            array[3].fill(1.125)
        arrays[4] = arrays[3].copy()
        self.assert_reference(arrays)
        actual = self.quality.evaluate(*arrays)
        np.testing.assert_array_equal(actual[:, 0], 0)
        self.assertEqual(actual[0, 1], -1)
        self.assertEqual(actual[1, 1], -1)
        self.assertEqual(actual[1, 2], -1)
        self.assertEqual(actual[2, 2], -1)
        np.testing.assert_array_equal(actual[3, 1:3], -1)
        np.testing.assert_array_equal(actual[3, 3:], 0)

    def test_large_finite_values_do_not_overflow_double_statistics(self):
        arrays = inputs()
        maximum = np.finfo(np.float32).max
        for array in arrays[:3]:
            array.flat[::2] = maximum
            array.flat[1::2] = -maximum
        arrays[3].fill(maximum)
        arrays[4].fill(-maximum)
        output = self.quality.evaluate(*arrays)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output, reference(arrays), rtol=2e-12, atol=2e-12)

    def test_nonfinite_in_each_input_is_rejected(self):
        for index in range(5):
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(index=index, value=value):
                    arrays = inputs()
                    arrays[index].flat[-1] = value
                    with self.assertRaises(ValueError):
                        self.quality.evaluate(*arrays)

    def test_c_count_null_guards_and_invalid_output_untouched(self):
        arrays = inputs()
        pointers = [a.ctypes.data for a in arrays]
        guarded = np.full(22, -12345.25, np.float64)
        target = guarded[1:21]
        function = self.quality.function
        for count in (-1, 0, 5, 2147483647):
            self.assertEqual(function(*pointers, count, target.ctypes.data), -1)
        for index in range(6):
            arguments = pointers + [4, target.ctypes.data]
            arguments[index if index < 5 else 6] = None
            self.assertEqual(function(*arguments), -1)
        arrays[2].flat[-1] = np.nan
        self.assertEqual(function(*pointers, 4, target.ctypes.data), -1)
        np.testing.assert_array_equal(guarded, -12345.25)
        arrays[2].flat[-1] = 1
        self.assertEqual(function(*pointers, 4, target.ctypes.data), 0)
        np.testing.assert_array_equal(guarded[[0, -1]], -12345.25)
        np.testing.assert_allclose(target.reshape(4, 5), reference(arrays), rtol=2e-12, atol=2e-12)


if __name__ == "__main__":
    unittest.main()

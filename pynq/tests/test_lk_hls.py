"""Host-side ABI and rejection checks; actual arithmetic is tested in HLS."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lk_hls import decode_results, validate_inputs
import lk_hls


class SparseLKContract(unittest.TestCase):
    def test_signed_q6_and_invalid_mask(self):
        words = np.array([(1 << 32) | ((-32 & 65535) << 16) | 64, 0], dtype=np.uint64)
        flow, valid = decode_results(words)
        np.testing.assert_allclose(flow[0], [1, -.5])
        self.assertTrue(np.isnan(flow[1]).all())
        self.assertEqual(valid.tolist(), [True, False])

    def test_coordinate_packing(self):
        gray = np.zeros((200, 320), np.uint8)
        packed = validate_inputs(gray, gray, [[10, 20], [319, 199]])
        self.assertEqual(packed.tolist(), [10 | (20 << 16), 319 | (199 << 16)])

    def test_reject_bad_frames_and_coordinates(self):
        gray = np.zeros((200, 320), np.uint8)
        for points in ([], [[-1, 0]], [[320, 0]], [[1.5, 2]], [[np.nan, 2]], [[0, 0]] * 65):
            with self.subTest(points=str(points)[:25]), self.assertRaises(ValueError):
                validate_inputs(gray, gray, points)
        for bad in (gray.astype(float), np.zeros((201, 320), np.uint8), gray[:, :, None]):
            with self.assertRaises(ValueError):
                validate_inputs(bad, gray, [[10, 10]])

    def test_timeout_keeps_memory_owned_by_axi(self):
        class Buffer:
            physical_address = 0x100000
            def __init__(self, dtype, count):
                self.values = np.zeros(count, dtype=dtype)
                self.freed = False
            def __setitem__(self, key, value):
                self.values[key] = value
            def __array__(self, dtype=None, copy=None):
                return np.asarray(self.values, dtype=dtype)
            def flush(self):
                pass
            def freebuffer(self):
                self.freed = True

        class Overlay:
            device = type("Device", (), {"timestamp": "test"})()
            def is_loaded(self):
                return True

        class MMIO:
            def read(self, offset):
                return 4  # Idle before dispatch, then never asserts done.
            def write(self, offset, value):
                pass

        driver = lk_hls.SparseLKHLS.__new__(lk_hls.SparseLKHLS)
        driver._lock = threading.Lock()
        driver._closed = driver._fault = False
        driver.overlay = Overlay()
        driver._timestamp = "test"
        driver._mmio = MMIO()
        driver.timeout = .001
        buffers = [Buffer(dtype, count) for dtype, count in
                   ((np.uint8, 64000), (np.uint8, 64000), (np.uint32, 64), (np.uint64, 64))]
        driver._buffers = list(buffers)
        gray = np.zeros((200, 320), np.uint8)
        quarantine = []
        with patch.object(lk_hls, "_QUARANTINE", quarantine):
            with self.assertRaises(TimeoutError):
                driver.process(gray, gray, [[10, 10]])
            driver.close()
            self.assertEqual(quarantine, buffers)
            self.assertFalse(any(buffer.freed for buffer in buffers))
            self.assertTrue(driver._fault)

    def test_changed_overlay_rejected_before_mmio_with_one_query(self):
        class Device:
            queries = 0
            @property
            def timestamp(self):
                self.queries += 1
                return "replacement"

        driver = lk_hls.SparseLKHLS.__new__(lk_hls.SparseLKHLS)
        driver._lock = threading.Lock()
        driver._closed = driver._fault = False
        device = Device()
        driver.overlay = type("Overlay", (), {"device": device})()
        driver._timestamp = "original"
        # No MMIO is installed: any hardware access would fail this test.
        gray = np.zeros((200, 320), np.uint8)
        with self.assertRaisesRegex(RuntimeError, "active overlay changed"):
            driver.process(gray, gray, [[10, 10]])
        self.assertEqual(device.queries, 1)


if __name__ == "__main__":
    unittest.main()

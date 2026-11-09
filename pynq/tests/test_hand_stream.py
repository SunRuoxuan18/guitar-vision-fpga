"""Host-side contracts for the optional fused hand_dw_pw_stream core."""
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_neural_pl import quantize_activation, quantize_weights
from hand_stream import (CORE_ADDRESS, RELU6_SCALE, HandStream, create_hand_stream,
                         stream_core_available)
from vision_hls import HLSKernel


class Buffer(np.ndarray):
    physical_address = 0x100000
    freed = False
    def flush(self): pass
    def invalidate(self): pass
    def freebuffer(self): self.freed = True


class HandStreamTests(unittest.TestCase):
    def test_packed_line_capacity_includes_tail_channels(self):
        x = np.zeros((1, 129, 31, 31), np.float32)
        dw = np.zeros((129, 1, 3, 3), np.float32)
        pw = np.zeros((24, 129, 1, 1), np.float32)
        self.assertFalse(HandStream.supports(x, dw, pw, 1, (1, 1, 1, 1)))

    @staticmethod
    def processor(description=None):
        ip_dict = {} if description is None else {"hand_dw_pw_stream": description}
        return type("Processor", (), {"_overlay": type("Overlay", (), {"ip_dict": ip_dict})()})()

    def engine(self):
        driver = HandStream.__new__(HandStream)
        driver.buffers = [np.zeros(size, dtype=dtype).view(Buffer) for size, dtype in (
            (32 * 32 * 288, np.int16), (25 * 288, np.int16), (48 * 288, np.int16),
            (2 * 288, np.float32), (48 * 32 * 32, np.int64))]
        driver.lock = threading.Lock()
        driver.closed = driver.fault = False
        driver.stopping = False
        driver.timeout = .01
        driver.weight_cache = None
        driver.calls = 0
        driver.kernel_ms = driver.total_ms = 0.0
        driver.profile = True
        driver.layer_profile = []
        driver.timings = {}
        driver.offset = lambda name: name
        driver.copy = lambda destination, source: np.copyto(destination, np.asarray(source).reshape(-1))
        driver.run = lambda pointers, scalars: .01
        return driver

    def operands(self, channels=4, outputs=3, height=5, width=7, kernel=3):
        return (np.ones((1, channels, height, width), np.float32),
                np.ones((channels, 1, kernel, kernel), np.float32), np.zeros(channels, np.float32),
                np.ones((outputs, channels, 1, 1), np.float32), np.arange(outputs, dtype=np.float32))

    @staticmethod
    def fixed_reference(x, dw, dw_bias, pw, pw_bias, stride, pads):
        """Host model of the HLS integer DW -> ReLU6 -> integer PW contract."""
        packed_x, input_scale = quantize_activation(x)
        packed_dw, dw_scales = quantize_weights(dw, depthwise=True)
        packed_pw, pw_scales = quantize_weights(pw)
        channels, height, width = x.shape[1:]
        outputs, kernel = pw.shape[0], dw.shape[2]
        top, left, bottom, right = pads
        out_h = (height + top + bottom - kernel) // stride + 1
        out_w = (width + left + right - kernel) // stride + 1
        blocks = (channels + 7) // 8
        values = packed_x.reshape(height, width, blocks, 8).reshape(height, width, -1)[:, :, :channels]
        dw_q = packed_dw.reshape(kernel, kernel, blocks, 8).reshape(kernel, kernel, -1)[:, :, :channels]
        pw_q = packed_pw.reshape(outputs, 1, 1, blocks, 8).reshape(outputs, -1)[:, :channels]
        feature = np.empty((out_h, out_w, channels), np.int16)
        params_scale = np.float32(dw_scales) * np.float32(input_scale)
        for y in range(out_h):
            for col in range(out_w):
                for channel in range(channels):
                    total = 0
                    for ky in range(kernel):
                        for kx in range(kernel):
                            iy, ix = y * stride - top + ky, col * stride - left + kx
                            if 0 <= iy < height and 0 <= ix < width:
                                total += int(values[iy, ix, channel]) * int(dw_q[ky, kx, channel])
                    relu6 = min(6., max(0., float(np.float32(total) * params_scale[channel] + dw_bias[channel])))
                    feature[y, col, channel] = min(32767, int(relu6 * (32767. / 6.) + .5))
        raw = np.einsum("hwc,oc->ohw", feature.astype(np.int64), pw_q.astype(np.int64), optimize=True)
        result = raw.astype(np.float32)[None] * (RELU6_SCALE * pw_scales)[None, :, None, None]
        return raw, result + pw_bias[None, :, None, None]

    def test_missing_core_is_optional(self):
        processor = self.processor()
        self.assertFalse(stream_core_available(processor))
        self.assertIsNone(create_hand_stream(processor))

    def test_factory_requires_explicit_fusion_flag(self):
        valid = {"phys_addr": CORE_ADDRESS, "type": "xilinx.com:hls:hand_dw_pw_stream:1.0"}
        with patch.dict("os.environ", {"HAND_STREAM_FUSION": "0"}):
            self.assertIsNone(create_hand_stream(self.processor(valid)))
        with patch.dict("os.environ", {"HAND_STREAM_FUSION": "1"}), \
             patch("hand_stream.HandStream", return_value="stream") as constructor:
            self.assertEqual(create_hand_stream(self.processor(valid)), "stream")
            constructor.assert_called_once()

    def test_present_core_requires_expected_identity(self):
        valid = {"phys_addr": CORE_ADDRESS, "type": "xilinx.com:hls:hand_dw_pw_stream:1.0"}
        self.assertTrue(stream_core_available(self.processor(valid)))
        for field, value in (("phys_addr", CORE_ADDRESS + 0x10000),
                             ("type", "xilinx.com:hls:other:1.0")):
            invalid = dict(valid, **{field: value})
            with self.assertRaisesRegex(ValueError, "identity"):
                stream_core_available(self.processor(invalid))

    def test_packed_ports_params_and_output_conversion(self):
        driver = self.engine()
        x, dw, dw_bias, pw, pw_bias = self.operands()
        pw.fill(32767)
        driver.buffers[4][:3 * 3 * 5] = 32767
        result = driver.convolve(x, dw, dw_bias, pw, pw_bias, stride=1, pads=(0, 0, 0, 0))
        self.assertEqual(result.shape, (1, 3, 3, 5))
        np.testing.assert_allclose(result, np.broadcast_to(6 + pw_bias[None, :, None, None], result.shape))
        self.assertEqual(driver.buffers[0][:5 * 7 * 8].reshape(5, 7, 8)[:, :, 4:].max(), 0)
        self.assertEqual(driver.buffers[3].dtype, np.float32)
        self.assertEqual(driver.calls, 1)
        self.assertEqual(driver.layer_profile[0]["kind"], "dw_pw_stream")

    def test_weight_cache_avoids_repacking_identical_weights(self):
        driver = self.engine()
        x, dw, dw_bias, pw, pw_bias = self.operands()
        driver.convolve(x, dw, dw_bias, pw, pw_bias)
        cached = driver.weight_cache
        driver.convolve(x, dw, dw_bias, pw, pw_bias)
        self.assertIs(driver.weight_cache, cached)
        self.assertEqual(driver.calls, 2)

    def test_weight_cache_strongly_retains_sources_against_address_reuse(self):
        driver = self.engine()
        x, dw, dw_bias, pw, pw_bias = self.operands()
        driver.convolve(x, dw, dw_bias, pw, pw_bias)
        cached = driver.weight_cache
        self.assertIs(cached[3], dw)
        self.assertIs(cached[4], pw)
        # Mutating a retained model weight may share the address, but it is not
        # an address-reuse event; callers must provide immutable model blobs.
        self.assertEqual(cached[0], (driver._weight_key(dw), driver._weight_key(pw)))

    def test_nonfinite_derived_scale_and_output_are_rejected(self):
        driver = self.engine()
        x, dw, dw_bias, pw, pw_bias = self.operands()
        with self.assertRaisesRegex(ValueError, "scale or bias"):
            driver.convolve(np.full_like(x, np.finfo(np.float32).max),
                            np.full_like(dw, np.finfo(np.float32).max), dw_bias, pw, pw_bias)
        driver = self.engine()
        driver.buffers[4][:45] = np.iinfo(np.int64).max
        with self.assertRaisesRegex(ValueError, "output"):
            driver.convolve(x, dw, dw_bias, np.full_like(pw, np.finfo(np.float32).max), pw_bias)

    def test_supports_exact_graph_api_and_all_padding_sides(self):
        x, dw, _, pw, _ = self.operands()
        self.assertTrue(HandStream.supports(x, dw, pw, 1, (2, 2, 2, 2)))
        self.assertFalse(HandStream.supports(x, dw, pw, 1, (0, 0, 3, 0)))
        self.assertFalse(HandStream.supports(x, dw, pw, 3, (0, 0, 0, 0)))
        self.assertFalse(HandStream.supports(x, dw[:, :, :1, :1], pw, 1, (0, 0, 0, 0)))

    def test_fused_quantized_reference_matches_driver_contract(self):
        driver = self.engine()
        rng = np.random.default_rng(22)
        x = rng.normal(0, .3, (1, 3, 5, 7)).astype(np.float32)
        dw = rng.normal(0, .2, (3, 1, 3, 3)).astype(np.float32)
        dw_bias = rng.normal(0, .1, 3).astype(np.float32)
        pw = rng.normal(0, .25, (4, 3, 1, 1)).astype(np.float32)
        pw_bias = rng.normal(0, .1, 4).astype(np.float32)
        raw, expected = self.fixed_reference(x, dw, dw_bias, pw, pw_bias, 1, (1, 1, 1, 1))
        def run(pointers, scalars):
            driver.buffers[4][:raw.size] = raw.reshape(-1)
            return .01
        driver.run = run
        actual = driver.convolve(x, dw, dw_bias, pw, pw_bias, 1, (1, 1, 1, 1))
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=0)

    def test_tight_dimension_and_line_storage_bounds_never_dispatch(self):
        driver = self.engine()
        dispatched = []
        driver.run = lambda *args: dispatched.append(args)
        for args in (self.operands(channels=145), self.operands(outputs=49),
                     self.operands(height=33), self.operands(width=33),
                     self.operands(channels=144, width=29)):
            with self.assertRaises(ValueError):
                driver.convolve(*args)
        self.assertEqual(dispatched, [])

    def test_faulted_driver_never_uses_a_cpu_fallback(self):
        driver = self.engine()
        driver.fault = True
        with self.assertRaisesRegex(RuntimeError, "faulted"):
            driver.convolve(*self.operands())

    def test_request_stop_and_close_release_cache_and_reject_work(self):
        driver = self.engine()
        driver.convolve(*self.operands())
        driver.request_stop()
        with self.assertRaisesRegex(RuntimeError, "stopping"):
            driver.convolve(*self.operands())
        driver.close()
        self.assertTrue(driver.closed)
        self.assertIsNone(driver.weight_cache)
        self.assertEqual(driver.buffers, [])

    def test_timeout_from_base_kernel_quarantines_all_ports(self):
        class MMIO:
            def read(self, _): return 4
            def write(self, *_): pass
        driver = HLSKernel.__new__(HLSKernel)
        driver.fault = driver.closed = False
        driver.timeout = .001
        driver.mmio = MMIO()
        driver.buffers = [object() for _ in range(5)]
        with patch("vision_hls._QUARANTINE", []) as quarantine:
            with self.assertRaises(TimeoutError):
                driver.run([])
            self.assertTrue(driver.fault)
            self.assertEqual(len(driver.buffers), 0)
            self.assertEqual(len(quarantine), 5)


if __name__ == "__main__":
    unittest.main()

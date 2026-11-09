"""Host tests for the fixed vision contract; no installed PYNQ is needed."""

from contextlib import redirect_stdout
import copy
import ctypes
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import vision_pl as vision
import verify_vision_pl as verify


PARAMETERS = {
    "c_include_sg": "0", "c_micro_dma": "0", "c_include_mm2s": "1",
    "c_include_s2mm": "1", "c_addr_width": "32", "c_sg_length_width": "23",
    "c_m_axi_mm2s_data_width": "64", "c_m_axi_s2mm_data_width": "64",
    "c_m_axis_mm2s_tdata_width": "32", "c_s_axis_s2mm_tdata_width": "32",
}

HWH_CLOCK = b'''<SYSTEM><MODULES>
<MODULE INSTANCE="ps7" MODTYPE="processing_system7"><PARAMETERS>
<PARAMETER NAME="PCW_FCLK0_PERIPHERAL_CLKSRC" VALUE="IO PLL"/>
<PARAMETER NAME="PCW_FCLK0_PERIPHERAL_DIVISOR0" VALUE="4"/>
<PARAMETER NAME="PCW_FCLK0_PERIPHERAL_DIVISOR1" VALUE="4"/>
<PARAMETER NAME="PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ" VALUE="100.000000"/>
</PARAMETERS><PORTS><PORT NAME="FCLK_CLK0" CLKFREQUENCY="100000000"/>
</PORTS></MODULE></MODULES></SYSTEM>'''


class FakeClocks:
    def __init__(self, runtime):
        self.runtime = runtime
        self.rate = 100.0
        self.set_result = None
        self.assignments = []

    @property
    def fclk0_mhz(self):
        return self.rate

    @fclk0_mhz.setter
    def fclk0_mhz(self, value):
        self.assignments.append(value)
        self.runtime.events.append(("set_fclk0", value))
        self.rate = value if self.set_result is None else self.set_result


class FakeBuffer(np.ndarray):
    def __new__(cls, address, events, name):
        result = np.zeros(vision.PIXELS, dtype=vision.WORD_DTYPE).view(cls)
        result.physical_address = address
        result.events = events
        result.name = name
        result.freed = False
        return result

    def flush(self):
        self.events.append(("flush", self.name))

    def invalidate(self):
        self.events.append(("invalidate", self.name))

    def freebuffer(self):
        self.events.append(("free", self.name))
        self.freed = True


class FakeMMIO:
    def __init__(self, runtime):
        self.runtime = runtime
        self.registers = {vision.MM2S_CONTROL: 0, vision.S2MM_CONTROL: 0,
                          vision.MM2S_STATUS: vision.HALTED,
                          vision.S2MM_STATUS: vision.HALTED}
        self.writes = []
        self.stuck_reset = False
        self.stuck_start = False
        self.stuck_frame = False
        self.short = False
        self.transfer_error = 0
        self.transfer_halted = False
        self.on_send = None

    def read(self, offset):
        self.runtime.events.append(("read", offset))
        return self.registers.get(offset, 0)

    def write(self, offset, value):
        self.runtime.events.append(("write", offset, value))
        self.writes.append((offset, value))
        if offset in (vision.MM2S_STATUS, vision.S2MM_STATUS):
            self.registers[offset] &= ~(value & vision.IRQ_MASK)
            return
        self.registers[offset] = value
        if offset == vision.MM2S_CONTROL and value & vision.RESET:
            if self.stuck_reset:
                return
            for base in (0, 0x30):
                self.registers[base] = 0
                self.registers[base + 4] = vision.HALTED
                self.registers[base + 0x28] = 0
        elif offset in (vision.MM2S_CONTROL, vision.S2MM_CONTROL) and value == 1:
            if not self.stuck_start:
                self.registers[offset + 4] = 0
        elif offset == vision.S2MM_LENGTH:
            self.registers[vision.S2MM_STATUS] = 0
        elif offset == vision.MM2S_LENGTH:
            self.registers[vision.MM2S_STATUS] = 0
            if self.on_send:
                self.on_send()
            if self.stuck_frame:
                return
            if self.transfer_error:
                self.registers[vision.MM2S_STATUS] = self.transfer_error
                return
            if self.transfer_halted:
                self.registers[vision.S2MM_STATUS] = vision.HALTED
                return
            tx, rx = self.runtime.buffers
            # Simulated hardware produces stream words; scalar/reference tests
            # independently prove arithmetic and OpenCV equivalence below.
            frame = tx.view(np.uint8).reshape(vision.HEIGHT, vision.WIDTH, 4)[:, :, :3]
            rx[:] = vision.reference_words(np.asarray(frame)).reshape(-1)
            for register in (vision.MM2S_STATUS, vision.S2MM_STATUS):
                self.registers[register] = vision.IDLE | 0x1000
            if self.short:
                self.registers[vision.S2MM_LENGTH] -= 4


class FakeOverlay:
    def __init__(self, runtime):
        self.runtime = runtime
        self.device = SimpleNamespace(timestamp="initial-device-state")
        self.ip_dict = {vision.DMA_NAME: {
            "type": "xilinx.com:ip:axi_dma:7.1", "phys_addr": 0x40400000,
            "addr_range": 0x10000, "parameters": dict(PARAMETERS)}}
        self.active = True
        self.downloads = 0

    def download(self):
        self.runtime.events.append(("download",))
        self.runtime.Clocks.rate = 62.5
        self.downloads += 1
        self.active = True
        self.device.timestamp = "download-{}".format(self.downloads)

    def is_loaded(self):
        return self.active

    def __getattr__(self, name):
        raise AssertionError("No lazy PYNQ DMA channel construction: " + name)


class FakeRuntime:
    __version__ = "2.7.0-fake"

    def __init__(self):
        self.events = []
        self.buffers = []
        self.Clocks = FakeClocks(self)
        self.overlay = FakeOverlay(self)
        self.mmio = FakeMMIO(self)
        self.overlay_calls = []
        self.mmio_calls = []
        self.allocate_fail = False

    def Overlay(self, bitfile, download):
        self.overlay_calls.append((bitfile, download))
        return self.overlay

    def MMIO(self, address, length, device):
        self.mmio_calls.append((address, length, device))
        return self.mmio

    def allocate(self, shape, dtype, target, cacheable=False):
        assert cacheable, 'PS/PL shared tensors must use explicit cache maintenance'
        if self.allocate_fail and self.buffers:
            raise MemoryError("CMA exhausted")
        if shape != (vision.PIXELS,) or dtype != vision.WORD_DTYPE or target is not self.overlay.device:
            raise AssertionError("Wrong allocation contract")
        buffer = FakeBuffer(0x10000000 + len(self.buffers) * 0x200000,
                            self.events, "tx" if not self.buffers else "rx")
        self.buffers.append(buffer)
        return buffer


def scalar_word(frame, y, x):
    def gray(row, column):
        b, g, r = (int(value) for value in frame[row, column])
        return (77 * r + 150 * g + 29 * b + 128) >> 8
    result = gray(y, x)
    if x < 2 or y < 2:
        return result
    p = [[gray(y - 2 + dy, x - 2 + dx) for dx in range(3)] for dy in range(3)]
    gx = -p[0][0] + p[0][2] - 2 * p[1][0] + 2 * p[1][2] - p[2][0] + p[2][2]
    gy = -p[0][0] - 2 * p[0][1] - p[0][2] + p[2][0] + 2 * p[2][1] + p[2][2]
    magnitude = min(abs(gx) + abs(gy), 255)
    return result | (magnitude << 8) | ((255 if magnitude >= 64 else 0) << 16)


class PackingAndReferenceTests(unittest.TestCase):
    def test_import_and_cli_help_do_not_import_pynq(self):
        code = ("import sys; sys.path.insert(0, {!r}); import vision_pl, verify_vision_pl; "
                "assert 'pynq' not in sys.modules; assert 'cv2' not in sys.modules; "
                "verify_vision_pl.parse_args(['--help'])").format(str(Path(vision.__file__).parent))
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--load-overlay", result.stdout)

    def test_packing_is_little_endian_bgr0_and_supports_strides(self):
        frame = np.zeros((vision.HEIGHT, vision.WIDTH * 2, 3), dtype=np.uint8)[:, ::2]
        frame[0, :3] = [[0x12, 0x34, 0x56], [255, 0, 0], [0, 0, 255]]
        out = np.full(vision.PIXELS, 0xffffffff, dtype=vision.WORD_DTYPE)
        self.assertIs(vision.pack_bgr(frame, out), out)
        self.assertEqual(list(out[:3]), [0x00563412, 0x000000ff, 0x00ff0000])
        self.assertEqual(out[:1].tobytes(), b"\x12\x34\x56\x00")
        self.assertFalse(np.any(out >> 24))

    def test_input_validation_and_output_storage_validation(self):
        bad = [None, [], np.zeros((480, 640, 3), dtype=np.float32),
               np.zeros((480, 640), dtype=np.uint8), np.zeros((640, 480, 3), dtype=np.uint8),
               np.zeros((480, 640, 4), dtype=np.uint8)]
        for frame in bad:
            with self.subTest(shape=getattr(frame, "shape", None)):
                with self.assertRaises((ValueError, TypeError)):
                    vision.pack_bgr(frame)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        outputs = [np.zeros(vision.PIXELS, dtype=">u4"),
                   np.zeros(vision.PIXELS, dtype=np.uint8),
                   np.zeros((480, 640), dtype=np.uint32),
                   np.zeros(vision.PIXELS * 2, dtype=np.uint32)[::2]]
        readonly = np.zeros(vision.PIXELS, dtype=np.uint32)
        readonly.flags.writeable = False
        outputs.append(readonly)
        for out in outputs:
            with self.assertRaises(ValueError):
                vision.pack_bgr(frame, out)

    def test_gray_coefficients_rounding_white_and_zero_border(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[0, :6] = [[255, 0, 0], [0, 255, 0], [0, 0, 255],
                         [255, 255, 255], [4, 0, 0], [5, 0, 0]]
        words = vision.reference_words(frame)
        self.assertEqual(list(words[0, :6]), [29, 149, 77, 255, 0, 1])
        self.assertFalse(np.any(words[:2] >> 8))
        self.assertFalse(np.any(words[:, :2] >> 8))

    def test_reference_matches_independent_scalar_causal_windows(self):
        frame = np.random.RandomState(7).randint(0, 256, (480, 640, 3)).astype(np.uint8)
        words = vision.reference_words(frame)
        locations = [(y, x) for y in (0, 1, 2, 3, 239, 479)
                     for x in (0, 1, 2, 3, 319, 639)]
        for y, x in locations:
            with self.subTest(y=y, x=x):
                self.assertEqual(int(words[y, x]), scalar_word(frame, y, x))

    def test_threshold_is_inclusive_and_l1_saturates(self):
        ramp = np.tile((np.arange(640) * 8 % 256).astype(np.uint8), (480, 1))
        words = vision.reference_words(np.repeat(ramp[:, :, None], 3, axis=2))
        self.assertEqual(int(words[2, 2] >> 8 & 255), 64)
        self.assertEqual(int(words[2, 2] >> 16 & 255), 255)
        ramp[:] = np.where(np.arange(640) < 4, 0, 255)
        words = vision.reference_words(np.repeat(ramp[:, :, None], 3, axis=2))
        self.assertEqual(int(words[2, 4] >> 8 & 255), 255)
        self.assertEqual(int(words[2, 3] >> 16 & 255), 0)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV not installed")
    def test_full_frame_cv2_exact_for_every_synthetic_pattern(self):
        for name, frame in verify.synthetic_frames():
            with self.subTest(pattern=name):
                np.testing.assert_array_equal(vision.reference_cv2_words(frame),
                                              vision.reference_words(frame))


class DriverTests(unittest.TestCase):
    def test_native_plane_transport_is_exact_detached_and_optional(self):
        driver = self.driver()
        calls = []
        def unpack(source, gray, edge, pixels):
            calls.append(pixels)
            raw = np.ctypeslib.as_array((ctypes.c_uint8 * (pixels * 4)).from_address(source)).reshape(pixels, 4)
            for address, lane in ((gray, 0), (edge, 2)):
                target = np.ctypeslib.as_array((ctypes.c_uint8 * pixels).from_address(address))
                target[:] = raw[:, lane]
        driver._native_unpack = unpack
        result = driver.process(self.frame, outputs=('gray', 'edge'))
        expected = vision.reference_words(self.frame)
        np.testing.assert_array_equal(result['gray'], (expected & 255).astype(np.uint8))
        np.testing.assert_array_equal(result['edge'], ((expected >> 16) & 255).astype(np.uint8))
        driver.process(np.zeros_like(self.frame), outputs=('gray',))
        self.assertEqual(calls, [vision.PIXELS])
        np.testing.assert_array_equal(result['gray'], (expected & 255).astype(np.uint8))

    def test_native_rgba_transport_preserves_dma_contract_for_strided_input(self):
        driver = self.driver()
        calls = []
        def native_pack(source,destination,pixels):
            calls.append(pixels)
            raw = np.ctypeslib.as_array((ctypes.c_uint8*(pixels*4)).from_address(source)).reshape(pixels,4)
            target = np.ctypeslib.as_array((ctypes.c_uint8*(pixels*4)).from_address(destination)).reshape(pixels,4)
            target[:] = raw[:,[2,1,0,3]]
            target[:,3] = 0
        driver._native_pack = native_pack
        rgba = np.empty((480,1280,4),np.uint8)[:,::2]
        rgba[:,:,:3] = self.frame[:,:,::-1]
        rgba[:,:,3] = 97
        actual = driver.process_rgba(rgba)
        np.testing.assert_array_equal(actual['words'],vision.reference_words(self.frame))
        self.assertEqual(calls,[vision.PIXELS])

    def test_direct_rgba_matches_bgr_with_strides_and_arbitrary_alpha(self):
        driver = self.driver()
        rgba = np.empty((480, 1280, 4), np.uint8)[:, ::2]
        rgba[:, :, :3] = self.frame[:, :, ::-1]
        rgba[:, :, 3] = np.arange(640, dtype=np.uint8)
        output = driver.process_rgba(rgba)
        np.testing.assert_array_equal(output["words"], vision.reference_words(self.frame))
        packed = np.asarray(self.runtime.buffers[0]).view(np.uint8).reshape(480, 640, 4)
        self.assertFalse(packed[:, :, 3].any())
        for bad in (rgba[:100], np.zeros((480,640,3),np.uint8), rgba.astype(float)):
            with self.assertRaises((TypeError, ValueError)):
                driver.process_rgba(bad)

    def test_receive_buffer_is_cleaned_once_then_invalidated_per_frame(self):
        driver = self.driver()
        driver.process(self.frame)
        self.assertEqual(self.runtime.events.count(("flush", "rx")), 1)
        self.runtime.events.clear()
        driver.process(np.zeros_like(self.frame))
        self.assertEqual(self.runtime.events.count(("flush", "tx")), 1)
        self.assertNotIn(("flush", "rx"), self.runtime.events)
        self.assertEqual(self.runtime.events.count(("invalidate", "rx")), 1)

    def test_requested_outputs_skip_unneeded_packed_words_and_magnitude(self):
        driver = self.driver()
        result = driver.process(self.frame, outputs=("gray", "edge"))
        self.assertTrue({"gray", "edge", "timings_ms", "provenance"}.issubset(result))
        self.assertNotIn("magnitude", result)
        self.assertNotIn("words", result)
        expected = vision.reference_words(self.frame)
        np.testing.assert_array_equal(result["gray"], (expected & 255).astype(np.uint8))
        np.testing.assert_array_equal(result["edge"], (expected >> 16 & 255).astype(np.uint8))
        self.assertTrue(result["gray"].flags.c_contiguous)
        self.assertTrue(result["edge"].flags.c_contiguous)
        gray, edge = result["gray"].copy(), result["edge"].copy()
        driver.process(np.zeros_like(self.frame), outputs=("gray", "edge"))
        np.testing.assert_array_equal(result["gray"], gray)
        np.testing.assert_array_equal(result["edge"], edge)

    def test_invalid_requested_outputs_do_not_start_dma(self):
        driver = self.driver()
        self.runtime.events.clear()
        for outputs, error in (((), ValueError), (("gray", "gray"), ValueError),
                               (("unknown",), ValueError), ("gray", TypeError),
                               ((1,), TypeError)):
            with self.subTest(outputs=outputs), self.assertRaises(error):
                driver.process(self.frame, outputs=outputs)
        self.assertFalse(self.runtime.events)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bitfile = Path(self.temp.name) / "vision_pl.bit"
        self.bitfile.write_bytes(b"test isolated bitstream")
        self.bitfile.with_suffix(".hwh").write_bytes(HWH_CLOCK)
        self.runtime = FakeRuntime()
        self.root_patch = patch.object(vision, "_require_root")
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.load_patch = patch.object(vision, "_load_pynq", return_value=self.runtime)
        self.loader = self.load_patch.start()
        self.addCleanup(self.load_patch.stop)
        self.frame = np.random.RandomState(12).randint(0, 256, (480, 640, 3)).astype(np.uint8)

    def driver(self, **kwargs):
        result = vision.VisionPL(self.bitfile, **kwargs)
        self.addCleanup(result.close)
        return result

    def test_root_required_before_loading_runtime(self):
        self.root_patch.stop()
        with patch.object(vision.os, "geteuid", return_value=1000, create=True):
            with self.assertRaises(PermissionError):
                vision.VisionPL(self.bitfile)
        self.loader.assert_not_called()

    def test_artifact_pair_and_finite_timeouts_required(self):
        for kwargs in ({"timeout": 0}, {"timeout": float("nan")},
                       {"timeout": float("inf")}, {"poll_interval": -1}, {"download": "yes"}):
            with self.assertRaises((ValueError, TypeError)):
                vision.VisionPL(self.bitfile, **kwargs)
        self.bitfile.with_suffix(".hwh").unlink()
        with self.assertRaises(FileNotFoundError):
            vision.VisionPL(self.bitfile)
        self.loader.assert_not_called()

    def test_download_opt_in_and_requires_matching_active_overlay(self):
        self.runtime.overlay.active = False
        with self.assertRaisesRegex(vision.VisionPLError, "not active"):
            vision.VisionPL(self.bitfile)
        self.assertEqual(self.runtime.overlay.downloads, 0)
        driver = self.driver(download=True)
        self.assertEqual(self.runtime.overlay.downloads, 1)
        self.assertTrue(all(download is False for _, download in self.runtime.overlay_calls))
        self.assertTrue(driver.snapshot()["ready"])

    def test_runtime_clock_is_checked_before_dma_access_and_recorded(self):
        for rate in (50.0, 99.0, 125.0, float("nan"), float("inf")):
            with self.subTest(fclk0_mhz=rate):
                self.runtime.Clocks.rate = rate
                with self.assertRaisesRegex(vision.VisionPLError, "FCLK0"):
                    vision.VisionPL(self.bitfile)
                self.assertFalse(self.runtime.mmio_calls)
                self.assertFalse(self.runtime.buffers)
        self.runtime.Clocks.rate = 100.0
        driver = self.driver()
        self.assertEqual(driver.fingerprint["clock"]["fclk0_hz"], vision.CLOCK_HZ)
        self.assertIn("register", driver.fingerprint["clock"]["evidence"])
        self.assertFalse(self.runtime.Clocks.assignments)

    def test_download_corrects_hwh_divisors_using_runtime_pll_before_dma(self):
        driver = self.driver(download=True)
        events = self.runtime.events
        self.assertLess(events.index(("download",)), events.index(("set_fclk0", 100.0)))
        self.assertLess(events.index(("set_fclk0", 100.0)),
                        events.index(("write", vision.MM2S_CONTROL, vision.RESET)))
        self.assertEqual(self.runtime.Clocks.assignments, [100.0])
        clock = driver.fingerprint["clock"]
        self.assertEqual(clock["before_configure_hz"], 62_500_000)
        self.assertEqual(clock["configured_hz"], 100_000_000)
        self.assertEqual(clock["actual_hz"], 100_000_000)
        self.assertEqual(clock["hwh"]["declared_hz"], 100_000_000)
        self.assertEqual(clock["hwh"]["divisor0"], 4)
        self.assertEqual(clock["hwh"]["divisor1"], 4)

    def test_download_clock_readback_tolerance_and_failed_set_before_dma(self):
        for readback in (62.5, 99.0, 101.0, float("nan")):
            with self.subTest(readback=readback):
                self.runtime.Clocks.set_result = readback
                with self.assertRaisesRegex(vision.VisionPLError, "FCLK0"):
                    vision.VisionPL(self.bitfile, download=True)
                self.assertFalse(self.runtime.mmio_calls)
                self.assertFalse(self.runtime.buffers)
        self.runtime.Clocks.set_result = 99.6
        driver = self.driver(download=True)
        self.assertEqual(driver.fingerprint["clock"]["actual_hz"], 99_600_000)

    def test_hwh_declared_clock_must_be_100mhz_before_download(self):
        hwh = self.bitfile.with_suffix(".hwh")
        for content in (HWH_CLOCK.replace(b'100000000', b'62500000'),
                        HWH_CLOCK.replace(b'100.000000', b'62.500000'), b'<SYSTEM/>'):
            with self.subTest(content=content):
                hwh.write_bytes(content)
                with self.assertRaisesRegex(vision.VisionPLError, "HWH"):
                    vision.VisionPL(self.bitfile, download=True)
                self.assertEqual(self.runtime.overlay.downloads, 0)
                self.assertFalse(self.runtime.mmio_calls)
                self.assertFalse(self.runtime.Clocks.assignments)

    def test_rejects_wrong_hwh_before_download_or_register_access(self):
        for key, bad in (("c_include_sg", "1"), ("c_sg_length_width", "14"),
                         ("c_include_s2mm", "0"), ("c_addr_width", "64"),
                         ("c_m_axi_mm2s_data_width", "128"),
                         ("c_s_axis_s2mm_tdata_width", "8"), ("c_micro_dma", "1")):
            with self.subTest(parameter=key):
                parameters = self.runtime.overlay.ip_dict[vision.DMA_NAME]["parameters"]
                old = parameters[key]
                parameters[key] = bad
                with self.assertRaises(vision.VisionPLError):
                    vision.VisionPL(self.bitfile, download=True)
                parameters[key] = old
        self.assertFalse(self.runtime.mmio.writes)
        self.assertEqual(self.runtime.overlay.downloads, 0)

    def test_both_memory_widths_and_propagated_mixed_width_are_supported(self):
        for tx_width, rx_width in ((32, 32), (64, 64), (64, 32), (32, 64)):
            with self.subTest(tx_width=tx_width, rx_width=rx_width):
                description = copy.deepcopy(self.runtime.overlay.ip_dict[vision.DMA_NAME])
                description["parameters"]["c_m_axi_mm2s_data_width"] = str(tx_width)
                description["parameters"]["c_m_axi_s2mm_data_width"] = str(rx_width)
                vision.VisionPL._validate_dma(description)

    def test_receive_first_cache_order_exact_words_and_stable_outputs(self):
        driver = self.driver()
        self.runtime.events.clear()
        result = driver.process(self.frame)
        expected = vision.reference_words(self.frame)
        np.testing.assert_array_equal(result["words"], expected)
        for key in ("gray", "magnitude", "edge"):
            self.assertEqual(result[key].dtype, np.uint8)
            self.assertEqual(result[key].shape, (480, 640))
        self.assertIs(result["timings"], result["timings_ms"])
        self.assertIs(result["hardware"], result["hardware_metadata"])
        self.assertFalse(result["provenance"]["cpu_fallback"])
        self.assertTrue(all(value >= 0 for value in result["timings_ms"].values()))
        events = self.runtime.events
        rx_start = events.index(("write", vision.S2MM_LENGTH, vision.FRAME_BYTES))
        tx_start = events.index(("write", vision.MM2S_LENGTH, vision.FRAME_BYTES))
        self.assertLess(rx_start, tx_start)
        self.assertLess(events.index(("flush", "tx")), rx_start)
        self.assertLess(events.index(("flush", "rx")), rx_start)
        self.assertGreater(events.index(("invalidate", "rx")), tx_start)
        driver.process(np.zeros_like(self.frame))
        self.assertEqual(len(self.runtime.buffers), 2)
        np.testing.assert_array_equal(result["words"], expected)
        driver.close()
        self.assertTrue(all(buffer.freed for buffer in self.runtime.buffers))
        self.assertEqual(driver.snapshot()["frames"], 2)
        self.assertFalse(driver.snapshot()["ready"])

    def test_single_timestamp_check_still_rejects_reloaded_overlay(self):
        driver = self.driver()
        self.runtime.overlay.timestamp = driver._device_timestamp
        driver.process(self.frame)
        self.runtime.overlay.device.timestamp = "replacement-overlay"
        with self.assertRaises(vision.VisionPLError):
            driver.process(self.frame)
        driver.close()

    def test_invalid_frame_does_not_touch_dma_or_poison_driver(self):
        driver = self.driver()
        self.runtime.events.clear()
        with self.assertRaises(ValueError):
            driver.process(self.frame[:100])
        self.assertFalse(self.runtime.events)
        self.assertTrue(driver.snapshot()["ready"])

    def test_sg_included_status_bit_is_rejected_before_start(self):
        driver = self.driver()
        self.runtime.mmio.registers[vision.MM2S_STATUS] |= 0x8
        writes = len(self.runtime.mmio.writes)
        with self.assertRaisesRegex(vision.VisionPLError, "scatter-gather"):
            driver._start(time.monotonic() + 1)
        self.assertEqual(len(self.runtime.mmio.writes), writes)

    def test_timeout_resets_and_faults_in_bounded_time(self):
        driver = self.driver(timeout=.02, poll_interval=.001)
        self.runtime.mmio.stuck_frame = True
        started = time.monotonic()
        with self.assertRaises(vision.VisionPLTimeout):
            driver.process(self.frame)
        self.assertLess(time.monotonic() - started, .5)
        status = driver.snapshot()
        self.assertFalse(status["ready"])
        self.assertTrue(driver.faulted)
        self.assertTrue(status["error"]["dma_reset_completed"])
        self.assertIn("s2mm_status", status["error"]["diagnostics"])
        writes = len(self.runtime.mmio.writes)
        with self.assertRaisesRegex(vision.VisionPLError, "faulted"):
            driver.process(self.frame)
        self.assertEqual(len(self.runtime.mmio.writes), writes)

    def test_stuck_reset_and_start_are_bounded(self):
        for phase in ("reset", "start"):
            with self.subTest(phase=phase):
                self.runtime = FakeRuntime()
                self.loader.return_value = self.runtime
                setattr(self.runtime.mmio, "stuck_" + phase, True)
                started = time.monotonic()
                with self.assertRaisesRegex(vision.VisionPLTimeout, phase):
                    vision.VisionPL(self.bitfile, timeout=.015, poll_interval=.001)
                self.assertLess(time.monotonic() - started, .5)
                self.assertTrue(all(buffer.freed for buffer in self.runtime.buffers))

    def test_failed_reset_quarantines_dma_memory_instead_of_freeing(self):
        driver = vision.VisionPL(self.bitfile, timeout=.01, poll_interval=.001)
        self.runtime.mmio.stuck_frame = self.runtime.mmio.stuck_reset = True
        count = len(vision._QUARANTINED_BUFFERS)
        with self.assertRaises(vision.VisionPLTimeout):
            driver.process(self.frame)
        with self.assertRaises(vision.VisionPLTimeout):
            driver.close()
        self.assertFalse(any(buffer.freed for buffer in self.runtime.buffers))
        self.assertEqual(len(vision._QUARANTINED_BUFFERS), count + 1)
        self.assertTrue(driver.snapshot()["buffers_quarantined"])
        driver.close()
        vision._QUARANTINED_BUFFERS.pop()

    def test_dma_errors_halt_and_short_frames_are_not_success(self):
        for attribute, value in (("transfer_error", 0x10), ("transfer_error", 0x20),
                                 ("transfer_error", 0x40), ("transfer_error", 0x4000),
                                 ("transfer_halted", True), ("short", True)):
            with self.subTest(attribute=attribute, value=value):
                self.runtime = FakeRuntime()
                self.loader.return_value = self.runtime
                driver = vision.VisionPL(self.bitfile)
                setattr(self.runtime.mmio, attribute, value)
                try:
                    with self.assertRaises(vision.VisionPLError):
                        driver.process(self.frame)
                    self.assertFalse(driver.snapshot()["ready"])
                    self.assertEqual(driver.snapshot()["frames"], 0)
                finally:
                    driver.close()

    def test_reject_stale_overlay_without_register_writes(self):
        driver = self.driver()
        self.runtime.overlay.device.timestamp = "another-overlay"
        self.runtime.events.clear()
        with self.assertRaisesRegex(vision.VisionPLError, "overlay changed"):
            driver.process(self.frame)
        self.assertFalse(self.runtime.events)

    def test_status_faults_on_overlay_replacement_without_register_access(self):
        for change in ("timestamp", "not_loaded"):
            with self.subTest(change=change):
                self.runtime = FakeRuntime()
                self.loader.return_value = self.runtime
                driver = self.driver()
                self.assertTrue(driver.status()["ready"])
                if change == "timestamp":
                    self.runtime.overlay.device.timestamp = "replacement-overlay"
                else:
                    self.runtime.overlay.active = False
                self.runtime.events.clear()
                status = driver.snapshot()
                self.assertFalse(status["ready"])
                self.assertEqual(status["error"]["type"], "VisionPLError")
                self.assertIn("overlay changed", status["error"]["message"])
                self.assertTrue(driver.faulted)
                self.assertEqual(status["hardware_fingerprint"], driver.fingerprint)
                self.runtime.overlay.active = True
                self.runtime.overlay.device.timestamp = driver._device_timestamp
                self.assertFalse(driver.status()["ready"])
                with self.assertRaisesRegex(vision.VisionPLError, "faulted"):
                    driver.process(self.frame)
                self.assertFalse(self.runtime.events)

    def test_lock_is_bounded_and_concurrent_frames_are_serialized(self):
        driver = self.driver(timeout=.02)
        driver._lock.acquire()
        try:
            with self.assertRaisesRegex(vision.VisionPLTimeout, "lock"):
                driver.process(self.frame)
        finally:
            driver._lock.release()
        self.assertTrue(driver.snapshot()["ready"])
        driver.timeout = 1
        entered, release = threading.Event(), threading.Event()
        def pause_first():
            if not entered.is_set():
                entered.set()
                if not release.wait(1):
                    raise AssertionError("Test release timed out")
        self.runtime.mmio.on_send = pause_first
        outputs, errors = [], []
        def run(frame):
            try:
                outputs.append(driver.process(frame))
            except Exception as exc:
                errors.append(exc)
        first = threading.Thread(target=run, args=(self.frame,))
        second = threading.Thread(target=run, args=(np.zeros_like(self.frame),))
        first.start()
        try:
            self.assertTrue(entered.wait(1))
            writes = len(self.runtime.mmio.writes)
            second.start()
            time.sleep(.01)
            self.assertEqual(len(self.runtime.mmio.writes), writes)
        finally:
            release.set()
            first.join(2)
            if second.ident is not None:
                second.join(2)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertFalse(errors)
        self.assertEqual([result["provenance"]["frame_sequence"] for result in outputs], [1, 2])
        np.testing.assert_array_equal(outputs[0]["words"], vision.reference_words(self.frame))
        self.assertFalse(np.any(outputs[1]["words"]))

    def test_partial_allocation_failure_frees_first_buffer(self):
        self.runtime.allocate_fail = True
        with self.assertRaises(MemoryError):
            vision.VisionPL(self.bitfile)
        self.assertTrue(self.runtime.buffers[0].freed)

    def test_physical_buffer_boundaries(self):
        buffer = FakeBuffer(0x10000000, [], "test")
        for address in (0, 3, 0x10000004, 2 ** 32 - vision.FRAME_BYTES + 8):
            buffer.physical_address = address
            with self.assertRaises(vision.VisionPLError):
                vision.VisionPL._validate_buffer(buffer)


class VerifierTests(unittest.TestCase):
    def test_compares_all_words_including_high_byte_and_last_pixel(self):
        expected = np.zeros((480, 640), dtype=vision.WORD_DTYPE)
        actual = expected.copy()
        actual[-1, -1] = 1 << 24
        report = verify.compare_words(expected, actual)
        self.assertFalse(report["exact"])
        self.assertEqual(report["mismatched_words"], 1)
        self.assertEqual(report["first_mismatch"]["x"], 639)
        self.assertEqual(report["first_mismatch"]["y"], 479)

    def test_cli_validation_and_npy_frame_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.npy"
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            np.save(str(path), frame)
            np.testing.assert_array_equal(verify.load_frame(path), frame)
            np.save(str(path), frame[:10])
            with self.assertRaises(ValueError):
                verify.load_frame(path)
        self.assertFalse(verify.parse_args([]).load_overlay)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV not installed")
    def test_board_report_with_fake_driver_and_mismatch_exit(self):
        class GoldenDriver:
            def __init__(self, *args, **kwargs):
                self.fingerprint = {"test_hardware": True}
            def process(self, frame):
                return {"words": vision.reference_words(frame),
                        "timings_ms": {"pack_ms": .2, "dma_ms": .5, "total_ms": 1.1},
                        "provenance": {"backend": "pl", "cpu_fallback": False}}
            def close(self):
                pass
            def status(self):
                return {"closed": True}
        args = verify.parse_args(["--iterations", "2", "--warmup", "1"])
        with patch.object(verify, "VisionPL", GoldenDriver):
            report = verify.verify(args)
        self.assertEqual(report["status"], "passed", report.get("error"))
        self.assertEqual(len(report["frames"]), 7)
        self.assertEqual(report["frames"][0]["timings_ms"]["dma_ms"]["samples"], 2)
        json.dumps(report, allow_nan=False)
        broken = copy.deepcopy(report)
        broken["status"] = "mismatch"
        with patch.object(verify, "verify", return_value=broken), redirect_stdout(io.StringIO()):
            self.assertEqual(verify.main([]), 1)


if __name__ == "__main__":
    unittest.main()

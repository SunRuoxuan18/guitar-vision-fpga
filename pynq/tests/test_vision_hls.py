"""Host contracts; arithmetic and performance require C/RTL and real hardware."""
import ctypes
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_hls import HLSKernel, PyramidLK, CPUPyramidLK, RoiProjection
from roi_fretboard import RoiFretboardTracker
from test_auto_fretboard import GuitarScene


class CompletionWaitTests(unittest.TestCase):
    def setUp(self):
        native = patch.dict(os.environ, {"FRETLENS_HLS_NATIVE_WAIT": "1"})
        native.start()
        self.addCleanup(native.stop)

    def driver(self, wait=None, control=None, reads=(4, 2)):
        driver = HLSKernel.__new__(HLSKernel)
        driver.fault, driver.closed, driver.timeout = False, False, .5
        driver.lock = threading.Lock()
        driver.buffers = [SimpleNamespace(physical_address=0x100000, nbytes=8, freebuffer=Mock()),
                          SimpleNamespace(physical_address=0x200000, nbytes=8, freebuffer=Mock())]
        driver._wait_done = wait
        driver._wait_library = None
        driver.mmio = SimpleNamespace(array=np.array([4], np.uint32) if control is None else control,
                                      read=Mock(side_effect=reads), write=Mock())
        return driver

    def test_constructor_loads_cdll_and_binds_wait_abi_for_lk_and_roi(self):
        for name in ("lk_pyramid", "roi_projection"):
            with self.subTest(name=name):
                wait = Mock(return_value=0)
                library = SimpleNamespace(fingertip_wait_done=wait)
                driver = self.driver()
                device = object()
                processor = SimpleNamespace(_overlay=SimpleNamespace(device=device, ip_dict={name: {
                    "type": "test:" + name + ":1.0", "phys_addr": 0x40000000, "addr_range": 0x1000}}))
                runtime = SimpleNamespace(MMIO=Mock(return_value=driver.mmio),
                                          allocate=Mock(side_effect=driver.buffers))
                with patch("vision_hls.ctypes.CDLL", return_value=library) as load, \
                        patch("vision_hls._load_pynq", return_value=runtime):
                    kernel = HLSKernel(processor, name, [((8,), np.uint8), ((8,), np.uint8)])
                self.assertEqual(Path(load.call_args.args[0]).name, "libfingertip_transport.so")
                self.assertIs(kernel._wait_library, library)
                self.assertIs(kernel._wait_done, wait)
                self.assertEqual(wait.argtypes, [ctypes.c_void_p, ctypes.c_double])
                self.assertIs(wait.restype, ctypes.c_int)
                kernel.run([])
                wait.assert_called_once_with(driver.mmio.array.ctypes.data, 2.)

    def test_neural_base_kernels_keep_existing_wait_selection(self):
        for name in ("hand_conv", "hand_conv_q16", "hand_requant", "hand_dw_pw_stream"):
            with self.subTest(name=name):
                driver = self.driver()
                processor = SimpleNamespace(_overlay=SimpleNamespace(device=object(), ip_dict={name: {
                    "type": "test:" + name + ":1.0", "phys_addr": 0x40000000, "addr_range": 0x1000}}))
                runtime = SimpleNamespace(MMIO=Mock(return_value=driver.mmio), allocate=Mock())
                with patch("vision_hls.ctypes.CDLL") as load, \
                        patch("vision_hls._load_pynq", return_value=runtime):
                    kernel = HLSKernel(processor, name, [])
                load.assert_not_called()
                self.assertIsNone(kernel._wait_done)
                kernel.run([])
                self.assertEqual(driver.mmio.read.call_count, 2)

    def test_native_success_uses_original_control_and_does_not_poll_python(self):
        wait = Mock(return_value=0)
        driver = self.driver(wait=wait, reads=(4,))
        buffers = driver.buffers[:]
        with patch("vision_hls.time.perf_counter", side_effect=[10., 10.025]), \
                patch("vision_hls.time.sleep") as sleep:
            elapsed = driver.run([(0x10, buffers[0])], [(0x40, 320)])
        self.assertAlmostEqual(elapsed, 25.)
        driver.mmio.read.assert_called_once_with(0)
        wait.assert_called_once_with(driver.mmio.array.ctypes.data, .5)
        self.assertEqual(driver.mmio.write.call_args_list,
                         [((0x10, 0x100000),), ((0x14, 0),), ((0x40, 320),), ((0, 1),)])
        sleep.assert_not_called()
        self.assertFalse(driver.fault)
        self.assertEqual(driver.buffers, buffers)

    def test_native_timeout_and_failed_wait_quarantine_without_poll_retry(self):
        for status, error in ((1, TimeoutError), (-1, RuntimeError), (7, RuntimeError)):
            with self.subTest(status=status):
                wait = Mock(return_value=status)
                driver = self.driver(wait=wait, reads=(4,))
                buffers = driver.buffers[:]
                with patch("vision_hls._QUARANTINE", []) as quarantine:
                    with self.assertRaises(error):
                        driver.run([])
                    self.assertTrue(driver.fault)
                    self.assertEqual(driver.buffers, [])
                    self.assertEqual(quarantine, buffers)
                    driver.close()
                    self.assertTrue(all(not b.freebuffer.called for b in buffers))
                    with self.assertRaises(RuntimeError):
                        driver.run([])
                driver.mmio.read.assert_called_once_with(0)
                wait.assert_called_once()
                driver.mmio.write.assert_called_once_with(0, 1)

    def test_native_exception_quarantines_and_preserves_original_error(self):
        failure = OSError("native wait failed")
        driver = self.driver(wait=Mock(side_effect=failure), reads=(4,))
        buffers = driver.buffers[:]
        with patch("vision_hls._QUARANTINE", []) as quarantine:
            with self.assertRaises(OSError) as raised:
                driver.run([])
            self.assertIs(raised.exception, failure)
            self.assertTrue(driver.fault)
            self.assertEqual(quarantine, buffers)
            self.assertEqual(driver.buffers, [])
        driver.mmio.read.assert_called_once_with(0)

    def test_uncertain_native_start_quarantines_without_entering_wait(self):
        wait = Mock(return_value=0)
        driver = self.driver(wait=wait, reads=(4,))
        driver.mmio.write.side_effect = OSError("uncertain start")
        buffers = driver.buffers[:]
        with patch("vision_hls._QUARANTINE", []) as quarantine:
            with self.assertRaisesRegex(OSError, "uncertain start"):
                driver.run([])
            self.assertTrue(driver.fault)
            self.assertEqual(driver.buffers, [])
            self.assertEqual(quarantine, buffers)
        wait.assert_not_called()

    def test_missing_library_or_symbol_preserves_polling(self):
        for library in (None, SimpleNamespace()):
            with self.subTest(missing="library" if library is None else "symbol"):
                with patch("vision_hls.ctypes.CDLL", return_value=library,
                           side_effect=OSError("unavailable") if library is None else None):
                    from vision_hls import _load_native_wait
                    loaded, wait = _load_native_wait()
                self.assertIsNone(loaded)
                self.assertIsNone(wait)
                driver = self.driver(wait=wait, reads=(4, 0, 2))
                with patch("vision_hls.time.sleep") as sleep:
                    driver.run([])
                self.assertEqual(driver.mmio.read.call_count, 3)
                sleep.assert_called_once_with(.00005)
                self.assertFalse(driver.fault)

    def test_native_opt_out_preserves_polling(self):
        wait = Mock(return_value=0)
        driver = self.driver(wait=wait)
        with patch.dict(os.environ, {"FRETLENS_HLS_NATIVE_WAIT": "0"}):
            driver.run([])
        wait.assert_not_called()
        self.assertEqual(driver.mmio.read.call_count, 2)

    def test_missing_control_array_preserves_polling(self):
        wait = Mock(return_value=0)
        driver = self.driver(wait=wait)
        del driver.mmio.array
        driver.run([])
        wait.assert_not_called()
        self.assertEqual(driver.mmio.read.call_count, 2)

    def test_invalid_control_array_uses_polling_without_passing_native_pointer(self):
        unaligned = np.ndarray((1,), np.uint32, buffer=bytearray(5), offset=1)
        for control in ([4], np.zeros(0, np.uint32), np.array([4], np.uint64),
                        np.array([4], dtype=">u4"), np.zeros(4, np.uint32)[::2], unaligned):
            with self.subTest(control=repr(control)):
                wait = Mock(return_value=0)
                driver = self.driver(wait=wait, control=control)
                driver.run([])
                wait.assert_not_called()
                self.assertEqual(driver.mmio.read.call_count, 2)
                self.assertFalse(driver.fault)

    def test_fault_closed_or_busy_core_does_not_start_or_wait(self):
        for state in ("fault", "closed", "busy"):
            with self.subTest(state=state):
                wait = Mock(return_value=0)
                driver = self.driver(wait=wait, reads=(0,) if state == "busy" else (4,))
                if state != "busy":
                    setattr(driver, state, True)
                with self.assertRaises(RuntimeError):
                    driver.run([])
                wait.assert_not_called()
                driver.mmio.write.assert_not_called()


class HardwareContractTests(unittest.TestCase):
    def test_reverse_reuses_only_the_same_pair_and_register_capability(self):
        driver=PyramidLK.__new__(PyramidLK)
        driver.lock=threading.Lock()
        driver.closed=driver.fault=False
        driver.cache_supported=True
        driver._cached_pair=None
        class Buffer(np.ndarray):
            def invalidate(self): pass
        driver.buffers=[np.zeros(64000,np.uint8).view(Buffer),np.zeros(64000,np.uint8).view(Buffer),
                        np.zeros(128,np.float32).view(Buffer),np.zeros(256,np.float32).view(Buffer)]
        calls=[]
        copies=[]
        driver.copy=lambda buffer,array:copies.append(array)
        driver.run=lambda ptrs,scalars: calls.append(dict(scalars)) or 1
        a=np.zeros((200,320),np.uint8);b=a.copy();points=np.float32([[50,50]])
        driver(a,b,points)
        driver(b,a,points)
        self.assertEqual([c[0x58] for c in calls],[0,1])
        self.assertEqual(len(copies),4)
        driver(b,a,points)
        self.assertEqual(calls[-1][0x58],0)
        driver.cache_supported=False
        driver(a,b,points)
        self.assertNotIn(0x58,calls[-1])

    def test_timeout_quarantines_live_axi_buffers(self):
        class Buffer:
            physical_address = 0x100000
            freed = False
            def freebuffer(self):
                self.freed = True
        class MMIO:
            def read(self, offset):
                return 4
            def write(self, offset, value):
                pass
        driver = HLSKernel.__new__(HLSKernel)
        driver.fault, driver.closed, driver.timeout, driver.mmio = False, False, .001, MMIO()
        driver.lock = threading.Lock()
        buffers = [Buffer(), Buffer()]
        driver.buffers = list(buffers)
        quarantine = []
        with patch("vision_hls._QUARANTINE", quarantine):
            with self.assertRaises(TimeoutError):
                driver.run([(0x10, buffers[0])])
            driver.close()
        self.assertEqual(quarantine, buffers)
        self.assertFalse(any(b.freed for b in buffers))

    def test_uncertain_start_write_quarantines_buffers(self):
        driver = HLSKernel.__new__(HLSKernel)
        driver.fault, driver.closed, driver.timeout = False, False, .001
        driver.lock = threading.Lock()
        driver.mmio = type("MMIO", (), {"read": lambda *_: 4,
            "write": lambda *_: (_ for _ in ()).throw(OSError("launch failed"))})()
        buffer = type("Buffer", (), {"freebuffer": lambda _: self.fail("freed live buffer")})()
        driver.buffers = [buffer]
        with patch("vision_hls._QUARANTINE", []) as quarantine:
            with self.assertRaises(OSError):
                driver.run([])
            driver.close()
            self.assertEqual(quarantine, [buffer])

    def test_close_quarantines_when_request_owns_lock(self):
        driver = HLSKernel.__new__(HLSKernel)
        driver.fault, driver.closed, driver.timeout = False, False, .001
        driver.lock = threading.Lock()
        buffer = type("Buffer", (), {"freebuffer": lambda _: self.fail("freed live buffer")})()
        driver.buffers = [buffer]
        driver.lock.acquire()
        try:
            with patch("vision_hls._QUARANTINE", []) as quarantine:
                driver.close()
                self.assertEqual(quarantine, [buffer])
        finally:
            driver.lock.release()

    def test_invalid_flow_never_dispatches(self):
        driver = PyramidLK.__new__(PyramidLK)
        frame = np.zeros((200, 320), np.uint8)
        for points in ([], [[np.nan, 1]], [[1, 1]] * 65):
            with self.assertRaises(ValueError):
                driver(frame, frame, points)

    def test_invalid_cpu_reference_never_enters_native_code(self):
        driver = CPUPyramidLK.__new__(CPUPyramidLK)
        frame = np.zeros((200, 320), np.uint8)
        for other, points in ((frame[:, :100], [[1, 1]]), (frame, []),
                              (frame, [[np.inf, 1]]), (frame, [[1, 1]] * 65)):
            with self.assertRaises(ValueError):
                driver(frame, other, points)

    def test_invalid_transform_never_dispatches_roi(self):
        driver = RoiProjection.__new__(RoiProjection)
        frame = np.zeros((480, 640), np.uint8)
        for matrix in (np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                driver(frame, matrix)

    def test_injected_backends_preserve_roi_tracking_and_occlusion(self):
        calls = {"flow": 0, "roi": 0}
        def flow(a, b, p):
            calls["flow"] += 1
            return cv2.calcOpticalFlowPyrLK(a, b, p, None, winSize=(21, 21), maxLevel=3)
        def roi(gray, transform):
            calls["roi"] += 1
            rectified = cv2.warpPerspective(gray, transform, (320, 128))
            blur = cv2.GaussianBlur(rectified, (3, 3), 0)
            return rectified, np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0)), np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1))
        scene = GuitarScene()
        tracker = RoiFretboardTracker(flow_backend=flow, projection_backend=roi)
        corners = [{"x": float(x / 639), "y": float(y / 399)} for x, y in scene.quad]
        tracker.initialize(scene.render()[0], corners)
        for step in range(3):
            result = tracker.update(scene.render(angle=step, dx=step)[0])
            self.assertEqual(result["status"], "tracking", result)
        self.assertGreaterEqual(calls["flow"], 6)
        self.assertGreaterEqual(calls["roi"], 4)
        self.assertEqual(tracker.update(np.zeros((400, 640), np.uint8))["strings"], [])


if __name__ == "__main__":
    unittest.main()

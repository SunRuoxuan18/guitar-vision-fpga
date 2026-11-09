"""Bounded HLS drivers sharing the active vision overlay; never program PS/DDR.

LK/ROI use an optional CDLL completion wait. FRETLENS_HLS_NATIVE_WAIT=0
keeps Python polling; absent libraries, symbols or MMIO arrays also fall back.
"""
import ctypes
import os
from pathlib import Path
import threading
import time

import numpy as np

from vision_pl import _load_pynq

_QUARANTINE = []


def _load_native_wait():
    try:
        library = ctypes.CDLL(str(Path(__file__).with_name("libfingertip_transport.so")))
        wait = library.fingertip_wait_done
    except (OSError, AttributeError):
        return None, None
    wait.argtypes = [ctypes.c_void_p, ctypes.c_double]
    wait.restype = ctypes.c_int
    return library, wait


class HLSKernel:
    def __init__(self, processor, name, buffers, timeout=2.0):
        self.processor = processor
        self.timeout = timeout
        self.lock = threading.Lock()
        self.fault = False
        self.closed = False
        self.buffers = []
        runtime = _load_pynq()
        overlay = processor._overlay
        description = overlay.ip_dict.get(name, {})
        if ":{}:".format(name) not in description.get("type", ""):
            raise RuntimeError("Combined overlay does not contain " + name)
        self.mmio = runtime.MMIO(description["phys_addr"], description["addr_range"], device=overlay.device)
        self._wait_library, self._wait_done = (None, None)
        if name in ("lk_pyramid", "roi_projection"):
            self._wait_library, self._wait_done = _load_native_wait()
        try:
            for shape, dtype in buffers:
                # PS repeatedly packs and reads these tensors. Cache maintenance
                # remains explicit at every handoff to the noncoherent HP port.
                buffer = runtime.allocate(shape=shape, dtype=dtype, target=overlay.device,
                                          cacheable=True)
                self.buffers.append(buffer)
                address = int(buffer.physical_address)
                if address <= 0 or address % 8 or address + buffer.nbytes > 2**32:
                    raise RuntimeError("Invalid HLS physical buffer")
        except Exception:
            self.close()
            raise
        self.timings = {}

    def run(self, pointers, scalars=()):
        if self.fault or self.closed:
            raise RuntimeError("HLS kernel faulted")
        # The bridge serializes all PL work. VisionPL.process validates overlay
        # identity immediately before these calls; standalone benchmarks do so
        # explicitly. Avoid redundant PL-server IPC for every LK direction.
        if not self.mmio.read(0) & 4:
            raise RuntimeError("HLS kernel is not idle")
        for offset, buffer in pointers:
            self.mmio.write(offset, int(buffer.physical_address))
            self.mmio.write(offset + 4, 0)
        for offset, value in scalars:
            self.mmio.write(offset, int(value))
        control = getattr(self.mmio, "array", None)
        wait = getattr(self, "_wait_done", None)
        if (os.environ.get("FRETLENS_HLS_NATIVE_WAIT", "1") != "1"
                or not isinstance(control, np.ndarray) or control.dtype != np.uint32
                or not control.dtype.isnative or not control.flags.c_contiguous
                or not control.flags.aligned or not control.size):
            wait = None
        started = time.perf_counter()
        try:
            self.mmio.write(0, 1)
            if wait is not None:
                # CDLL releases the GIL for the complete wait. Keep the original
                # mapped array alive locally until it returns; never copy MMIO.
                status = wait(control.ctypes.data, self.timeout)
                if status == 1:
                    raise TimeoutError("HLS timeout; live DMA buffers quarantined")
                if status != 0:
                    raise RuntimeError("Native HLS wait failed; live DMA buffers quarantined")
            else:
                while not self.mmio.read(0) & 2:
                    if time.perf_counter() - started > self.timeout:
                        raise TimeoutError("HLS timeout; live DMA buffers quarantined")
                    time.sleep(.00005)
        except Exception:
            self.fault = True
            _QUARANTINE.extend(self.buffers)
            self.buffers = []
            raise
        return (time.perf_counter() - started) * 1000

    @staticmethod
    def copy(buffer, array):
        source = np.ascontiguousarray(array)
        ctypes.memmove(np.asarray(buffer).ctypes.data, source.ctypes.data, source.nbytes)
        buffer.flush()

    def close(self):
        self.closed = True
        acquired = self.lock.acquire(timeout=self.timeout)
        try:
            if acquired:
                for buffer in self.buffers:
                    buffer.freebuffer()
            else:
                # A request may still own the core during HTTP shutdown.
                _QUARANTINE.extend(self.buffers)
            self.buffers = []
            self.fault = True
        finally:
            if acquired:
                self.lock.release()


class PyramidLK(HLSKernel):
    backend_name = "pl_pyramidal_lk_ps_ransac"
    def __init__(self, processor):
        super().__init__(processor, "lk_pyramid", [((64000,), np.uint8), ((64000,), np.uint8),
                         ((128,), np.float32), ((256,), np.float32)])
        self.cache_supported = "reuse_reverse" in processor._overlay.ip_dict["lk_pyramid"].get("registers", {})
        self._cached_pair = None

    def __call__(self, previous, current, points):
        started = time.perf_counter()
        points = np.asarray(points, np.float32).reshape(-1, 2)
        if (previous.dtype != np.uint8 or current.dtype != np.uint8
                or previous.ndim != 2 or previous.shape != current.shape
                or not 16 <= previous.shape[0] <= 200 or not 16 <= previous.shape[1] <= 320
                or not 1 <= len(points) <= 64 or not np.isfinite(points).all()):
            raise ValueError("Invalid pyramid LK inputs")
        with self.lock:
            if self.fault or self.closed:
                raise RuntimeError("HLS flow is closed or faulted")
            before, after, features, output = self.buffers
            reverse = (self.cache_supported and self._cached_pair is not None
                       and self._cached_pair[0] is current and self._cached_pair[1] is previous)
            self._cached_pair = None
            if not reverse:
                self.copy(before, previous)
                self.copy(after, current)
            self.copy(features, points)
            copied = time.perf_counter()
            kernel_ms = self.run([(0x10, before), (0x1c, after), (0x28, features), (0x34, output)],
                                 [(0x40, previous.shape[1]), (0x48, previous.shape[0]), (0x50, len(points))]
                                 + ([(0x58, int(reverse))] if self.cache_supported else []))
            if self.cache_supported and not reverse:
                self._cached_pair = (previous, current)
            output.invalidate()
            values = np.asarray(output)[:len(points) * 4].copy().reshape(-1, 4)
            self.timings = {"total_ms": (time.perf_counter() - started) * 1000,
                            "copy_flush_ms": (copied - started) * 1000, "kernel_ms": kernel_ms,
                            "pyramid_reused": reverse}
            return values[:, :2].reshape(-1, 1, 2), (values[:, 2] > .5).astype(np.uint8).reshape(-1, 1), values[:, 3:4]


class CPUPyramidLK:
    """Exact C algorithm on ARM for comparison, explicitly labelled as CPU."""
    backend_name = "ps_exact_pyramidal_lk_ransac"

    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        self.function = self.library.cpu_lk_pyramid
        self.function.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_uint]*3
        self.timings = {}

    def __call__(self, previous, current, points):
        previous = np.ascontiguousarray(previous, dtype=np.uint8)
        current = np.ascontiguousarray(current, dtype=np.uint8)
        points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 2)
        if (previous.ndim != 2 or previous.shape != current.shape
                or not 16 <= previous.shape[0] <= 200 or not 16 <= previous.shape[1] <= 320
                or not 1 <= len(points) <= 64 or not np.isfinite(points).all()):
            raise ValueError("Invalid CPU pyramid LK inputs")
        output = np.empty((len(points), 4), np.float32)
        started = time.perf_counter()
        self.function(previous.ctypes.data, current.ctypes.data, points.ctypes.data,
                      output.ctypes.data, previous.shape[1], previous.shape[0], len(points))
        self.timings = {"total_ms": (time.perf_counter()-started)*1000}
        return output[:, :2].reshape(-1, 1, 2), (output[:, 2] > .5).astype(np.uint8).reshape(-1, 1), output[:, 3:4]

    def close(self):
        pass


class RoiProjection(HLSKernel):
    def __init__(self, processor):
        super().__init__(processor, "roi_projection", [((640*480,), np.uint8),
                         ((9,), np.float32), ((320*128,), np.uint32)])

    def __call__(self, gray, transform):
        started = time.perf_counter()
        if gray.shape != (480, 640) or gray.dtype != np.uint8:
            raise ValueError("ROI core requires uint8 640x480")
        transform = np.asarray(transform, dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("Invalid perspective transform")
        try:
            inverse = np.linalg.inv(transform).astype(np.float32)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular perspective transform") from exc
        scale = np.max(np.abs(inverse))
        if scale == 0 or not np.isfinite(scale):
            raise ValueError("Invalid inverse perspective transform")
        inverse /= scale
        if not np.isfinite(inverse).all():
            raise ValueError("Invalid inverse perspective transform")
        with self.lock:
            if self.fault or self.closed:
                raise RuntimeError("HLS ROI is closed or faulted")
            image, matrix, output = self.buffers
            self.copy(image, gray)
            self.copy(matrix, inverse)
            copied = time.perf_counter()
            kernel_ms = self.run([(0x10, image), (0x1c, matrix), (0x28, output)])
            output.invalidate()
            words = np.asarray(output).copy().reshape(128, 320)
            rectified = (words & 255).astype(np.uint8)
            gx = ((words >> 8) & 2047).astype(np.float32)
            gy = ((words >> 19) & 2047).astype(np.float32)
            self.timings = {"total_ms": (time.perf_counter() - started) * 1000,
                            "copy_flush_ms": (copied - started) * 1000, "kernel_ms": kernel_ms}
            return rectified, gx, gy

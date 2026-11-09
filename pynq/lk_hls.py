"""Explicitly enabled experimental single-scale sparse LK accelerator.

This is not the pyramidal, forward/backward checked production tracker.
The separate overlay replaces other PL functions. Never initializes PS/DDR.
"""
import math
import ctypes
from pathlib import Path
import threading
import time

import numpy as np

from vision_pl import _hwh_clock, _load_pynq, _require_root

DEFAULT_BITFILE = Path(__file__).resolve().parents[1] / "output/lk_hls/lk_hls.bit"
MAX_POINTS = 64
_QUARANTINE = []
POINTERS = (0x10, 0x1c, 0x28, 0x34)


def validate_inputs(previous, current, points):
    for frame in (previous, current):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 2:
            raise ValueError("frames must be uint8 grayscale arrays")
        if not (16 <= frame.shape[0] <= 200 and 16 <= frame.shape[1] <= 320):
            raise ValueError("frame dimensions must be 16..200 by 16..320")
    if previous.shape != current.shape:
        raise ValueError("frame shapes must match")
    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] != 2 or not 1 <= len(values) <= MAX_POINTS:
        raise ValueError("points must have shape (1..64, 2)")
    if not np.isfinite(values).all() or (values != np.rint(values)).any():
        raise ValueError("experimental kernel requires finite integer pixel coordinates")
    height, width = previous.shape
    if (values < 0).any() or (values >= (width, height)).any():
        raise ValueError("point is outside the image")
    coordinates = values.astype(np.uint32)
    return coordinates[:, 0] | (coordinates[:, 1] << 16)


def decode_results(words):
    words = np.asarray(words, dtype=np.uint64)
    dx = (words & 0xffff).astype(np.uint16).view(np.int16).astype(np.float32) / 64
    dy = ((words >> 16) & 0xffff).astype(np.uint16).view(np.int16).astype(np.float32) / 64
    valid = ((words >> 32) & 1).astype(bool)
    flow = np.stack((dx, dy), axis=-1)
    flow[~valid] = np.nan
    return flow, valid


class SparseLKHLS:
    def __init__(self, bitfile=DEFAULT_BITFILE, *, download=False, timeout=1.0):
        if not isinstance(download, bool):
            raise TypeError("download must be explicitly boolean")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        _require_root()
        self.timeout = timeout
        self._lock = threading.Lock()
        self._buffers = []
        self._fault = False
        self._closed = False
        bitfile = Path(bitfile).resolve()
        if not bitfile.is_file() or not bitfile.with_suffix(".hwh").is_file():
            raise FileNotFoundError("matching HLS bit/hwh files are required")
        _hwh_clock(bitfile.with_suffix(".hwh"))
        runtime = _load_pynq()
        self.overlay = runtime.Overlay(str(bitfile), download=False)
        description = self.overlay.ip_dict.get("lk_sparse", {})
        if ":lk_sparse:" not in description.get("type", ""):
            raise RuntimeError("HWH does not describe lk_sparse")
        if download:
            self.overlay.download()
            runtime.Clocks.fclk0_mhz = 100
        if not self.overlay.is_loaded() or abs(runtime.Clocks.fclk0_mhz - 100) > .5:
            raise RuntimeError("HLS overlay must be active at 100MHz")
        self._timestamp = getattr(self.overlay.device, "timestamp", None)
        if not self._timestamp:
            raise RuntimeError("active overlay must have a programming timestamp")
        self._mmio = runtime.MMIO(description["phys_addr"], description["addr_range"],
                                  device=self.overlay.device)
        try:
            for shape, dtype in (((64000,), np.uint8), ((64000,), np.uint8),
                                 ((MAX_POINTS,), np.uint32), ((MAX_POINTS,), np.uint64)):
                buffer = runtime.allocate(shape=shape, dtype=dtype, target=self.overlay.device)
                self._buffers.append(buffer)
                address = int(buffer.physical_address)
                if address <= 0 or address % 8 or address + buffer.nbytes > 2 ** 32:
                    raise RuntimeError("buffer must be aligned in 32-bit physical memory")
        except Exception:
            self.close()
            raise

    def process(self, previous, current, points):
        started = time.monotonic()
        packed = validate_inputs(previous, current, points)
        with self._lock:
            if self._fault or self._closed:
                raise RuntimeError("HLS driver is closed or faulted")
            # Device.timestamp already performs a fresh PL-server query. Calling
            # is_loaded as well duplicates the same expensive metadata IPC.
            if getattr(self.overlay.device, "timestamp", None) != self._timestamp:
                raise RuntimeError("active overlay changed")
            if not self._mmio.read(0) & 4:
                raise RuntimeError("HLS kernel is not idle")
            checked = time.monotonic()
            before, after, coordinates, result = self._buffers
            pixels = previous.size
            previous = np.ascontiguousarray(previous)
            current = np.ascontiguousarray(current)
            ctypes.memmove(np.asarray(before).ctypes.data, previous.ctypes.data, pixels)
            ctypes.memmove(np.asarray(after).ctypes.data, current.ctypes.data, pixels)
            np.copyto(np.asarray(coordinates)[:len(packed)], packed)
            np.asarray(result).fill(0)
            copied = time.monotonic()
            for buffer in self._buffers:
                buffer.flush()
            flushed = time.monotonic()
            for offset, buffer in zip(POINTERS, self._buffers):
                self._mmio.write(offset, int(buffer.physical_address))
                self._mmio.write(offset + 4, 0)
            height, width = previous.shape
            for offset, value in ((0x40, width), (0x48, height), (0x50, len(packed))):
                self._mmio.write(offset, value)
            dispatched = time.monotonic()
            self._mmio.write(0, 1)
            try:
                while not self._mmio.read(0) & 2:
                    if time.monotonic() - dispatched > self.timeout:
                        raise TimeoutError("HLS kernel timed out; buffers quarantined")
                    time.sleep(.00005)
            except Exception:
                # The AXI master can still own these addresses after a timeout.
                self._fault = True
                _QUARANTINE.extend(self._buffers)
                self._buffers = []
                raise
            completed = time.monotonic()
            result.invalidate()
            flow, valid = decode_results(np.asarray(result)[:len(packed)].copy())
            ended = time.monotonic()
            return {"flow": flow, "valid": valid,
                    "total_ms": (ended - started) * 1000,
                    "dispatch_to_done_ms": (completed - dispatched) * 1000,
                    "identity_check_ms": (checked - started) * 1000,
                    "copy_ms": (copied - checked) * 1000,
                    "flush_ms": (flushed - copied) * 1000,
                    "register_ms": (dispatched - flushed) * 1000,
                    "decode_ms": (ended - completed) * 1000,
                    "backend": "experimental_single_scale_lk_hls"}

    def close(self):
        with self._lock:
            for buffer in self._buffers:
                buffer.freebuffer()
            self._buffers = []
            self._closed = True

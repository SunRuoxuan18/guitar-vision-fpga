"""Bounded PS driver for the isolated 640x480 vision overlay (Python 3.8).

Input is uint8 BGR. AXI words are little-endian B,G,R,0 on input and
gray, saturated L1 Sobel, edge (255/0), 0 on output. No CPU fallback exists.
Overlay download is opt-in and replaces the current PL configuration.

PYNQ 2.7 upstream evidence (consulted for this implementation):
https://github.com/Xilinx/PYNQ/blob/v2.7.0/pynq/overlay.py#L310-L354
https://github.com/Xilinx/PYNQ/blob/v2.7.0/pynq/mmio.py#L55-L110
https://github.com/Xilinx/PYNQ/blob/v2.7.0/pynq/buffer.py#L82-L172
https://github.com/Xilinx/PYNQ/blob/v2.7.0/pynq/lib/dma.py#L119-L235
https://github.com/Xilinx/PYNQ/blob/v2.7.0/pynq/ps.py#L179-L191
https://github.com/Xilinx/PYNQ/blob/v2.7.0/pynq/ps.py#L404-L424
Overlay(download=False) parses the matching HWH without programming PL.
allocate supplies physical_address, flush, invalidate and freebuffer. The
stock DMA start/stop/wait loops have no deadline, including during lazy IP
construction, so this driver maps the DMA directly and polls with deadlines.
"""

import hashlib
import ctypes
import importlib
import math
import os
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np


WIDTH = 640
HEIGHT = 480
PIXELS = WIDTH * HEIGHT
FRAME_BYTES = PIXELS * 4
THRESHOLD = 64
CLOCK_HZ = 100_000_000
DMA_NAME = "vision_dma"
DEFAULT_BITFILE = Path(__file__).resolve().parents[1] / "output/vision_pl/vision_pl.bit"
WORD_DTYPE = np.dtype("<u4")
OUTPUT_PLANES = {"gray": 0, "magnitude": 1, "edge": 2}

MM2S_CONTROL, MM2S_STATUS, MM2S_ADDRESS, MM2S_LENGTH = 0x00, 0x04, 0x18, 0x28
S2MM_CONTROL, S2MM_STATUS, S2MM_ADDRESS, S2MM_LENGTH = 0x30, 0x34, 0x48, 0x58
HALTED, IDLE, RESET = 0x1, 0x2, 0x4
ERROR_MASK, IRQ_MASK = 0x4770, 0x7000

# A failed reset must never return memory still reachable by DMA to the allocator.
_QUARANTINED_BUFFERS = []


class VisionPLError(RuntimeError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class VisionPLTimeout(VisionPLError, TimeoutError):
    pass


def validate_frame(frame):
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy array")
    if frame.dtype != np.uint8:
        raise TypeError("frame dtype must be uint8 BGR")
    if frame.shape != (HEIGHT, WIDTH, 3):
        raise ValueError("frame shape must be (480, 640, 3) BGR; resize explicitly")


def _requested_outputs(outputs):
    """Validate the detached PL outputs a caller needs for this frame."""
    if outputs is None:
        return ("gray", "magnitude", "edge", "words")
    if isinstance(outputs, str):
        raise TypeError("outputs must be an iterable of output names, not a string")
    try:
        requested = tuple(outputs)
    except TypeError as exc:
        raise TypeError("outputs must be an iterable of output names") from exc
    if not requested:
        raise ValueError("outputs must request at least one output")
    if not all(isinstance(name, str) for name in requested):
        raise TypeError("outputs must contain only string output names")
    if len(set(requested)) != len(requested):
        raise ValueError("outputs must not contain duplicates")
    unknown = set(requested).difference(OUTPUT_PLANES).difference(("words",))
    if unknown:
        raise ValueError("unsupported output(s): " + ", ".join(sorted(unknown)))
    return requested


def pack_bgr(frame, out=None):
    """Pack even strided BGR frames into reusable uint32 BGR0 storage."""
    validate_frame(frame)
    if out is None:
        out = np.empty(PIXELS, dtype=WORD_DTYPE)
    if (not isinstance(out, np.ndarray) or out.dtype != WORD_DTYPE
            or out.shape != (PIXELS,) or not out.flags.c_contiguous
            or not out.flags.writeable):
        raise ValueError("out must be writable contiguous little-endian uint32[307200]")
    lanes = out.view(np.uint8).reshape(HEIGHT, WIDTH, 4)
    # Native channel copying avoids slow strided numpy stores into DMA memory.
    # This only packs BGR bytes; grayscale and Sobel still execute in PL.
    cv2 = importlib.import_module("cv2")
    cv2.mixChannels([frame], [lanes], [0, 0, 1, 1, 2, 2, -1, 3])
    return out


def _pack_outputs(gray, magnitude):
    words = gray.astype(WORD_DTYPE)
    words |= magnitude.astype(WORD_DTYPE) << 8
    words |= (magnitude >= THRESHOLD).astype(WORD_DTYPE) * (255 << 16)
    return words


def reference_words(frame):
    """Explicit CPU golden model for verification, never called by VisionPL."""
    validate_frame(frame)
    b, g, r = (frame[:, :, lane].astype(np.uint16) for lane in range(3))
    gray = ((77 * r + 150 * g + 29 * b + 128) >> 8).astype(np.uint8)
    p = gray.astype(np.int16)
    gx = (-p[:-2, :-2] + p[:-2, 2:] - 2 * p[1:-1, :-2]
          + 2 * p[1:-1, 2:] - p[2:, :-2] + p[2:, 2:])
    gy = (-p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]
          + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:])
    magnitude = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    magnitude[2:, 2:] = np.minimum(np.abs(gx) + np.abs(gy), 255).astype(np.uint8)
    return _pack_outputs(gray, magnitude)


def reference_cv2_words(frame):
    """OpenCV benchmark with the SAME integer gray, causal Sobel and L1 clip."""
    validate_frame(frame)
    cv2 = importlib.import_module("cv2")
    # cvtColor uses different integer coefficients; uint16 transform is exact here.
    weighted = cv2.transform(frame.astype(np.uint16),
                             np.array([[29, 150, 77]], dtype=np.float32))
    gray = ((weighted + 128) >> 8).astype(np.uint8)
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
    magnitude = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    magnitude[2:, 2:] = np.minimum(np.abs(gx[1:-1, 1:-1])
                                  + np.abs(gy[1:-1, 1:-1]), 255).astype(np.uint8)
    return _pack_outputs(gray, magnitude)


def _require_root():
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise PermissionError("Vision PL requires root on the Linux PYNQ board")


def _load_pynq():
    try:
        module = importlib.import_module("pynq")
        if not all(hasattr(module, name) for name in ("Overlay", "MMIO", "allocate")):
            raise ImportError("installed PYNQ runtime not found")
        return module
    except (ImportError, OSError) as exc:
        raise VisionPLError("PYNQ 2.7 is required on the board: {}".format(exc)) from exc


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_seconds(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0 or value > threading.TIMEOUT_MAX:
        raise ValueError(name + " must be finite and positive")
    return value


def _hwh_clock(path):
    try:
        root = ET.parse(path).getroot()
        ps = next(module for module in root.iter("MODULE")
                  if module.get("MODTYPE") == "processing_system7")
        port = next(port for port in ps.findall("./PORTS/PORT")
                    if port.get("NAME") == "FCLK_CLK0")
        parameters = {item.get("NAME"): item.get("VALUE")
                      for item in ps.findall("./PARAMETERS/PARAMETER")}
        declared_hz = int(port.attrib["CLKFREQUENCY"])
        actual_mhz = float(parameters["PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ"])
        metadata = {"declared_hz": declared_hz,
                    "source": parameters["PCW_FCLK0_PERIPHERAL_CLKSRC"],
                    "divisor0": int(parameters["PCW_FCLK0_PERIPHERAL_DIVISOR0"]),
                    "divisor1": int(parameters["PCW_FCLK0_PERIPHERAL_DIVISOR1"])}
    except (ET.ParseError, StopIteration, KeyError, TypeError, ValueError) as exc:
        raise VisionPLError("Missing/invalid HWH FCLK0 clock metadata") from exc
    if (declared_hz != CLOCK_HZ or not math.isfinite(actual_mhz)
            or abs(actual_mhz - CLOCK_HZ / 1e6) > .01):
        raise VisionPLError("HWH FCLK0 must declare the 100 MHz design constraint")
    return metadata


class VisionPL:
    """Single-owner simple DMA driver; process/close/status are serialized.

    process(frame) returns a dict with detached output words and byte planes. A fault
    disables further transfers: explicitly reload the vision overlay and create
    a new driver after resolving it. The hardware has no runtime threshold or
    image size register; these are fixed build-contract values.
    """

    def __init__(self, bitfile=DEFAULT_BITFILE, *, download=False,
                 timeout=1.0, poll_interval=0.0005):
        self.timeout = _positive_seconds(timeout, "timeout")
        self.poll_interval = _positive_seconds(poll_interval, "poll_interval")
        if not isinstance(download, bool):
            raise TypeError("download must be an explicit bool")
        self.bitfile = Path(bitfile).expanduser().resolve()
        self.hwh = self.bitfile.with_suffix(".hwh")
        if self.bitfile.suffix != ".bit":
            raise ValueError("overlay must be a .bit file with matching .hwh")
        for path in (self.bitfile, self.hwh):
            if not path.is_file():
                raise FileNotFoundError("Vision overlay artifact missing: {}".format(path))
        _require_root()
        self._lock = threading.Lock()
        self._tx = self._rx = self._mmio = None
        self._active = False
        self._closed = False
        self._fault = None
        self._frames = 0
        self._last_timings = None
        self._quarantined = False
        # The receive buffer is CPU-written only by allocation.  After its
        # first clean, DMA owns writes and invalidate() makes later reads safe.
        self._rx_needs_flush = True
        self._native_pack = None
        self._native_unpack = None
        transport_path = Path(__file__).with_name('libvision_transport.so')
        if transport_path.is_file() and os.environ.get('VISION_NATIVE_UNPACK', '1') == '1':
            transport = ctypes.CDLL(str(transport_path))
            transport.vision_unpack_planes.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                                       ctypes.c_void_p, ctypes.c_int]
            transport.vision_unpack_planes.restype = None
            self._native_unpack = transport.vision_unpack_planes
        native_path = Path(__file__).with_name('libhand_q16.so')
        if native_path.is_file():
            native = ctypes.CDLL(str(native_path))
            if hasattr(native,'vision_rgba_pack'):
                native.vision_rgba_pack.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
                native.vision_rgba_pack.restype = None
                self._native_pack = native.vision_rgba_pack
        pynq = _load_pynq()
        self._overlay = pynq.Overlay(str(self.bitfile), download=False)
        description = self._overlay.ip_dict.get(DMA_NAME)
        self._validate_dma(description)
        hwh_clock = _hwh_clock(self.hwh)
        self.fingerprint = {
            "bitfile": str(self.bitfile), "hwh": str(self.hwh),
            "bit_sha256": _sha256(self.bitfile), "hwh_sha256": _sha256(self.hwh),
            "dma_instance": DMA_NAME, "dma_type": description["type"],
            "dma_phys_addr": int(description["phys_addr"]),
            "dma_parameters": dict(description["parameters"]),
            "pynq_version": getattr(pynq, "__version__", "unknown"),
            "contract": {"width": WIDTH, "height": HEIGHT, "clock_hz": CLOCK_HZ,
                         "threshold": THRESHOLD, "dma_length_width": 23,
                         "input": "uint32 little-endian B,G,R,0",
                         "output": "uint32 little-endian gray,L1,edge255,0"},
            "identity_evidence": "artifact hashes, HWH and PYNQ active-overlay state; no PL ID register",
        }
        if download:
            self._overlay.download()
        if not self._overlay.is_loaded():
            raise VisionPLError("Vision overlay is not active; download requires download=True")
        try:
            before_mhz = float(pynq.Clocks.fclk0_mhz)
            if download:
                # HWH divisors assume Vivado's PLL rate; use the running PS PLL.
                pynq.Clocks.fclk0_mhz = CLOCK_HZ / 1e6
            fclk0_mhz = float(pynq.Clocks.fclk0_mhz)
        except Exception as exc:
            raise VisionPLError("Cannot configure/read FCLK0 clock: {}".format(exc)) from exc
        if not math.isfinite(fclk0_mhz) or abs(fclk0_mhz - CLOCK_HZ / 1e6) > .5:
            raise VisionPLError("FCLK0 must be 100 MHz; register-derived rate is {} MHz".format(
                fclk0_mhz))
        self.fingerprint["clock"] = {
            "hwh": hwh_clock,
            "configured_hz": CLOCK_HZ if download else None,
            "before_configure_hz": round(before_mhz * 1e6) if math.isfinite(before_mhz) else None,
            "actual_hz": round(fclk0_mhz * 1e6),
            "fclk0_hz": round(fclk0_mhz * 1e6),
            "evidence": "PYNQ Clocks.fclk0_mhz from PS registers, assuming 50 MHz board reference",
        }
        self._device_timestamp = getattr(self._overlay.device, "timestamp", None)
        self.fingerprint["overlay_timestamp"] = str(self._device_timestamp)
        self._mmio = pynq.MMIO(int(description["phys_addr"]),
                               int(description["addr_range"]), device=self._overlay.device)
        try:
            self._reset(time.monotonic() + self.timeout)
            for name in ("_tx", "_rx"):
                buffer = pynq.allocate(shape=(PIXELS,), dtype=WORD_DTYPE,
                                       target=self._overlay.device, cacheable=True)
                setattr(self, name, buffer)
                self._validate_buffer(buffer)
            self._start(time.monotonic() + self.timeout)
        except Exception:
            self._free_buffers()
            raise

    @staticmethod
    def _validate_dma(description):
        if not isinstance(description, dict) or not str(description.get("type", "")).startswith(
                "xilinx.com:ip:axi_dma:"):
            raise VisionPLError("HWH must contain AXI DMA instance vision_dma")
        parameters = description.get("parameters", {})
        expected = {"c_include_sg": 0, "c_include_mm2s": 1, "c_include_s2mm": 1,
                    "c_sg_length_width": 23, "c_addr_width": 32,
                    "c_m_axis_mm2s_tdata_width": 32, "c_s_axis_s2mm_tdata_width": 32}
        for name, value in expected.items():
            try:
                actual = int(parameters[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise VisionPLError("Missing/invalid DMA HWH parameter " + name) from exc
            if actual != value:
                raise VisionPLError("DMA {} must be {}, got {}".format(name, value, actual))
        for name in ("c_m_axi_mm2s_data_width", "c_m_axi_s2mm_data_width"):
            try:
                actual = int(parameters[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise VisionPLError("Missing/invalid DMA HWH parameter " + name) from exc
            if actual not in (32, 64):
                raise VisionPLError("DMA {} must be 32 or 64, got {}".format(name, actual))
        if int(parameters.get("c_micro_dma", 0)) or int(parameters.get("c_enable_multi_channel", 0)):
            raise VisionPLError("Micro DMA and multichannel DMA are unsupported")
        if int(description.get("addr_range", 0)) < 0x60:
            raise VisionPLError("DMA register range is too small")

    @staticmethod
    def _validate_buffer(buffer):
        if (buffer.dtype != WORD_DTYPE or buffer.shape != (PIXELS,)
                or not buffer.flags.c_contiguous or not buffer.flags.writeable):
            raise VisionPLError("allocate returned an invalid contiguous uint32 frame buffer")
        address = int(buffer.physical_address)
        if address <= 0 or address % 8 or address + FRAME_BYTES > 2 ** 32:
            raise VisionPLError("DMA buffers must be 8-byte aligned within 32-bit physical memory")

    def _pause(self, deadline, phase):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise VisionPLTimeout("Vision DMA timeout during " + phase, self._registers())
        time.sleep(min(self.poll_interval, remaining))

    def _registers(self):
        return {"mm2s_status": int(self._mmio.read(MM2S_STATUS)),
                "s2mm_status": int(self._mmio.read(S2MM_STATUS))}

    def _reset(self, deadline):
        self._mmio.write(MM2S_CONTROL, RESET)
        while True:
            reset = (self._mmio.read(MM2S_CONTROL) | self._mmio.read(S2MM_CONTROL)) & RESET
            status = self._registers()
            if not reset and all(value & HALTED for value in status.values()):
                self._active = False
                return
            self._pause(deadline, "reset")

    def _start(self, deadline):
        # Status SGIncld would contradict the HWH simple-mode description.
        if any(value & 0x8 for value in self._registers().values()):
            raise VisionPLError("DMA registers report scatter-gather mode", self._registers())
        self._mmio.write(S2MM_CONTROL, 1)
        self._mmio.write(MM2S_CONTROL, 1)
        while True:
            status = self._registers()
            self._check_errors(status)
            if all(not value & HALTED for value in status.values()):
                return
            self._pause(deadline, "start")

    @staticmethod
    def _check_errors(status):
        if any(value & ERROR_MASK for value in status.values()):
            raise VisionPLError("Vision DMA hardware error: {}".format(status), status)

    def _complete(self, deadline):
        while True:
            # Always inspect receive first, and detect either direction's errors.
            status = {"s2mm_status": int(self._mmio.read(S2MM_STATUS)),
                      "mm2s_status": int(self._mmio.read(MM2S_STATUS))}
            self._check_errors(status)
            if any(value & HALTED for value in status.values()):
                raise VisionPLError("Vision DMA halted during transfer", status)
            if all(value & IDLE for value in status.values()):
                received = int(self._mmio.read(S2MM_LENGTH))
                sent = int(self._mmio.read(MM2S_LENGTH))
                if received != FRAME_BYTES or sent != FRAME_BYTES:
                    raise VisionPLError("Short DMA frame: received {}, sent {}, expected {}".format(
                        received, sent, FRAME_BYTES), status)
                self._active = False
                return
            self._pause(deadline, "frame completion")

    def _free_buffers(self):
        if self._active:
            buffers = tuple(buffer for buffer in (self._tx, self._rx) if buffer is not None)
            if buffers:
                _QUARANTINED_BUFFERS.append(buffers)
                self._quarantined = True
        else:
            for buffer in (self._tx, self._rx):
                if buffer is not None:
                    buffer.freebuffer()
        self._tx = self._rx = None

    def _ensure_active_overlay(self):
        expected = vars(self._overlay).get("timestamp")
        from overlay_identity import live_timestamp
        compact,observed = live_timestamp(self.timeout)
        if compact:
            # Attach-only Overlay objects have an empty timestamp. Constructor
            # already checked their filename and saved the live device stamp.
            active = (observed == self._device_timestamp and
                      isinstance(expected,str) and expected in ('',observed))
        elif isinstance(expected, str) and expected:
            # PYNQ 2.7 is_loaded compares this very timestamp. Read the live
            # device once instead of issuing identical PL-server IPC twice.
            observed = getattr(self._overlay.device, "timestamp", None)
            active = expected == observed == self._device_timestamp
        else:
            active = (self._overlay.is_loaded()
                      and getattr(self._overlay.device, "timestamp", None) == self._device_timestamp)
        if not active:
            raise VisionPLError("Active overlay changed; refusing stale vision MMIO access")

    def process(self, frame, *, outputs=None):
        """Process BGR and return only the requested detached PL outputs.

        ``outputs=None`` preserves the full historical result.  Supplying a
        subset such as ``("gray", "edge")`` avoids copying unused planes.
        """
        validate_frame(frame)
        return self._process(frame, False, _requested_outputs(outputs))

    def process_rgba(self, frame, *, outputs=None):
        """Pack browser RGBA directly; all image algorithms still run in PL."""
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise TypeError("frame dtype must be uint8 RGBA")
        if frame.shape != (HEIGHT, WIDTH, 4):
            raise ValueError("frame shape must be (480, 640, 4) RGBA")
        return self._process(frame, True, _requested_outputs(outputs))

    def _process(self, frame, rgba, outputs):
        started = time.perf_counter()
        if not self._lock.acquire(timeout=self.timeout):
            raise VisionPLTimeout("Vision driver lock acquisition timed out")
        try:
            acquired = time.perf_counter()
            if self._closed or self._fault:
                raise VisionPLError("Vision driver is closed or faulted", self._fault)
            self._ensure_active_overlay()
            validated = time.perf_counter()
            # Pack in cached RAM, then stream contiguous bytes into DMA RAM.
            # Strided per-channel writes to uncached CMA memory are expensive.
            if rgba and self._native_pack is not None:
                source = np.ascontiguousarray(frame)
                self._native_pack(source.ctypes.data,np.asarray(self._tx).ctypes.data,PIXELS)
            elif rgba:
                cv2 = importlib.import_module("cv2")
                staging = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
                staging[:, :, 3] = 0
                ctypes.memmove(np.asarray(self._tx).ctypes.data, staging.ctypes.data, FRAME_BYTES)
            else:
                staging = pack_bgr(frame)
                ctypes.memmove(np.asarray(self._tx).ctypes.data, staging.ctypes.data, FRAME_BYTES)
            packed = time.perf_counter()
            self._tx.flush()
            if self._rx_needs_flush:
                self._rx.flush()
                self._rx_needs_flush = False
            flushed = time.perf_counter()
            deadline = time.monotonic() + self.timeout
            status = self._registers()
            self._check_errors(status)
            if any(value & HALTED for value in status.values()):
                raise VisionPLError("Vision DMA is halted", status)
            for offset in (S2MM_STATUS, MM2S_STATUS):
                self._mmio.write(offset, IRQ_MASK)
            for offset, buffer in ((S2MM_ADDRESS, self._rx), (MM2S_ADDRESS, self._tx)):
                self._mmio.write(offset, int(buffer.physical_address))
                self._mmio.write(offset + 4, 0)
            self._active = True
            self._mmio.write(S2MM_LENGTH, FRAME_BYTES)
            self._mmio.write(MM2S_LENGTH, FRAME_BYTES)
            self._complete(deadline)
            completed = time.perf_counter()
            self._rx.invalidate()
            invalidated = time.perf_counter()
            lanes = self._rx.view(np.uint8).reshape(HEIGHT, WIDTH, 4)
            result = {}
            if self._native_unpack is not None and 'gray' in outputs and 'edge' in outputs:
                result['gray'] = np.empty((HEIGHT, WIDTH), np.uint8)
                result['edge'] = np.empty((HEIGHT, WIDTH), np.uint8)
                self._native_unpack(np.asarray(self._rx).ctypes.data,
                                    result['gray'].ctypes.data, result['edge'].ctypes.data, PIXELS)
            if "words" in outputs:
                result["words"] = np.array(self._rx, dtype=WORD_DTYPE,
                                           copy=True).reshape(HEIGHT, WIDTH)
            for name, lane in OUTPUT_PLANES.items():
                if name in outputs and name not in result:
                    result[name] = lanes[:, :, lane].copy()
            copied = time.perf_counter()
            timings = {"lock_wait_ms": (acquired - started) * 1000,
                       "overlay_validation_ms": (validated - acquired) * 1000,
                       "pack_ms": (packed - validated) * 1000,
                       "cache_flush_ms": (flushed - packed) * 1000,
                       "dma_ms": (completed - flushed) * 1000,
                       "cache_invalidate_ms": (invalidated - completed) * 1000,
                       "copy_ms": (copied - invalidated) * 1000,
                       "total_ms": (copied - started) * 1000}
            self._frames += 1
            self._last_timings = timings
            provenance = {"backend": "pl", "cpu_fallback": False,
                          "frame_sequence": self._frames,
                          "hardware_fingerprint": self.fingerprint}
            result.update({"timings": timings, "timings_ms": timings,
                           "hardware": self.fingerprint,
                           "hardware_metadata": self.fingerprint,
                           "provenance": provenance})
            return result
        except Exception as exc:
            if not self._closed and self._fault is None:
                self._fault = {"type": type(exc).__name__, "message": str(exc),
                               "diagnostics": getattr(exc, "diagnostics", {})}
                if self._active:
                    try:
                        self._ensure_active_overlay()
                        self._reset(time.monotonic() + min(self.timeout, 0.1))
                        self._fault["dma_reset_completed"] = True
                    except Exception as reset_exc:
                        self._fault["dma_reset_completed"] = False
                        self._fault["reset_error"] = str(reset_exc)
            raise
        finally:
            self._lock.release()

    process_bgr = process

    def status(self):
        if not self._lock.acquire(timeout=self.timeout):
            raise VisionPLTimeout("Vision driver status lock acquisition timed out")
        try:
            if not self._closed and self._fault is None:
                try:
                    self._ensure_active_overlay()
                except VisionPLError as exc:
                    self._fault = {"type": type(exc).__name__, "message": str(exc),
                                   "diagnostics": exc.diagnostics}
            return {"backend": "pl", "cpu_fallback": False,
                    "ready": not self._closed and self._fault is None,
                    "closed": self._closed, "frames": self._frames,
                    "error": self._fault, "last_timings_ms": self._last_timings,
                    "buffers_quarantined": self._quarantined,
                    "hardware_fingerprint": self.fingerprint,
                    "hardware": self.fingerprint}
        finally:
            self._lock.release()

    def close(self):
        if not self._lock.acquire(timeout=self.timeout):
            raise VisionPLTimeout("Vision driver close lock acquisition timed out")
        try:
            if self._closed:
                return
            try:
                if self._active:
                    self._ensure_active_overlay()
                    self._reset(time.monotonic() + min(self.timeout, 0.1))
            finally:
                self._free_buffers()
                self._closed = True
        finally:
            self._lock.release()

    snapshot = status

    @property
    def faulted(self):
        return self.status()["error"] is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

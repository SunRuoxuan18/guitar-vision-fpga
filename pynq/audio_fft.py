"""Coherent snapshots using the verified AUD4 map or an explicit override."""

from dataclasses import dataclass
from ctypes import c_uint32, c_size_t, POINTER, cast, addressof, CDLL
from pathlib import Path
import mmap
import os
import threading
import time
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class FFTRegisterMap:
    control_offset: int
    sequence_offset: int
    status_offset: int
    freeze_mask: int
    freeze_ack_mask: int
    power_base_offset: Optional[int] = None
    bin_index_offset: Optional[int] = None
    bin_power_offset: Optional[int] = None
    ready_mask: int = 0
    core_id_offset: Optional[int] = None
    core_id_value: Optional[int] = None
    busy_mask: int = 0
    error_mask: int = 0
    fft_size_offset: Optional[int] = None
    sample_rate_offset: Optional[int] = None
    drops_offset: Optional[int] = None
    peak_count_offset: Optional[int] = None
    peak_base_offset: Optional[int] = None
    max_peaks: int = 24
    power_shift: int = 20
    power_scale_offset: Optional[int] = None
    window_denominator_offset: Optional[int] = None
    window: str = "hann"
    sample_rate: int = 48000
    fft_size: int = 4096

    def __post_init__(self):
        if self.sample_rate != 48000 or self.fft_size != 4096:
            raise ValueError("This driver expects 4096-point FFT frames at 48 kHz")
        offsets = (self.control_offset, self.sequence_offset, self.status_offset,
                   self.power_base_offset, self.bin_index_offset,
                   self.bin_power_offset, self.core_id_offset, self.fft_size_offset,
                   self.sample_rate_offset, self.drops_offset, self.peak_count_offset,
                   self.peak_base_offset, self.power_scale_offset, self.window_denominator_offset)
        if any(offset is not None and (offset < 0 or offset % 4) for offset in offsets):
            raise ValueError("Register offsets must be nonnegative and 32-bit aligned")
        if self.freeze_mask <= 0 or self.freeze_ack_mask <= 0 or min(self.ready_mask, self.busy_mask, self.error_mask) < 0:
            raise ValueError("Explicit freeze request and acknowledgement masks are required")
        indexed = self.bin_index_offset is not None and self.bin_power_offset is not None
        if (self.power_base_offset is not None) == indexed:
            raise ValueError("Choose contiguous power bins OR an indexed power port")
        if (self.bin_index_offset is None) != (self.bin_power_offset is None):
            raise ValueError("Both indexed-port offsets are required")
        if (self.core_id_offset is None) != (self.core_id_value is None):
            raise ValueError("Provide both core ID offset and expected value")
        if (self.peak_count_offset is None) != (self.peak_base_offset is None):
            raise ValueError("Provide both peak table and count offsets")
        if self.max_peaks < 1 or self.power_shift < 0 or self.window not in ("hann", "rectangular"):
            raise ValueError("Invalid FFT metadata")


# AUD4 map verified by the PL checks and hardware identity read.
AUD4_REGISTERS = FFTRegisterMap(
    core_id_offset=0x00, core_id_value=0x41554434,
    status_offset=0x04, sequence_offset=0x08, control_offset=0x0C,
    freeze_mask=0x1, busy_mask=0x1, ready_mask=0x2,
    freeze_ack_mask=0x4, error_mask=0x8,
    fft_size_offset=0x10, sample_rate_offset=0x14, drops_offset=0x18,
    peak_count_offset=0x1C, peak_base_offset=0x100, power_base_offset=0x1000,
    power_scale_offset=0x20, window_denominator_offset=0x24,
)

AUD4_BASE_ADDRESS = 0x43C00000
AUD4_ADDRESS_RANGE = 0x4000


class DevMemMMIO:
    """Minimal Linux MMIO using /dev/mem, without PYNQ device discovery or XRT."""

    def __init__(self, base_address, address_range=AUD4_ADDRESS_RANGE, device="/dev/mem"):
        if base_address < 0 or base_address % 4 or address_range < 4 or address_range % 4:
            raise ValueError("MMIO base and range must be nonnegative and word aligned")
        page_size = mmap.PAGESIZE
        page_base = base_address // page_size * page_size
        self._page_offset = base_address - page_base
        self._range = address_range
        self._mapping = None
        self._native_words = None
        descriptor = os.open(device, os.O_RDWR | getattr(os, "O_SYNC", 0))
        try:
            self._mapping = mmap.mmap(descriptor, self._page_offset + address_range,
                                      flags=getattr(mmap, "MAP_SHARED", 1),
                                      prot=getattr(mmap, "PROT_READ", 1) | getattr(mmap, "PROT_WRITE", 2),
                                      offset=page_base)
        finally:
            os.close(descriptor)
        if os.environ.get("FRETLENS_AUDIO_NATIVE") == "1":
            try:
                library = CDLL(str(Path(__file__).with_name("libaudio_fit.so")))
                self._native_words = library.audio_mmio_words
                self._native_words.argtypes = [POINTER(c_uint32), POINTER(c_uint32), c_size_t]
                self._native_words.restype = None
            except Exception:
                self.close()
                raise

    def _offset(self, offset):
        if self._mapping is None:
            raise RuntimeError("MMIO mapping is closed")
        if offset < 0 or offset % 4 or offset + 4 > self._range:
            raise ValueError("MMIO register access is unaligned or outside the mapping")
        return self._page_offset + offset

    def read(self, offset):
        mapped_offset = self._offset(offset)
        # Packed-byte access can fault on ARM device memory; use an aligned native word.
        return c_uint32.from_buffer(self._mapping, mapped_offset).value

    def write(self, offset, value):
        mapped_offset = self._offset(offset)
        c_uint32.from_buffer(self._mapping, mapped_offset).value = int(value) & 0xFFFFFFFF

    def read_words(self, offset, count):
        if not isinstance(count, int) or count < 1:
            raise ValueError("MMIO count must be a positive integer")
        start = self._offset(offset)
        self._offset(offset + (count-1)*4)
        result = np.empty(count, dtype=np.uint32)
        if self._native_words is None:
            for index in range(count):
                result[index] = self.read(offset+index*4)
        else:
            source = c_uint32.from_buffer(self._mapping, start)
            self._native_words(cast(addressof(source), POINTER(c_uint32)),
                               result.ctypes.data_as(POINTER(c_uint32)), count)
        return result

    def close(self):
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None


@dataclass(frozen=True)
class FFTSnapshot:
    sequence: int
    power: np.ndarray
    sample_rate: int = 48000
    fft_size: int = 4096
    metadata: Optional[Dict] = None


class AudioFFT:
    """Freeze, await acknowledgement, read unsigned bins, verify sequence, release.

    Supplying MMIO explicitly supports offline tests without importing PYNQ.
    The AUD4 map checks hardware identity, FFT length and sample rate before use.
    """

    def __init__(self, overlay=None, ip_name: Optional[str] = None,
                 register_map: FFTRegisterMap = AUD4_REGISTERS, mmio=None,
                 timeout: float = 0.25, retries: int = 3,
                 base_address: Optional[int] = None, address_range: int = AUD4_ADDRESS_RANGE,
                 dev_mem: bool = False):
        if register_map is None:
            raise ValueError("An FFTRegisterMap is required")
        if timeout <= 0 or retries < 1:
            raise ValueError("Positive timeout and retry count are required")
        self._owns_mmio = mmio is None
        if mmio is None:
            if base_address is not None:
                description = {"phys_addr": base_address, "addr_range": address_range}
            elif overlay is not None and ip_name is not None:
                description = overlay.ip_dict[ip_name]
            else:
                raise ValueError("Supply direct base_address, MMIO, or existing overlay and IP name")
            required = register_map.power_base_offset
            if required is not None and description["addr_range"] < required + register_map.fft_size // 2 * 4:
                raise ValueError("The IP address range does not contain all FFT bins")
            if dev_mem:
                mmio = DevMemMMIO(description["phys_addr"], description["addr_range"])
            else:
                from pynq import MMIO
                mmio = MMIO(description["phys_addr"], description["addr_range"])
        self.mmio = mmio
        self.registers = register_map
        self.timeout = timeout
        self.retries = retries
        self._lock = threading.Lock()
        try:
            if register_map.core_id_offset is not None:
                if self.mmio.read(register_map.core_id_offset) != register_map.core_id_value:
                    raise RuntimeError("FFT core ID does not match the supplied register map")
            for offset, expected, name in ((register_map.fft_size_offset, register_map.fft_size, "FFT length"),
                                           (register_map.sample_rate_offset, register_map.sample_rate, "sample rate"),
                                           (register_map.power_scale_offset, 1 << register_map.power_shift, "power scale"),
                                           (register_map.window_denominator_offset, register_map.fft_size - 1, "Hann denominator")):
                if offset is not None and int(self.mmio.read(offset)) != expected:
                    raise RuntimeError("FFT core {} does not match the register contract".format(name))
        except Exception:
            self.close()
            raise

    def close(self):
        if self._owns_mmio and hasattr(self.mmio, "close"):
            self.mmio.close()

    def _valid_frozen(self, status, require_ready=True):
        registers = self.registers
        if status & registers.error_mask:
            raise RuntimeError("FFT core asserted its error status bit")
        return ((status & registers.freeze_ack_mask) == registers.freeze_ack_mask
                and not (status & registers.busy_mask)
                and (not require_ready or not registers.ready_mask or
                     (status & registers.ready_mask) == registers.ready_mask))

    def _await_frozen(self, require_ready=True):
        registers = self.registers
        deadline = time.monotonic() + self.timeout
        while True:
            status = int(self.mmio.read(registers.status_offset))
            if self._valid_frozen(status, require_ready):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("FFT freeze/ready/busy-clear status timed out")
            time.sleep(0.001)

    def set_test_mode(self, mode: int) -> None:
        """Safely select live input (0), A4 (1), pure C major (2), or A minor (3).

        PL source_changed resets capture banks. Hold freeze through the reset,
        drain any active transform, then give the reset at least 100 microseconds.
        """
        if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 3:
            raise ValueError("test mode must be an integer in 0..3")
        registers = self.registers
        with self._lock:
            control = int(self.mmio.read(registers.control_offset))
            already_frozen = bool(control & registers.freeze_mask)
            try:
                self.mmio.write(registers.control_offset, control | registers.freeze_mask)
                self._await_frozen(require_ready=False)
                frozen_control = int(self.mmio.read(registers.control_offset))
                self.mmio.write(registers.control_offset,
                                (frozen_control & ~0x6) | (mode << 1) | registers.freeze_mask)
                time.sleep(0.0001)
            finally:
                if not already_frozen:
                    current = int(self.mmio.read(registers.control_offset))
                    self.mmio.write(registers.control_offset, current & ~registers.freeze_mask)

    def read_snapshot_if_new(self, previous_sequence):
        # An unchanged sequence needs no freeze or 2048-bin MMIO transfer.
        # A publication racing this read is picked up on the next bounded poll.
        if (previous_sequence is not None
                and int(self.mmio.read(self.registers.sequence_offset)) & 0xFFFFFFFF
                == previous_sequence):
            return None
        return self.read_snapshot()

    def read_snapshot(self) -> FFTSnapshot:
        registers = self.registers
        with self._lock:
            for _ in range(self.retries):
                control = int(self.mmio.read(registers.control_offset))
                already_frozen = bool(control & registers.freeze_mask)
                try:
                    self.mmio.write(registers.control_offset, control | registers.freeze_mask)
                    self._await_frozen()
                    before = int(self.mmio.read(registers.sequence_offset)) & 0xFFFFFFFF
                    power = np.empty(registers.fft_size // 2, dtype=np.uint32)
                    bulk = registers.power_base_offset is not None and hasattr(self.mmio, "read_words")
                    if bulk:
                        power[:] = self.mmio.read_words(registers.power_base_offset, len(power))
                    for index in (() if bulk else range(len(power))):
                        if registers.power_base_offset is not None:
                            offset = registers.power_base_offset + index * 4
                        else:
                            self.mmio.write(registers.bin_index_offset, index)
                            offset = registers.bin_power_offset
                        power[index] = int(self.mmio.read(offset)) & 0xFFFFFFFF
                    peaks = []
                    if registers.peak_count_offset is not None:
                        count = min(int(self.mmio.read(registers.peak_count_offset)), registers.max_peaks)
                        for index in range(count):
                            offset = registers.peak_base_offset + index * 8
                            bin_index = int(self.mmio.read(offset))
                            if not 0 <= bin_index < len(power):
                                raise RuntimeError("FFT peak table contains an invalid bin")
                            peaks.append({"bin": bin_index,
                                          "frequency_hz": bin_index * registers.sample_rate / registers.fft_size,
                                          "power": int(self.mmio.read(offset + 4)) & 0xFFFFFFFF})
                    drops = None if registers.drops_offset is None else int(self.mmio.read(registers.drops_offset)) & 0xFFFFFFFF
                    after = int(self.mmio.read(registers.sequence_offset)) & 0xFFFFFFFF
                    status = int(self.mmio.read(registers.status_offset))
                    if before == after and self._valid_frozen(status):
                        power.flags.writeable = False
                        metadata = {"fft_size": registers.fft_size,
                                    "sample_rate_hz": registers.sample_rate,
                                    "bin_count": len(power), "bin_spacing_hz": registers.sample_rate / registers.fft_size,
                                    "window": registers.window, "power_format": "uint32_linear",
                                    "power_shift": registers.power_shift,
                                    "power_scale_factor": 1 << registers.power_shift,
                                    "window_denominator": registers.fft_size - 1 if registers.window == "hann" else None,
                                    "window_format": "Q15" if registers.window == "hann" else None,
                                    "power_scale": "approximately abs(unscaled_fft(int16_pcm))**2 >> {}".format(registers.power_shift),
                                    "contract": "AUD4" if registers == AUD4_REGISTERS else "supplied_register_map",
                                    "drops": drops, "peaks": peaks}
                        if registers == AUD4_REGISTERS:
                            metadata["test_mode"] = (int(self.mmio.read(registers.control_offset)) >> 1) & 3
                            metadata["analysis_test_mode"] = metadata["test_mode"]
                        return FFTSnapshot(after, power, metadata=metadata)
                finally:
                    if not already_frozen:
                        current = int(self.mmio.read(registers.control_offset))
                        self.mmio.write(registers.control_offset, current & ~registers.freeze_mask)
            raise RuntimeError("FFT sequence changed during every frozen read; snapshot discarded")

"""Small PYNQ driver for the audio status AXI-Lite core."""

from dataclasses import dataclass
from typing import Dict

from pynq import MMIO


NOTE_NAMES = (
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
)


@dataclass(frozen=True)
class AudioResult:
    frequency_hz: float
    note_name: str
    midi_note: int
    target_frequency_hz: float
    frequency_error_hz: float
    peak_level: int
    signal_active: bool
    sequence: int


class AudioStatus:
    CONTROL = 0x00
    STATUS = 0x04
    FREQUENCY = 0x08
    NOTE_INFO = 0x0C
    TARGET_FREQUENCY = 0x10
    FREQUENCY_ERROR = 0x14
    PEAK_LEVEL = 0x18
    FRAME_NUMBER = 0x1C
    RESULT_SEQUENCE = 0x20
    FRAME_SEQUENCE = 0x24
    CORE_ID = 0x28
    CORE_VERSION = 0x2C

    def __init__(self, overlay, ip_name: str = "audio_status_axi_lite_0"):
        description: Dict = overlay.ip_dict[ip_name]
        self.mmio = MMIO(description["phys_addr"], description["addr_range"])

        if self.mmio.read(self.CORE_ID) != 0x41554430:
            raise RuntimeError("The mapped IP is not the AUD0 audio status core")

    @staticmethod
    def _signed32(value: int) -> int:
        return value - (1 << 32) if value & 0x80000000 else value

    def configure(self, use_test_tone: bool, gain_shift: int = 0) -> None:
        if not 0 <= gain_shift <= 7:
            raise ValueError("gain_shift must be in the range 0..7")

        control = int(use_test_tone) | (gain_shift << 1)
        self.mmio.write(self.CONTROL, control)

    def clear_events(self) -> None:
        self.mmio.write(self.STATUS, 0x6)

    def read_result(self) -> AudioResult:
        status = self.mmio.read(self.STATUS)
        frequency_x100 = self.mmio.read(self.FREQUENCY)
        note_info = self.mmio.read(self.NOTE_INFO)
        target_x100 = self.mmio.read(self.TARGET_FREQUENCY)
        error_x100 = self._signed32(self.mmio.read(self.FREQUENCY_ERROR))

        midi_note = note_info & 0x7F
        note_class = (note_info >> 8) & 0x0F
        octave = (note_info >> 12) & 0x0F

        return AudioResult(
            frequency_hz=frequency_x100 / 100.0,
            note_name=f"{NOTE_NAMES[note_class]}{octave}",
            midi_note=midi_note,
            target_frequency_hz=target_x100 / 100.0,
            frequency_error_hz=error_x100 / 100.0,
            peak_level=self.mmio.read(self.PEAK_LEVEL) & 0xFFFF,
            signal_active=bool(status & 0x1),
            sequence=self.mmio.read(self.RESULT_SEQUENCE),
        )

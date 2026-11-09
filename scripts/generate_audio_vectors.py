"""Generate deterministic Hann coefficients and independent FFT test vectors."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N = 4096
FS = 48000


def main():
    window = np.rint(np.hanning(N) * 32767).astype(np.int64)
    (ROOT / "rtl" / "hann4096.mem").write_text(
        "".join(f"{int(value):04x}\n" for value in window), encoding="ascii")
    folder = ROOT / "sim" / "vectors"
    folder.mkdir(parents=True, exist_ok=True)
    cases = {
        "silence": [],
        "a4": [(440.0, 12000)],
        "c_major": [(261.6256, 4000), (329.6276, 4000), (391.9954, 4000)],
        "a_minor": [(220.0, 4000), (261.6256, 4000), (329.6276, 4000)],
    }
    samples = np.arange(N) / FS
    metadata = {}
    for name, notes in cases.items():
        pcm = np.rint(sum((amp * np.sin(2 * np.pi * freq * samples)
                          for freq, amp in notes), np.zeros(N))).astype(np.int64)
        windowed = (pcm * window) >> 15
        spectrum = np.fft.fft(windowed)
        # Unscaled fixed-point IP is checked with quantization tolerance.
        reference = np.rint(np.column_stack((spectrum.real, spectrum.imag))).astype(np.int64)
        (folder / f"{name}_pcm.mem").write_text(
            "".join(f"{int(value) & 0xffff:04x}\n" for value in pcm), encoding="ascii")
        (folder / f"{name}_fft.mem").write_text(
            "".join(f"{int(im) & 0xffffffff:08x}{int(re) & 0xffffffff:08x}\n"
                    for re, im in reference), encoding="ascii")
        metadata[name] = {"frequencies": [f for f, _ in notes],
                          "max_fft_magnitude": float(np.abs(spectrum).max())}
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="ascii")
    print("GENERATED: Hann4096 and four independent NumPy FFT vectors")


if __name__ == "__main__":
    main()

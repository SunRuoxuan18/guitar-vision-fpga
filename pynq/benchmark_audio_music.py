"""Measure actual ARM inference time from a coherent PL diagnostic spectrum."""

import json
from pathlib import Path
import time

from audio_fft import AudioFFT, AUD4_BASE_ADDRESS
from audio_music import MusicConfig, MusicRecognizer


def main():
    driver = AudioFFT(base_address=AUD4_BASE_ADDRESS, dev_mem=True)
    try:
        driver.set_test_mode(2)
        sequence = driver.read_snapshot().sequence
        for _ in range(3):
            deadline = time.monotonic() + 5
            while True:
                time.sleep(0.09)
                snapshot = driver.read_snapshot()
                if snapshot.sequence != sequence:
                    sequence = snapshot.sequence
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("PL did not publish a new spectrum")
        recognizer = MusicRecognizer(MusicConfig(power_scale=1048576.0))
        times_ms = []
        for _ in range(6):
            started = time.perf_counter()
            result = recognizer.analyze(snapshot.power, snapshot.sequence)
            times_ms.append((time.perf_counter() - started) * 1000)
        if result["chord"]["label"] != "C":
            raise AssertionError("Expected C major in diagnostic mode 2")
        report = {"source": "actual_PL_C_major", "cold_ms": times_ms[0],
                  "warm_ms": times_ms[1:], "warm_mean_ms": sum(times_ms[1:]) / 5,
                  "warm_max_ms": max(times_ms[1:]), "notes": result["notes"],
                  "chord": result["chord"], "frame_window_ms": 4096 / 48,
                  "includes_capture_or_web_latency": False}
        encoded = json.dumps(report, indent=2, allow_nan=False)
        Path("board_music_benchmark.json").write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
    finally:
        driver.set_test_mode(0)
        driver.close()


if __name__ == "__main__":
    main()

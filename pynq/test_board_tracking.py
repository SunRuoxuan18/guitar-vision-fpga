"""Actual PL -> PS temporal tracking check; stop the HTTP service first."""

import json
from pathlib import Path
import time

from audio_fft import AudioFFT, AUD4_BASE_ADDRESS
from audio_music import MusicConfig, MusicRecognizer
from audio_music_server import MusicState


def main():
    driver = AudioFFT(base_address=AUD4_BASE_ADDRESS, dev_mem=True)
    state = MusicState(driver, MusicRecognizer(MusicConfig(power_scale=1048576.0)))
    checks = []
    try:
        for mode, label in ((1, None), (2, "C"), (3, "Am")):
            driver.set_test_mode(mode)
            previous = driver.read_snapshot().sequence
            for _ in range(3):
                deadline = time.monotonic() + 5
                while True:
                    time.sleep(.09)
                    sequence = driver.read_snapshot().sequence
                    if sequence != previous:
                        previous = sequence
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("No new diagnostic frame")
            state.update()
            first = state.snapshot()
            time.sleep(.1)
            state.update()
            second = state.snapshot()
            assert first["status"] == second["status"] == "recognized", second
            assert second["fft"]["analysis_test_mode"] == mode, second
            assert first["stable_chord"]["label"] is None, first
            assert second["stable_chord"]["label"] == label, second
            assert first["sequence"] != second["sequence"], second
            if label:
                assert first["tracking"]["chord_status"] == "pending", first
                assert second["tracking"]["chord_status"] == "confirmed", second
                assert second["scale_evidence"]["status"] == "ambiguous", second
                assert all(len(group["modes"]) == 7 for group in second["scale_evidence"]["groups"])
            checks.append({"mode": mode, "passed": True,
                           "first_sequence": first["sequence"], "last_sequence": second["sequence"],
                           "raw_chord": second["raw_chord"], "stable_chord": second["stable_chord"],
                           "tracking": second["tracking"], "scale_evidence": second["scale_evidence"],
                           "inference_ms": second["inference_ms"]})
        report = {"passed": True, "source": "actual_PL_to_PS", "checks": checks,
                  "hardware_id": "41554434", "algorithm_version": "harmonic_tracking_v3"}
        Path("board_music_tracking.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("PASS: actual PL A4/C/Am notes, temporal confirmation, mode reset and modal alternatives")
    finally:
        driver.set_test_mode(0)
        driver.close()


if __name__ == "__main__":
    main()

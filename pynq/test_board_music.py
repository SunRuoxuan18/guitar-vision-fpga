"""Parent-run real AUD4 FFT -> PS inference smoke check; restores live mode."""

import argparse
import json
from pathlib import Path
import sys
import time

from audio_fft import AUD4_BASE_ADDRESS, AUD4_ADDRESS_RANGE, AudioFFT
from audio_music import MusicConfig, MusicRecognizer


def run_smoke(driver, timeout=30.0, settle_frames=3, recognizer=None):
    if timeout <= 0 or settle_frames < 3:
        raise ValueError("Use a positive timeout and at least three new frames")
    recognizer = recognizer or MusicRecognizer(MusicConfig(power_scale=1048576.0))
    report = {"passed": False, "started_at": time.time(), "checks": [],
              "restored_live_mode": False, "error": None}
    try:
        for mode, expected_notes, expected_chord in ((1, [69], None),
                                                      (2, [60, 64, 67], "C"),
                                                      (3, [57, 60, 64], "Am")):
            driver.set_test_mode(mode)
            deadline = time.monotonic() + timeout
            snapshot = driver.read_snapshot()
            first_sequence = snapshot.sequence
            previous_sequence = snapshot.sequence
            observed_changes = 0
            while observed_changes < settle_frames:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Mode {} did not publish {} new frames".format(mode, settle_frames))
                time.sleep(0.09)
                snapshot = driver.read_snapshot()
                delta = (snapshot.sequence - previous_sequence) & 0xFFFFFFFF
                if delta:
                    if delta >= 0x80000000:
                        raise RuntimeError("FFT sequence moved backward during mode settling")
                    # Count observed advances, not just a large first sequence jump.
                    observed_changes += 1
                    previous_sequence = snapshot.sequence
            state = recognizer.analyze(snapshot.power, snapshot.sequence)
            actual_notes = [note["midi_note"] for note in state["notes"]]
            actual_chord = state["chord"]["label"]
            metadata = snapshot.metadata or {}
            actual_mode = metadata.get("analysis_test_mode", metadata.get("test_mode"))
            passed = (state["status"] == "recognized" and actual_notes == expected_notes
                      and actual_chord == expected_chord and actual_mode == mode)
            check = {"mode": mode, "expected_notes": expected_notes,
                     "expected_chord": expected_chord, "first_sequence": first_sequence,
                     "last_sequence": snapshot.sequence, "observed_new_frames": observed_changes,
                     "passed": passed, "fft": metadata, "analysis": state}
            report["checks"].append(check)
            if not passed:
                raise AssertionError("Mode {}: notes {}, chord {}, mode {}; expected {}, {}, {}".format(
                    mode, actual_notes, actual_chord, actual_mode, expected_notes, expected_chord, mode))
        report["passed"] = True
    except Exception as error:
        report["error"] = "{}: {}".format(type(error).__name__, error)
    finally:
        try:
            driver.set_test_mode(0)
            report["restored_live_mode"] = True
        except Exception as error:
            report["passed"] = False
            report["restore_error"] = "{}: {}".format(type(error).__name__, error)
        report["finished_at"] = time.time()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-mem", action="store_true", help="Use mmap /dev/mem instead of PYNQ/XRT")
    parser.add_argument("--base-address", type=lambda value: int(value, 0), default=AUD4_BASE_ADDRESS)
    parser.add_argument("--address-range", type=lambda value: int(value, 0), default=AUD4_ADDRESS_RANGE)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--settle-frames", type=int, default=3)
    parser.add_argument("--minimum-rms", type=float, default=0.001)
    parser.add_argument("--output", default="board_music_smoke.json")
    args = parser.parse_args(argv)
    driver = None
    try:
        driver = AudioFFT(base_address=args.base_address, address_range=args.address_range,
                          dev_mem=args.dev_mem)
        recognizer = MusicRecognizer(MusicConfig(power_scale=1048576.0, minimum_rms=args.minimum_rms))
        report = run_smoke(driver, args.timeout, args.settle_frames, recognizer)
    except Exception as error:
        report = {"passed": False, "error": "{}: {}".format(type(error).__name__, error),
                  "checks": [], "restored_live_mode": False}
    finally:
        if driver is not None:
            driver.close()
    encoded = json.dumps(report, indent=2, allow_nan=False)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

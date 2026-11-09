"""Compare PS versions on identical actual PL spectra, excluding I/O latency."""

import importlib.util
import json
from pathlib import Path
import time

from audio_fft import AudioFFT, AUD4_BASE_ADDRESS
from audio_music import MusicConfig, MusicRecognizer
from music_tracking import MusicTracker


def main():
    baseline_path = Path(__file__).with_name("audio_music_baseline.py")
    spec = importlib.util.spec_from_file_location("music_baseline", str(baseline_path))
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    recognizers = {"baseline": baseline.MusicRecognizer(baseline.MusicConfig(power_scale=1048576.0)),
                   "upgrade": MusicRecognizer(MusicConfig(power_scale=1048576.0))}
    driver = AudioFFT(base_address=AUD4_BASE_ADDRESS, dev_mem=True)
    tracker = MusicTracker()
    checks = []
    cold = {}
    try:
        for mode in (2, 3, 1):
            driver.set_test_mode(mode)
            previous = driver.read_snapshot().sequence
            for _ in range(3):
                deadline = time.monotonic() + 5
                while True:
                    time.sleep(.09)
                    snapshot = driver.read_snapshot()
                    if snapshot.sequence != previous:
                        previous = snapshot.sequence
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("No new diagnostic frame")
            timings = {name: [] for name in recognizers}
            results = {}
            for name, recognizer in recognizers.items():
                started = time.perf_counter()
                results[name] = recognizer.analyze(snapshot.power, snapshot.sequence)
                if name not in cold:
                    cold[name] = (time.perf_counter() - started) * 1000
            for _ in range(5):
                for name, recognizer in recognizers.items():
                    started = time.perf_counter()
                    results[name] = recognizer.analyze(snapshot.power, snapshot.sequence)
                    timings[name].append((time.perf_counter() - started) * 1000)
            assert results["baseline"] == results["upgrade"], "Diagnostic output changed"
            tracker_times = []
            for sequence in range(10):
                tracked_input = dict(results["upgrade"], sequence=sequence)
                started = time.perf_counter()
                tracker.update(tracked_input, time.monotonic(), context=mode)
                tracker_times.append((time.perf_counter() - started) * 1000)
            means = {name: sum(values) / len(values) for name, values in timings.items()}
            checks.append({"analysis_test_mode": mode, "results_equal": True,
                           "warm_ms": timings, "warm_mean_ms": means,
                           "reduction_percent": 100 * (1 - means["upgrade"] / means["baseline"]),
                           "tracker_mean_ms": sum(tracker_times) / len(tracker_times),
                           "notes": [item["note_name"] for item in results["upgrade"]["notes"]],
                           "chord": results["upgrade"]["chord"]["label"]})
        report = {"source": "actual_PL_identical_spectrum", "checks": checks, "cold_ms": cold,
                  "frame_window_ms": 4096 / 48, "includes_capture_or_web_latency": False,
                  "baseline": "frozen AUD4 PS", "upgrade": "harmonic_tracking_v2"}
        encoded = json.dumps(report, indent=2, allow_nan=False)
        Path("board_music_upgrade_comparison.json").write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
    finally:
        driver.set_test_mode(0)
        driver.close()


if __name__ == "__main__":
    main()

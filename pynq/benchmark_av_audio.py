"""Bounded actual-PL audio benchmark. Stop the regular audio service first."""

import argparse
import json
from pathlib import Path
import statistics
import threading
import time

from audio_fft import AudioFFT
from audio_music import MusicConfig, MusicRecognizer
from audio_music_server import MusicState, make_server, settle_analysis_source
from av_system import AUDIO_BASE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=45)
    parser.add_argument("--output", required=True)
    parser.add_argument("--poll-interval", type=float, default=.05)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error("seconds must be 1..300")
    driver = AudioFFT(base_address=AUDIO_BASE, dev_mem=True)
    state = MusicState(driver, MusicRecognizer(MusicConfig(power_scale=1048576.0)),
                       poll_interval=args.poll_interval)
    state.recognizer._build_templates()
    server = make_server(state, host="0.0.0.0", port=8081)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    samples = []
    started = None
    try:
        driver.set_test_mode(2)
        settle_analysis_source(driver)
        worker.start()
        started = time.monotonic()
        print("AV_AUDIO_BENCH_READY", flush=True)
        previous = None
        while time.monotonic() - started < args.seconds:
            tick = time.monotonic()
            state.update()
            current = state.snapshot()
            if current["sequence"] != previous:
                previous = current["sequence"]
                samples.append({"elapsed_seconds": time.monotonic() - started,
                    "sequence": previous, "status": current["status"],
                    "chord": current.get("chord", {}).get("label"),
                    "stable_chord": current.get("stable_chord", {}).get("label"),
                    "timing": current.get("timing"),
                    "drops": (current.get("fft") or {}).get("drops"),
                    "error": current.get("error")})
            time.sleep(max(.001, args.poll_interval - (time.monotonic() - tick)))
    finally:
        if worker.is_alive():
            server.shutdown()
            worker.join(timeout=3)
        server.server_close()
        try:
            driver.set_test_mode(0)
        finally:
            driver.close()
    report = {"scope": "actual PL C major to ARM recognition; not microphone-to-display latency",
              "duration_seconds": time.monotonic() - started, "samples": samples,
              "restored_live_mode": True}
    good = [s for s in samples if s["status"] == "recognized" and s["chord"] == "C"]
    report["recognized_frames"] = len(good)
    report["published_updates_per_second"] = len(samples) / report["duration_seconds"]
    report["processing_median_ms"] = statistics.median(s["timing"]["processing_ms"] for s in good) if good else None
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not good or any(s["error"] or s["status"] != "recognized" or s["chord"] != "C" for s in samples):
        raise RuntimeError("Audio benchmark did not continuously recognize PL C major")
    print("AV_AUDIO_BENCH_PASS {} frames".format(len(good)), flush=True)


if __name__ == "__main__":
    main()

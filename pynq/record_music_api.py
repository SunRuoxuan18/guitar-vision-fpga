"""Record an actual running board API without opening another MMIO owner."""

import argparse
import json
from pathlib import Path
import time
from urllib.request import urlopen
from urllib.error import URLError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-mode", type=int, choices=range(4), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + 45
    while True:
        try:
            with urlopen("http://127.0.0.1:8081/api/audio", timeout=3) as response:
                payload = json.load(response)
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(.2)
            continue
        valid = (payload.get("source") == "hardware" and not payload.get("stale") and
                 payload.get("status") in ("silence", "recognized", "unknown") and
                 payload.get("fft", {}).get("analysis_test_mode") == args.expected_mode and
                 payload.get("algorithm_version") == "harmonic_tracking_v2")
        if args.expected_mode in (2, 3):
            label = {2: "C", 3: "Am"}[args.expected_mode]
            valid = valid and payload.get("stable_chord", {}).get("label") == label
        if valid:
            assert len(payload["spectrum"]["bins"]) == 2048, payload
            Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
            print(json.dumps({"passed": True, "mode": args.expected_mode, "source": payload["source"],
                              "status": payload["status"], "sequence": payload["sequence"],
                              "chord": payload["chord"]["label"],
                              "stable_chord": payload["stable_chord"]["label"],
                              "algorithm_version": payload["algorithm_version"],
                              "bin_count": len(payload["spectrum"]["bins"])}))
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("API did not publish expected confirmed result: {}".format(payload.get("status")))
        time.sleep(.2)


if __name__ == "__main__":
    main()

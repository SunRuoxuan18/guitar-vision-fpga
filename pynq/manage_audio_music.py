"""Start or stop the board HTTP service independently of the UART terminal."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PID_FILE = ROOT / "audio_music_server.pid"


def running_pid():
    if not PID_FILE.exists():
        return None
    pid = int(PID_FILE.read_text().strip())
    try:
        command = Path("/proc/{}/cmdline".format(pid)).read_bytes()
        if b"audio_music_server.py" in command:
            return pid
    except FileNotFoundError:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--test-mode", type=int, choices=range(4), default=0)
    args = parser.parse_args()
    pid = running_pid()
    if args.action == "stop":
        if pid is not None:
            os.kill(pid, signal.SIGINT)
            deadline = time.monotonic() + 6
            while running_pid() is not None and time.monotonic() < deadline:
                time.sleep(0.1)
            if running_pid() is not None:
                raise RuntimeError("Service did not stop; do not start another MMIO reader")
        PID_FILE.unlink(missing_ok=True)
        print("AUDIO_SERVICE_STOPPED")
    elif args.action == "start":
        if pid is not None:
            raise RuntimeError("Audio service already running: {}".format(pid))
        environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        with (ROOT / "audio_music_server.log").open("ab") as logfile:
            process = subprocess.Popen([
                sys.executable, "-u", "-B", str(ROOT / "pynq/audio_music_server.py"),
                "--dev-mem", "--host", "0.0.0.0", "--port", "8081",
                "--test-mode", str(args.test_mode)
            ], cwd=str(ROOT), env=environment, stdin=subprocess.DEVNULL,
                stdout=logfile, stderr=logfile, start_new_session=True)
        PID_FILE.write_text(str(process.pid) + "\n")
        print("AUDIO_SERVICE_STARTED", process.pid)
    else:
        print(json.dumps({"running": pid is not None, "pid": pid}))


if __name__ == "__main__":
    main()

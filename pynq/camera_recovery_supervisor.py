#!/usr/bin/env python3
"""Reset the PS when the enumerated C922 capture stack stops publishing frames."""
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen


VENDOR = "046d"
PRODUCT = "085c"
STATUS_URL = "http://127.0.0.1:8082/api/vision/status"
STATE_PATH = Path("/var/lib/fretlens-camera-supervisor/last-reboot")


def capture_process_blocked(proc_root=Path("/proc")):
    """Return true when any camera thread is in kernel wait state D."""
    try:
        processes = list(proc_root.iterdir())
    except OSError:
        return False
    for process in processes:
        try:
            command = (process / "cmdline").read_bytes()
            if b"vision_server.py" not in command:
                continue
            tasks = list((process / "task").iterdir())
        except (FileNotFoundError, IndexError, OSError):
            continue
        for task in tasks:
            try:
                stat = (task / "stat").read_text()
                state = stat.rsplit(")", 1)[1].split()[0]
                if state == "D":
                    return True
            except (FileNotFoundError, IndexError, OSError):
                continue
    return False


def camera_present(vendor=VENDOR, product=PRODUCT):
    for device in Path("/sys/bus/usb/devices").glob("*"):
        try:
            if ((device / "idVendor").read_text().strip() == vendor and
                    (device / "idProduct").read_text().strip() == product):
                return True
        except (FileNotFoundError, OSError):
            continue
    return False


def read_status(timeout=3.0):
    with urlopen(STATUS_URL, timeout=timeout) as response:
        return json.load(response)


def reboot_board():
    os.sync()
    subprocess.run(["/sbin/reboot", "-f"], check=False)


class RecoverySupervisor:
    def __init__(self, status_reader=read_status, camera_reader=camera_present,
                 capture_blocked_reader=capture_process_blocked,
                 reboot=reboot_board, state_path=STATE_PATH, poll_seconds=5.0,
                 startup_grace=90.0, stale_after=30.0, reboot_cooldown=600.0,
                 clock=time.monotonic, wall_clock=time.time):
        if min(poll_seconds, startup_grace, stale_after, reboot_cooldown) <= 0:
            raise ValueError("All recovery intervals must be positive")
        self.status_reader = status_reader
        self.camera_reader = camera_reader
        self.capture_blocked_reader = capture_blocked_reader
        self.reboot = reboot
        self.state_path = Path(state_path)
        self.poll_seconds = poll_seconds
        self.startup_grace = startup_grace
        self.stale_after = stale_after
        self.reboot_cooldown = reboot_cooldown
        self.clock = clock
        self.wall_clock = wall_clock
        self.started_at = clock()
        self.unhealthy_since = None
        self.blocked_since = None
        self.ever_healthy = False
        self.last_action = None

    def _cooldown_remaining(self):
        try:
            last_reboot = float(self.state_path.read_text().strip())
        except (FileNotFoundError, OSError, ValueError):
            return 0.0
        return max(0.0, self.reboot_cooldown - (self.wall_clock() - last_reboot))

    def _record_reboot(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text("{:.3f}\n".format(self.wall_clock()))

    @staticmethod
    def _healthy(status):
        try:
            return (status.get("status") == "streaming" and
                    status.get("frame_available") is True and
                    float(status.get("age_ms")) <= 5000.0)
        except (AttributeError, TypeError, ValueError):
            return False

    def step(self):
        now = self.clock()
        if not self.camera_reader():
            self.unhealthy_since = None
            self.blocked_since = None
            self.last_action = "camera_missing"
            return self.last_action
        status = None
        try:
            status = self.status_reader()
        except Exception:
            pass
        if status is not None and self._healthy(status):
            self.unhealthy_since = None
            self.blocked_since = None
            self.ever_healthy = True
            self.last_action = "healthy"
            return self.last_action
        if not self.ever_healthy and now - self.started_at < self.startup_grace:
            self.last_action = "starting"
            return self.last_action
        if self.capture_blocked_reader():
            if self.blocked_since is None:
                self.blocked_since = now
            if now - self.blocked_since >= self.stale_after:
                return self._reboot_or_wait()
            self.last_action = "capture_blocked"
            return self.last_action
        self.blocked_since = None
        if status is None:
            self.unhealthy_since = None
            self.last_action = "status_unavailable"
            return self.last_action
        if self.unhealthy_since is None:
            self.unhealthy_since = now
        if now - self.unhealthy_since < self.stale_after:
            self.last_action = "unhealthy"
            return self.last_action
        return self._reboot_or_wait()

    def _reboot_or_wait(self):
        if self._cooldown_remaining() > 0:
            self.last_action = "reboot_cooldown"
            return self.last_action
        self._record_reboot()
        self.last_action = "rebooting"
        self.reboot()
        return self.last_action

    def run(self):
        while True:
            try:
                previous = self.last_action
                action = self.step()
                if action != previous and action in (
                        "starting", "unhealthy", "capture_blocked",
                        "status_unavailable", "reboot_cooldown", "rebooting"):
                    print("Camera recovery: {}".format(action), flush=True)
            except Exception as exc:
                print("Camera recovery error: {}".format(exc), flush=True)
            time.sleep(self.poll_seconds)


if __name__ == "__main__":
    RecoverySupervisor().run()

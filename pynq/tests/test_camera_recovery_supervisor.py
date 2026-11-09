import tempfile
import unittest
from pathlib import Path
import sys
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import camera_recovery_supervisor as supervisor


class RecoverySupervisorTests(unittest.TestCase):
    def make(self, *statuses, camera=True, blocked=False, times=None, **kwargs):
        clock = Mock(side_effect=[0.0] + list(times or [10.0, 15.0, 40.0]))
        status_reader = Mock(side_effect=list(statuses))
        blocked_reader = Mock(return_value=blocked)
        reboot = Mock()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        recovery = supervisor.RecoverySupervisor(
            status_reader=status_reader,
            camera_reader=Mock(return_value=camera),
            capture_blocked_reader=blocked_reader,
            reboot=reboot,
            state_path=supervisor.Path(directory.name) / "last-reboot",
            clock=clock,
            wall_clock=Mock(return_value=1000.0),
            **kwargs)
        return recovery, status_reader, reboot

    def test_missing_camera_never_reboots(self):
        recovery, status_reader, reboot = self.make(camera=False)
        self.assertEqual(recovery.step(), "camera_missing")
        status_reader.assert_not_called()
        reboot.assert_not_called()

    def test_fresh_camera_status_is_healthy(self):
        recovery, _, reboot = self.make(
            {"status": "streaming", "frame_available": True, "age_ms": 20})
        self.assertEqual(recovery.step(), "healthy")
        reboot.assert_not_called()

    def test_persistent_unhealthy_capture_reboots_after_grace(self):
        stale = {"status": "stale", "frame_available": False, "age_ms": 2000}
        recovery, _, reboot = self.make(
            stale, stale, stale, stale,
            times=[0.0, 50.0, 100.0, 130.0])
        self.assertEqual(recovery.step(), "starting")
        self.assertEqual(recovery.step(), "starting")
        self.assertEqual(recovery.step(), "unhealthy")
        self.assertEqual(recovery.step(), "rebooting")
        reboot.assert_called_once_with()

    def test_known_bad_stack_does_not_reboot_within_cooldown(self):
        stale = {"status": "error", "frame_available": False, "age_ms": None}
        recovery, _, reboot = self.make(
            stale, stale, stale, times=[0.0, 130.0, 160.0])
        recovery.state_path.parent.mkdir(parents=True, exist_ok=True)
        recovery.state_path.write_text("950\n")
        self.assertEqual(recovery.step(), "starting")
        self.assertEqual(recovery.step(), "unhealthy")
        self.assertEqual(recovery.step(), "reboot_cooldown")
        reboot.assert_not_called()

    def test_status_timeout_does_not_reboot_a_running_camera(self):
        recovery, _, reboot = self.make(
            OSError("status timeout"),
            OSError("status timeout"),
            times=[0.0, 130.0])
        self.assertEqual(recovery.step(), "starting")
        self.assertEqual(recovery.step(), "status_unavailable")
        reboot.assert_not_called()

    def test_kernel_blocked_capture_reboots_after_confirmation(self):
        recovery, _, reboot = self.make(
            OSError("status timeout"),
            OSError("status timeout"),
            OSError("status timeout"),
            blocked=True,
            times=[0.0, 130.0, 160.0])
        self.assertEqual(recovery.step(), "starting")
        self.assertEqual(recovery.step(), "capture_blocked")
        self.assertEqual(recovery.step(), "rebooting")
        reboot.assert_called_once_with()

    def test_blocked_capture_thread_is_detected(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        proc_root = Path(directory.name)
        process = proc_root / "123"
        task = process / "task" / "456"
        task.mkdir(parents=True)
        (process / "cmdline").write_bytes(
            b"python3\0/home/xilinx/vision_server.py\0")
        (task / "stat").write_text(
            "456 (vision_server) D 1 456 456 0 -1 4194560")
        self.assertTrue(supervisor.capture_process_blocked(proc_root))


if __name__ == "__main__":
    unittest.main()

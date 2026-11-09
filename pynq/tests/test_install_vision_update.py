"""Verify staged video updates leave audio alone and restore failed copies."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from install_vision_update import install


class VisionUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stage = self.root / "stage"
        self.home = self.root / "home"
        self.system = self.root / "system"
        (self.stage / "pynq").mkdir(parents=True)
        self.home.mkdir()
        (self.system / "etc/systemd/system").mkdir(parents=True)
        (self.system / "etc/udev/rules.d").mkdir(parents=True)
        (self.stage / "pynq/vision_server.py").write_text("VERSION = 2\n")
        (self.stage / "pynq/yolo_detector.py").write_text("VERSION = 2\n")
        (self.home / "vision_server.py").write_text("VERSION = 1\n")
        (self.home / "audio_music_server.py").write_text("AUDIO_UNCHANGED\n")
        self.unit = self.system / "etc/systemd/system/fretlens-vision.service"
        self.rule = self.system / "etc/udev/rules.d/99-fretlens-camera.rules"
        for target in (self.unit, self.rule):
            target.write_text("original\n")
            (self.stage / "pynq" / target.name).write_text("updated\n")

    @staticmethod
    def succeeded(command, **kwargs):
        return subprocess.CompletedProcess(command, 0)

    def test_install_validates_all_yolo_files_and_backs_up_configuration(self):
        (self.stage / "pynq/patch_similarity.py").write_text("VERSION = 3\n")
        with patch("install_vision_update.subprocess.run", side_effect=self.succeeded) as run:
            install(self.stage, self.home, self.system)
        self.assertEqual((self.home / "yolo_detector.py").read_text(), "VERSION = 2\n")
        self.assertEqual((self.home / "patch_similarity.py").read_text(), "VERSION = 3\n")
        backup = next((self.home / "vision_backups").iterdir())
        self.assertEqual((backup / self.rule.name).read_text(), "original\n")
        self.assertEqual((backup / self.unit.name).read_text(), "original\n")
        self.assertEqual((self.home / "audio_music_server.py").read_text(), "AUDIO_UNCHANGED\n")
        commands = [call.args[0] for call in run.call_args_list]
        self.assertTrue(any(command[:2] == ["systemctl", "start"] for command in commands))
        self.assertFalse(any("audio" in " ".join(command) for command in commands))

    def test_invalid_yolo_code_is_rejected_before_stopping_service(self):
        (self.stage / "pynq/yolo_detector.py").write_text("def broken(:\n")
        with patch("install_vision_update.subprocess.run") as run:
            with self.assertRaises(SyntaxError):
                install(self.stage, self.home, self.system)
        run.assert_not_called()
        self.assertEqual((self.home / "vision_server.py").read_text(), "VERSION = 1\n")

    def test_start_failure_restores_files_and_removes_new_modules(self):
        starts = []

        def run(command, **kwargs):
            if command[:2] == ["systemctl", "start"]:
                starts.append(command)
                if len(starts) == 1:
                    raise subprocess.CalledProcessError(1, command)
            return self.succeeded(command, **kwargs)

        with patch("install_vision_update.subprocess.run", side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError):
                install(self.stage, self.home, self.system)
        self.assertEqual((self.home / "vision_server.py").read_text(), "VERSION = 1\n")
        self.assertFalse((self.home / "yolo_detector.py").exists())
        self.assertEqual(self.unit.read_text(), "original\n")
        self.assertEqual(self.rule.read_text(), "original\n")
        self.assertEqual(len(starts), 2)


if __name__ == "__main__":
    unittest.main()

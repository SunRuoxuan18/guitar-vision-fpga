"""PL preview provenance, timestamp and source isolation contracts."""

from email.message import Message
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_pl_server import PLPreview, main


class PLPreviewTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.processor = Mock()
        self.processor.process.return_value = {
            "gray": np.full((480, 640), 90, np.uint8),
            "edge": np.full((480, 640), 255, np.uint8),
            "timings_ms": {"dma_ms": 4}, "hardware_metadata": {"backend": "pl"}}
        self.state = PLPreview(self.processor, clock=lambda: self.now)
        self.headers = Message()
        for key, value in (("Sequence", "7"), ("Stream", "stream-a"),
                           ("Captured-Ms", "100000"), ("Source", "camera")):
            self.headers["X-Vision-" + key] = value
        self.jpeg = cv2.imencode(".jpg", np.full((480, 640, 3), 80, np.uint8))[1].tobytes()

    def test_real_outputs_are_packed_with_original_camera_identity(self):
        self.assertTrue(self.state.process_frame(self.jpeg, self.headers))
        payload, encoded = self.state.snapshot()
        self.assertEqual((payload["backend"], payload["stream_id"], payload["sequence"]), ("pl", "stream-a", 7))
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(image.shape, (480, 1280))
        self.assertLess(abs(float(image[:, :640].mean()) - 90), 2)
        self.assertGreater(float(image[:, 640:].mean()), 250)
        self.assertFalse(self.state.process_frame(self.jpeg, self.headers))
        self.assertEqual(self.processor.process.call_count, 1)

    def test_stale_output_is_not_served(self):
        self.state.process_frame(self.jpeg, self.headers)
        self.now += 1.01
        payload, jpeg = self.state.snapshot()
        self.assertTrue(payload["stale"])
        self.assertIsNone(jpeg)

    def test_expired_input_never_reaches_hardware(self):
        self.now += 2
        with self.assertRaises(ValueError):
            self.state.process_frame(self.jpeg, self.headers)
        self.processor.process.assert_not_called()

    def test_hardware_failure_is_propagated_without_cpu_fallback(self):
        self.processor.process.side_effect = TimeoutError("DMA timeout")
        with self.assertRaises(TimeoutError):
            self.state.process_frame(self.jpeg, self.headers)
        self.assertIsNone(self.state.snapshot()[1])

    def test_stream_restart_accepts_smaller_sequence(self):
        self.state.process_frame(self.jpeg, self.headers)
        self.headers.replace_header("X-Vision-Stream", "stream-b")
        self.headers.replace_header("X-Vision-Sequence", "1")
        self.assertTrue(self.state.process_frame(self.jpeg, self.headers))
        self.assertEqual(self.state.snapshot()[0]["stream_id"], "stream-b")

    def test_invalid_cli_does_not_program_pl(self):
        with patch("vision_pl.VisionPL") as driver:
            for fps in ("0", "nan", "31"):
                with self.subTest(fps=fps), self.assertRaises(SystemExit):
                    main(["--bitfile", "unused.bit", "--fps", fps])
            driver.assert_not_called()

    def test_http_bind_failure_does_not_program_pl(self):
        with patch("vision_pl.VisionPL") as driver, patch(
                "vision_pl_server.make_server", side_effect=OSError("occupied")):
            with self.assertRaises(OSError):
                main(["--bitfile", "unused.bit"])
            driver.assert_not_called()


if __name__ == "__main__":
    unittest.main()

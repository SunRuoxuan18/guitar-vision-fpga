"""Independent image-space checks for an uncalibrated six-string detector."""

import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_fretboard import AutoFretboardTracker

try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except (ImportError, OSError):
    CV_AVAILABLE = False


class LazyImportTests(unittest.TestCase):
    def test_import_construct_snapshot_invalidate_without_cv(self):
        path = Path(__file__).resolve().parents[1] / "auto_fretboard.py"
        code = """
import importlib.abc
import importlib.util
import sys
class BlockCV(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cv2', 'numpy'):
            raise AssertionError('eager import: ' + fullname)
sys.meta_path.insert(0, BlockCV())
spec = importlib.util.spec_from_file_location('auto_fretboard', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tracker = module.AutoFretboardTracker()
assert tracker.snapshot()['status'] == 'searching'
assert tracker.update(None)['corners'] is None
assert tracker.invalidate('manual')['reason'] == 'manual'
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(path)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


class GuitarScene:
    """A 22-fret physical scale projected from an independently drawn texture.

    The renderer exposes image-space string/fret endpoints for geometric checks;
    no detector code or detector line estimates enter the ground truth.
    """

    def __init__(self, width=640, height=400):
        self.width, self.height = width, height
        self.source = np.float32([[0, 0], [1099, 0], [1099, 179], [0, 179]])
        self.quad = np.float32([[65, 130], [585, 105], [570, 267], [55, 245]])
        self.quad *= (width / 640, height / 400)
        self.string_y = np.linspace(20, 159, 6)
        scale = 1400
        self.fret_x = scale * (1 - 2.0 ** (-np.arange(1, 23) / 12))
        rng = np.random.default_rng(392)
        grain = rng.normal(0, 1.2, (180, 1100))
        self.wood = np.uint8(np.clip(65 + grain + 4 * np.sin(np.arange(1100)[None, :] / 27), 0, 255))

    def render(self, *, angle=0, dx=0, dy=0, blur=0, occlusion=None,
               regular=False, strings=6, distractors=False, string_width=2,
               visible_frets=None):
        texture = self.wood.copy()
        fret_x = np.linspace(50, 1050, 22) if regular else self.fret_x
        for x in fret_x if visible_frets is None else fret_x[:visible_frets]:
            cv2.line(texture, (int(round(x)), 5), (int(round(x)), 174), 180, 3, cv2.LINE_AA)
        for y in self.string_y[:strings]:
            cv2.line(texture, (0, int(round(y))), (1099, int(round(y))), 235, string_width, cv2.LINE_AA)
        for fret in (3, 5, 7, 9, 15, 17, 19):
            x = int(round((fret_x[fret - 1] + fret_x[fret - 2]) / 2))
            cv2.circle(texture, (x, 89), 5, 130, -1, cv2.LINE_AA)
        if occlusion is not None:
            x, y, w, h = occlusion
            cv2.rectangle(texture, (x, y), (x + w, y + h), 95, -1)
        affine = cv2.getRotationMatrix2D(tuple(self.quad.mean(axis=0).astype(float)), angle, 1.0)
        affine[:, 2] += (dx, dy)
        quad = cv2.transform(self.quad[:, None, :], affine)[:, 0]
        transform = cv2.getPerspectiveTransform(self.source, quad)
        size = self.width, self.height
        frame = np.full((self.height, self.width), 28, np.uint8)
        if distractors:
            cv2.rectangle(frame, (18, 22), (185, 72), 120, 2)
            cv2.line(frame, (5, self.height - 40), (self.width - 5, self.height - 12), 210, 2)
            for x in range(330, 610, 23):
                cv2.line(frame, (x, 12), (x + 8, 65), 170, 2)
        warped = cv2.warpPerspective(texture, transform, size)
        mask = cv2.warpPerspective(np.full(texture.shape, 255, np.uint8), transform, size)
        frame[mask != 0] = warped[mask != 0]
        if blur:
            frame = cv2.GaussianBlur(frame, (5, 5), blur)
        endpoints = np.float32([[[0, y], [1099, y]] for y in self.string_y])
        expected_strings = cv2.perspectiveTransform(endpoints.reshape(-1, 1, 2), transform).reshape(6, 2, 2)
        endpoints = np.float32([[[x, 20], [x, 159]] for x in fret_x])
        expected_frets = cv2.perspectiveTransform(endpoints.reshape(-1, 1, 2), transform).reshape(len(fret_x), 2, 2)
        return frame, expected_strings, expected_frets


@unittest.skipUnless(CV_AVAILABLE, "OpenCV/numpy unavailable; use vision/.venv")
class AutoFretboardTests(unittest.TestCase):
    def setUp(self):
        self.scene = GuitarScene()
        self.tracker = AutoFretboardTracker()

    def assert_searching(self, result):
        self.assertEqual(result["status"], "searching", result)
        self.assertTrue(result["reason"])
        self.assertIsNone(result["corners"])
        self.assertEqual(result["strings"], [])
        self.assertEqual(result["frets"], [])
        self.assertEqual(result["confidence"], 0)
        json.dumps(result, allow_nan=False)

    def assert_geometry(self, result, strings, frets, scene=None):
        scene = scene or self.scene
        self.assertEqual(result["status"], "tracking", result)
        self.assertIsNone(result["reason"])
        self.assertEqual(result["mode"], "auto")
        self.assertEqual(result["orientation"], "image_order")
        self.assertFalse(result["absolute_frets_known"])
        self.assertEqual(result["string_count"], 6)
        self.assertGreaterEqual(result["fret_count"], 5)
        self.assertLessEqual(result["fret_count"], min(25, len(frets)))
        self.assertGreater(result["confidence"], .4)
        scale = np.array([scene.width - 1, scene.height - 1])
        for kind, expected in (("strings", strings), ("frets", frets)):
            seen = set()
            for item in result[kind]:
                self.assertTrue(item["observed"])
                points = np.array([[item[key]["x"], item[key]["y"]] for key in ("start", "end")])
                self.assertTrue(((points >= 0) & (points <= 1)).all())
                points *= scale
                vectors = expected[:, 1] - expected[:, 0]
                delta = points - expected[:, None, 0, :]
                cross = vectors[:, None, 0] * delta[:, :, 1] - vectors[:, None, 1] * delta[:, :, 0]
                distances = np.abs(cross) / np.linalg.norm(vectors, axis=1)[:, None]
                errors = distances.max(axis=1)
                closest = int(np.argmin(errors))
                self.assertLess(float(errors[closest]), 3.5, (kind, item))
                self.assertNotIn(closest, seen, "duplicate image edge reported as a wire")
                seen.add(closest)
        self.assertEqual([item["index"] for item in result["frets"]], list(range(result["fret_count"])))
        self.assertEqual(len(result["corners"]), 4)
        expected_corners = [result["strings"][0]["start"], result["strings"][0]["end"],
                            result["strings"][-1]["end"], result["strings"][-1]["start"]]
        for corner, expected in zip(result["corners"], expected_corners):
            np.testing.assert_allclose([corner["x"], corner["y"]],
                                       [expected["x"], expected["y"]], atol=1e-6)
        np.testing.assert_allclose(list(result["frets"][0]["start"].values()),
                                   list(result["strings"][0]["start"].values()), atol=1e-6)
        np.testing.assert_allclose(list(result["frets"][-1]["end"].values()),
                                   list(result["strings"][-1]["end"].values()), atol=1e-6)
        json.dumps(result, allow_nan=False)

    def test_acquires_six_strings_and_real_nonuniform_frets(self):
        frame, strings, frets = self.scene.render()
        result = self.tracker.update(frame)
        self.assert_geometry(result, strings, frets)
        self.assertEqual(result["fret_count"], 22)

    def test_extraction_bounds_working_frame_angles_segments_and_candidates(self):
        scene = GuitarScene(1280, 800)
        frame = scene.render()[0]
        with patch.object(cv2, "HoughLinesP", wraps=cv2.HoughLinesP) as hough:
            with patch.object(self.tracker, "_candidate", wraps=self.tracker._candidate) as candidate:
                result = self.tracker.update(frame)
        self.assertEqual(result["string_count"], 6)
        self.assertEqual(result["fret_count"], 22)
        hough.assert_called_once()
        args, kwargs = hough.call_args
        self.assertLessEqual(max(args[0].shape), 640)
        self.assertGreaterEqual(args[2], np.pi / 360)
        self.assertGreaterEqual(args[3], 35)
        self.assertGreaterEqual(kwargs["minLineLength"], 30)
        self.assertGreater(candidate.call_count, 0)
        self.assertLessEqual(candidate.call_count, 2)
        for args, _ in candidate.call_args_list:
            self.assertLessEqual(len(args[1]), 240)

    def test_motion_perspective_blur_and_distractors(self):
        for options in ({"angle": -12, "dx": 3, "dy": 12},
                        {"angle": 15, "dx": -5, "blur": .8, "distractors": True},
                        {"angle": 2, "dx": 9, "dy": -12, "blur": .6}):
            with self.subTest(options=options):
                frame, strings, frets = self.scene.render(**options)
                self.assert_geometry(self.tracker.update(frame), strings, frets)

    def test_end_to_end_continuous_translation_and_rotation(self):
        for step in range(5):
            with self.subTest(step=step):
                frame, strings, frets = self.scene.render(
                    angle=step * 2, dx=step * 2, dy=step)
                result = self.tracker.update(frame)
                self.assert_geometry(result, strings, frets)
                self.assertEqual(result["fret_count"], 22)

    def test_arbitrary_rotations_keep_corners_and_axes_consistent(self):
        scene = GuitarScene(640, 640)
        for angle in (-170, -135, -90, -60, -45, 0, 35, 45, 60, 90, 135, 170):
            with self.subTest(angle=angle):
                frame, strings, frets = scene.render(angle=angle, blur=.6, distractors=True)
                self.assert_geometry(self.tracker.update(frame), strings, frets, scene)

    def test_small_pose_jitter_does_not_swap_axes_or_strings(self):
        scene = GuitarScene(640, 640)
        previous = None
        for step in range(8):
            angle = 39 + (-1) ** step * .3
            frame, strings, frets = scene.render(angle=angle, dx=step * .4,
                                                 dy=(-1) ** step * .4, blur=.5)
            result = self.tracker.update(frame)
            self.assert_geometry(result, strings, frets, scene)
            midpoints = np.array([[(line["start"][key] + line["end"][key]) / 2
                                  for key in ("x", "y")] for line in result["strings"]])
            if previous is not None:
                self.assertLess(float(np.linalg.norm(midpoints - previous, axis=1).max()), .025)
            previous = midpoints

    def test_six_thin_strings_and_fewer_visible_frets_are_observed(self):
        for count in (5, 8, 14):
            with self.subTest(visible_frets=count):
                frame, strings, frets = self.scene.render(string_width=1, visible_frets=count)
                result = self.tracker.update(frame)
                self.assert_geometry(result, strings, frets[:count])
                self.assertEqual(result["fret_count"], count)

    def test_vertical_pose_jitter_keeps_image_order(self):
        scene = GuitarScene(640, 640)
        previous = None
        for angle in (89.4, 89.7, 90, 90.3, 90.6):
            frame, strings, frets = scene.render(angle=angle, blur=.5)
            result = self.tracker.update(frame)
            self.assert_geometry(result, strings, frets, scene)
            first = result["strings"][0]
            self.assertLess(first["start"]["y"], first["end"]["y"])
            center = np.array([(first["start"][key] + first["end"][key]) / 2 for key in ("x", "y")])
            if previous is not None:
                self.assertLess(float(np.linalg.norm(center - previous)), .025)
            previous = center

    def test_high_resolution_frames_are_detected_at_bounded_working_size(self):
        scene = GuitarScene(1280, 800)
        frame, strings, frets = scene.render(string_width=1)
        self.assert_geometry(self.tracker.update(frame), strings, frets, scene)

    def test_current_physical_frame_without_guitar_does_not_acquire(self):
        path = Path(__file__).resolve().parents[2] / "diagnostics" / "current_neck_frame.jpg"
        if not path.exists():
            self.skipTest("Optional physical no-guitar fixture unavailable")
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(frame)
        self.assert_searching(self.tracker.update(frame))

    def test_occlusion_does_not_fabricate_missing_lines(self):
        for occlusion in ((400, 0, 45, 180), (300, 40, 60, 80),
                          (0, 17, 1100, 10), (200, 0, 700, 180)):
            with self.subTest(occlusion=occlusion):
                frame = self.scene.render(occlusion=occlusion)[0]
                self.assert_searching(self.tracker.update(frame))

    def test_blank_background_and_projected_regular_grid_do_not_acquire(self):
        rng = np.random.default_rng(351)
        for frame in (np.zeros((400, 640), np.uint8),
                      np.full((400, 640), 127, np.uint8),
                      rng.integers(0, 256, (400, 640), dtype=np.uint8),
                      self.scene.render(regular=True)[0]):
            self.assert_searching(self.tracker.update(frame))

    def test_projected_grids_at_multiple_angles_and_irregular_spacing_are_rejected(self):
        scene = GuitarScene(640, 640)
        for angle in (-170, -90, -45, 0, 35, 90, 135):
            with self.subTest(angle=angle):
                frame = scene.render(regular=True, angle=angle, blur=.6, distractors=True)[0]
                self.assert_searching(self.tracker.update(frame))
        scene.fret_x = np.array([30, 69, 98, 151, 173, 201, 266, 279, 339, 359, 381,
                                444, 459, 516, 570, 606, 633, 710, 749, 860, 940, 1020])
        self.assert_searching(self.tracker.update(scene.render()[0]))

    def test_more_than_25_supported_frets_is_rejected_not_truncated(self):
        scene = GuitarScene()
        scene.fret_x = 1400 * (1 - 2.0 ** (-np.arange(1, 27) / 12))
        self.assert_searching(self.tracker.update(scene.render()[0]))

    def test_missing_string_and_disappearance_clear_then_reacquire(self):
        frame, strings, frets = self.scene.render()
        self.assert_geometry(self.tracker.update(frame), strings, frets)
        self.assert_searching(self.tracker.update(self.scene.render(strings=5)[0]))
        self.assert_searching(self.tracker.update(np.zeros_like(frame)))
        self.assert_geometry(self.tracker.update(frame), strings, frets)

    def test_snapshots_invalid_frames_and_explicit_invalidation(self):
        frame, strings, frets = self.scene.render()
        result = self.tracker.update(frame)
        self.assert_geometry(result, strings, frets)
        result["strings"][0]["start"]["x"] = -10
        self.assert_geometry(self.tracker.snapshot(), strings, frets)
        self.assert_searching(self.tracker.invalidate("camera_stopped"))
        self.assertEqual(self.tracker.snapshot()["reason"], "camera_stopped")
        for invalid in (None, [], frame.astype(np.float32), frame[:, :, None], np.zeros((0, 0), np.uint8)):
            self.assert_searching(self.tracker.update(invalid))
        readonly = frame[:, ::-1].copy()[:, ::-1]
        readonly.flags.writeable = False
        before = readonly.copy()
        self.assert_geometry(self.tracker.update(readonly), strings, frets)
        np.testing.assert_array_equal(readonly, before)


if __name__ == "__main__":
    unittest.main()

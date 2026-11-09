"""Offline tracker checks against independently rendered, known board poses."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fretboard_tracker import FretboardTracker

try:
    import cv2
    import numpy as np
    CV_AVAILABLE, CV_SKIP_REASON = True, ""
except (ImportError, OSError) as exc:
    CV_AVAILABLE = False
    CV_SKIP_REASON = "OpenCV/numpy unavailable: {}; use vision/.venv".format(exc)


@unittest.skipUnless(CV_AVAILABLE, "OpenCV/numpy unavailable")
class CorrelationTests(unittest.TestCase):
    def test_native_cv_warp_matches_reference_on_texture_and_borders(self):
        tracker = FretboardTracker()
        tracker._dependencies()
        native = tracker._warp_native
        if native is None:
            self.skipTest('board batched OpenCV library unavailable')
        rng = np.random.RandomState(125)
        image = rng.randint(0,256,(200,320)).astype(np.uint8)
        points = np.float32([[0,0],[319,199],[50.3,70.7],[120.5,80.2]]+
            (rng.rand(32,2)*(319,199)).tolist())
        for h in (np.eye(3),np.array([[1,.01,2.3],[-.02,1,-1.7],[.0001,-.0002,1]])):
            for other in (image,255-image,np.roll(image,1,axis=1),np.full_like(image,80)):
                actual=tracker._patch_correlations(other,image,points,h)
                tracker._warp_native=None
                expected=tracker._patch_correlations(other,image,points,h)
                tracker._warp_native=native
                np.testing.assert_allclose(actual,expected,atol=2e-5)

    def test_native_spatial_support_matches_reference(self):
        tracker = FretboardTracker()
        tracker._dependencies()
        native = tracker._spread_native
        if native is None:
            self.skipTest('board native support library unavailable')
        rng = np.random.RandomState(38)
        quad = np.float32([[0,0],[319,0],[319,199],[0,199]])
        for count in (16,24,32,48,240):
            for scale in (1., .8, .5, .01):
                for offset in (0., -.05, .1):
                    points = ((rng.rand(count,2)*scale+offset)*(319,199)).astype(np.float32)
                    tracker._spread_native = None
                    reference = tracker._well_spread(points,quad)
                    tracker._spread_native = native
                    self.assertEqual(tracker._well_spread(points,quad),reference)

    def test_cached_float32_patch_grid_preserves_previous_coordinates(self):
        tracker = FretboardTracker()
        tracker._dependencies()
        image = np.random.RandomState(12).randint(0,256,(200,320)).astype(np.uint8)
        points = np.float32([[0,0],[319,199],[50.3,70.7],[120.5,80.2]])
        transform = np.array([[1,.01,2.3],[-.02,1,-1.7],[.0001,-.0002,1]])
        yy,xx = np.mgrid[:16,:16]
        origins = np.floor(points-7).astype(np.float32)
        old = origins[:,None,None,:]+np.stack((xx,yy),axis=-1)
        old[...,0] = np.clip(old[...,0],0,319)
        old[...,1] = np.clip(old[...,1],0,199)
        mapped = cv2.perspectiveTransform(old.astype(np.float32).reshape(-1,1,2),transform)
        mapped = mapped.reshape(len(points)*16,16,2)
        atlas = cv2.remap(image,mapped[...,0],mapped[...,1],cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        expected = np.stack([cv2.getRectSubPix(atlas[i*16:(i+1)*16],(15,15),tuple(p-o))
                             for i,(p,o) in enumerate(zip(points,origins))])
        np.testing.assert_array_equal(tracker._aligned_patches(image,transform,points),expected)
        grid = tracker._patch_grid
        self.assertEqual(grid.dtype,np.float32)
        tracker._aligned_patches(image,transform,points)
        self.assertIs(tracker._patch_grid,grid)

    def test_patch_only_warp_matches_full_warp_including_border(self):
        tracker=FretboardTracker()
        tracker._dependencies()
        image=np.random.RandomState(81).randint(0,256,(200,320)).astype(np.uint8)
        points=np.float32([[0,0],[319,199],[50.3,70.7],[120.5,80.2]])
        for h in (np.eye(3),np.array([[1,.01,2.3],[-.02,1,-1.7],[.0001,-.0002,1]])):
            full=cv2.warpPerspective(image,h,(320,200),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
            expected=np.stack([cv2.getRectSubPix(full,(15,15),tuple(p)) for p in points])
            actual=tracker._aligned_patches(image,h,points)
            # Both use OpenCV's 1/32-pixel interpolation table; coordinate
            # rounding at bin boundaries can differ by one intensity level.
            self.assertLessEqual(float(np.mean(np.abs(actual.astype(float)-expected))), .15)
            np.testing.assert_allclose(tracker._patch_correlations(image,image,points,h),
                tracker._patch_correlations(full,image,points),atol=.003)

    def test_batch_matches_opencv_for_texture_flat_patches_and_borders(self):
        tracker = FretboardTracker()
        tracker._dependencies()
        rng = np.random.RandomState(42)
        textured = rng.randint(0,256,(200,320)).astype(np.uint8)
        points = np.float32([[0,0],[319,199],[50.3,70.7],[120.5,80.2]])
        flat = np.full_like(textured, 80)
        for a,b in ((textured,textured),(textured,255-textured),
                    (textured,np.roll(textured,1,axis=1)),(textured,flat),(flat,textured),(flat,flat)):
            expected = [cv2.matchTemplate(cv2.getRectSubPix(a,(15,15),tuple(p)),
                cv2.getRectSubPix(b,(15,15),tuple(p)),cv2.TM_CCOEFF_NORMED)[0,0] for p in points]
            np.testing.assert_allclose(tracker._patch_correlations(a,b,points),expected,atol=2e-5)


class LazyImportTests(unittest.TestCase):
    def test_import_construction_and_lost_state_need_no_cv_dependencies(self):
        path = Path(__file__).resolve().parents[1] / "fretboard_tracker.py"
        code = """
import importlib.abc
import importlib.util
import sys
class BlockCV(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('cv2', 'numpy'):
            raise AssertionError('eager dependency import: ' + fullname)
sys.meta_path.insert(0, BlockCV())
spec = importlib.util.spec_from_file_location('fretboard_tracker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
tracker = module.FretboardTracker()
assert tracker.snapshot()['status'] == 'lost'
assert tracker.update(None)['corners'] is None
assert tracker.invalidate('manual')['reason'] == 'manual'
assert not any(name in sys.modules for name in ('cv2', 'numpy'))
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(path)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


class BoardScene:
    """Render each pose from original texture, avoiding compounded warp errors."""

    def __init__(self, width=640, height=400, seed=713):
        self.width, self.height = width, height
        self.quad = np.float32([[.15, .35], [.80, .30], [.82, .61], [.14, .66]])
        self.quad *= (width - 1, height - 1)
        rng = np.random.default_rng(seed)
        self.texture = cv2.GaussianBlur(
            rng.integers(35, 215, (120, 440), dtype=np.uint8), (3, 3), 0)
        for x in (9, 39, 76, 119, 161, 208, 258, 311, 367, 428):
            cv2.line(self.texture, (x, 0), (x, 119), 215, 2)
        for y in (12, 31, 50, 69, 88, 107):
            cv2.line(self.texture, (0, y), (439, y), 55, 1)
        for x, y in ((60, 60), (140, 48), (234, 72), (335, 58), (390, 35)):
            cv2.circle(self.texture, (x, y), 5, 235, -1)
        self.background = rng.integers(15, 55, (height, width), dtype=np.uint8)

    def pose(self, step, *, angle=.45, scale=.0015, dx=1.7, dy=-.6):
        center = tuple(float(v) for v in self.quad.mean(axis=0))
        affine = cv2.getRotationMatrix2D(center, angle * step, 1 + scale * step)
        affine[:, 2] += (dx * step, dy * step)
        homography = np.vstack((affine, (step * 1.5e-6, -step * 2e-6, 1.0)))
        return cv2.perspectiveTransform(self.quad[:, None, :], homography)[:, 0]

    def corners(self, quad=None):
        normalized = (self.quad if quad is None else quad) / (self.width - 1, self.height - 1)
        return [{"x": float(x), "y": float(y)} for x, y in normalized]

    def render(self, quad=None, occluder=None):
        quad = self.quad if quad is None else quad
        source = np.float32([[0, 0], [439, 0], [439, 119], [0, 119]])
        homography = cv2.getPerspectiveTransform(source, quad.astype(np.float32))
        size = (self.width, self.height)
        warped = cv2.warpPerspective(self.texture, homography, size)
        mask = cv2.warpPerspective(np.full((120, 440), 255, dtype=np.uint8),
                                   homography, size, flags=cv2.INTER_NEAREST)
        frame = self.background.copy()
        frame[mask != 0] = warped[mask != 0]
        if occluder is not None:
            x, y, width, height = occluder
            cv2.rectangle(frame, (x, y), (x + width, y + height), 175, -1)
            for column in range(x + 3, x + width, 7):
                cv2.line(frame, (column, y), (column, y + height), 45, 2)
            cv2.line(frame, (x, y + height // 2),
                     (x + width, y + height // 2), 240, 3)
        return frame


@unittest.skipUnless(CV_AVAILABLE, CV_SKIP_REASON)
class FretboardTrackerTests(unittest.TestCase):
    def setUp(self):
        self.scene = BoardScene()
        self.tracker = FretboardTracker()

    def assert_tracking(self, result, expected, scene=None, tolerance=2.5):
        scene = scene or self.scene
        self.assertEqual(result["status"], "tracking", result)
        self.assertIsNone(result["reason"])
        self.assertEqual(len(result["corners"]), 4)
        actual = np.array([[p["x"], p["y"]] for p in result["corners"]])
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())
        actual *= (scene.width - 1, scene.height - 1)
        error = np.linalg.norm(actual - expected, axis=1).max()
        self.assertLessEqual(error, tolerance, "corner error {:.3f}px: {}".format(error, result))
        self.assertGreaterEqual(result["inliers"], 16)
        self.assertGreaterEqual(result["points"], result["inliers"])
        json.dumps(result, allow_nan=False)

    def assert_lost(self, result):
        self.assertEqual(result["status"], "lost", result)
        self.assertIsNone(result["corners"], result)
        self.assertTrue(result["reason"])
        json.dumps(result, allow_nan=False)

    def start(self, frame=None, corners=None):
        result = self.tracker.initialize(self.scene.render() if frame is None else frame,
                                         self.scene.corners() if corners is None else corners)
        self.assert_tracking(result, self.scene.quad)
        return result

    def test_translation_rotation_scale_and_perspective_over_many_frames(self):
        self.start()
        for step in range(1, 41):
            with self.subTest(step=step):
                quad = self.scene.pose(step)
                self.assert_tracking(self.tracker.update(self.scene.render(quad)), quad)

    def test_stronger_perspective_changes_without_an_affine_shortcut(self):
        self.start()
        offsets = np.float32([[18, 15], [-8, -20], [22, 8], [-10, -5]])
        for step in range(1, 21):
            quad = self.scene.quad + offsets * step / 20
            self.assert_tracking(self.tracker.update(self.scene.render(quad)), quad)

    def test_dimensions_and_corner_order_are_preserved(self):
        for width, height in ((320, 240), (960, 540)):
            for reverse in (False, True):
                with self.subTest(width=width, height=height, reverse=reverse):
                    scene = BoardScene(width, height)
                    tracker = FretboardTracker()
                    order = [3, 2, 1, 0] if reverse else [0, 1, 2, 3]
                    quad = scene.quad[order]
                    self.assert_tracking(tracker.initialize(scene.render(), scene.corners(quad)),
                                         quad, scene)
                    for step in range(1, 13):
                        expected = scene.pose(step, angle=-.7, scale=-.002, dx=-1.1, dy=.8)
                        self.assert_tracking(tracker.update(scene.render(expected)),
                                             expected[order], scene)

    def test_small_foreground_occluder_moves_independently(self):
        initial = self.start(self.scene.render(occluder=(210, 160, 36, 42)))
        for step in range(1, 26):
            quad = self.scene.pose(step, dx=-.7)
            frame = self.scene.render(quad, occluder=(210 + 4 * step, 160, 36, 42))
            self.assert_tracking(self.tracker.update(frame), quad)
        self.assertLess(self.tracker.snapshot()["inliers"], initial["inliers"])

    def test_exclusion_mask_keeps_hand_features_out_of_neck_track(self):
        frame = self.scene.render()
        mask = np.zeros_like(frame)
        cv2.rectangle(mask, (210, 145), (260, 205), 255, -1)
        result = self.tracker.initialize(frame, self.scene.corners(), mask)
        self.assertEqual(result["status"], "tracking", result)
        xy = np.rint(self.tracker._points[:, 0]).astype(int)
        self.assertTrue(np.all(mask[xy[:, 1], xy[:, 0]] == 0))

        current = self.scene.render(occluder=(214, 148, 60, 58))
        current_mask = np.zeros_like(current)
        cv2.rectangle(current_mask, (210, 144), (278, 210), 255, -1)
        result = self.tracker.update(current, current_mask)
        self.assertEqual(result["status"], "tracking", result)
        xy = np.rint(self.tracker._points[:, 0]).astype(int)
        self.assertTrue(np.all(current_mask[xy[:, 1], xy[:, 0]] == 0))

    def test_endpoint_features_keep_the_outer_edges_of_a_neck(self):
        tracker = FretboardTracker(endpoint_weighted=True)
        result = tracker.initialize(
            self.scene.render(), self.scene.corners())
        self.assert_tracking(result, self.scene.quad)
        self.assertEqual(result["sampling"], "endpoint_bands")
        self.assertGreaterEqual(min(result["endpoint_points"]), 8)
        tracker._dependencies()
        board = cv2.perspectiveTransform(
            tracker._points,
            cv2.getPerspectiveTransform(
                tracker._quad, np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])))[:, 0]
        self.assertTrue(np.logical_or(
            board[:, 0] <= .34, board[:, 0] >= .66).all())
        self.assertLessEqual(max(result["endpoint_points"]), 16)
        for end in (board[:, 0] <= .34, board[:, 0] >= .66):
            span = np.quantile(board[end], [.1, .9], axis=0)
            self.assertGreaterEqual(float(span[1, 1] - span[0, 1]), .24)

    def test_endpoint_sampling_never_uses_relaxed_feature_spacing(self):
        tracker = FretboardTracker(endpoint_weighted=True)
        original = tracker._feature_points
        calls = []

        def record(gray, mask, min_distance=5, erosion=7):
            calls.append((min_distance, erosion))
            return original(gray, mask, min_distance=min_distance, erosion=erosion)

        tracker._feature_points = record
        result = tracker.initialize(
            self.scene.render(), self.scene.corners())
        self.assert_tracking(result, self.scene.quad)
        self.assertNotIn((2, 3), calls)

    def test_endpoint_fast_path_always_runs_backward_flow_checks(self):
        class CountingFlow:
            def __init__(self):
                self.calls = 0

            def __call__(self, previous, current, points):
                self.calls += 1
                return cv2.calcOpticalFlowPyrLK(previous, current, points, None)

        flow = CountingFlow()
        tracker = FretboardTracker(
            flow_backend=flow, fast_motion=True, endpoint_weighted=True)
        self.assert_tracking(
            tracker.initialize(self.scene.render(), self.scene.corners()),
            self.scene.quad)
        for step in range(1, 9):
            quad = self.scene.pose(step, dx=.7, dy=-.3)
            result = tracker.update(self.scene.render(quad))
            self.assert_tracking(result, quad)
            self.assertNotIn("backward_skipped", tracker.timings)
        self.assertEqual(flow.calls, 16)

    def test_endpoint_support_keeps_two_transverse_caps_without_four_cells(self):
        tracker = FretboardTracker(endpoint_weighted=True)
        tracker._dependencies()
        unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        board = np.vstack((
            np.column_stack((np.linspace(.05, .30, 12), np.linspace(.05, .45, 12))),
            np.column_stack((np.linspace(.70, .95, 12), np.linspace(.55, .90, 12))),
        )).astype(np.float32)
        points = cv2.perspectiveTransform(
            board.reshape(-1, 1, 2),
            cv2.getPerspectiveTransform(unit, self.scene.quad))[:, 0]
        self.assertTrue(tracker._endpoint_spread(points, self.scene.quad))

    def test_endpoint_support_rejects_clustered_or_single_end_texture(self):
        tracker = FretboardTracker(endpoint_weighted=True)
        tracker._dependencies()
        unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        transform = cv2.getPerspectiveTransform(unit, self.scene.quad)

        def pixels(board):
            return cv2.perspectiveTransform(
                np.asarray(board, np.float32).reshape(-1, 1, 2), transform)[:, 0]

        clustered = np.vstack((
            np.column_stack((np.linspace(.05, .30, 12), np.linspace(.46, .54, 12))),
            np.column_stack((np.linspace(.70, .95, 12), np.linspace(.46, .54, 12))),
        ))
        self.assertFalse(tracker._endpoint_spread(
            pixels(clustered), self.scene.quad))
        one_end = np.column_stack((
            np.linspace(.05, .30, 24), np.linspace(.05, .95, 24)))
        self.assertFalse(tracker._endpoint_spread(
            pixels(one_end), self.scene.quad))

    def test_appearance_guard_is_bounded_and_requires_high_support(self):
        tracker = FretboardTracker(endpoint_weighted=True)
        tracker._dependencies()
        tracker._inliers = 28
        tracker._inlier_ratio = .96
        transient = np.array([.96] * 20 + [.40] * 8)
        self.assertTrue(tracker._appearance_supported(transient))
        self.assertTrue(tracker._appearance_supported(transient))
        self.assertFalse(tracker._appearance_supported(transient))

        strong = np.full(28, .98)
        self.assertTrue(tracker._appearance_supported(strong))
        weak = np.full(28, .80)
        self.assertFalse(tracker._appearance_supported(weak))

    def test_large_middle_hand_occlusion_leaves_both_ends_tracking(self):
        tracker = FretboardTracker(endpoint_weighted=True)
        result = tracker.initialize(self.scene.render(), self.scene.corners())
        self.assert_tracking(result, self.scene.quad)
        for step in range(1, 26):
            quad = self.scene.pose(step, dx=-.7)
            frame = self.scene.render(
                quad, occluder=(250, 70, 150, 290))
            result = tracker.update(frame)
            self.assert_tracking(result, quad)
            self.assertEqual(result["sampling"], "endpoint_bands")

    def test_one_occluded_end_is_bridged_for_a_bounded_time(self):
        tracker = FretboardTracker(endpoint_weighted=True)
        self.assert_tracking(
            tracker.initialize(self.scene.render(), self.scene.corners()),
            self.scene.quad)
        quad = self.scene.pose(1)
        mask = np.zeros((self.scene.height, self.scene.width), np.uint8)
        unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        right = np.float32([[[.58, 0], [1.1, 0], [1.1, 1], [.58, 1]]])
        mapped = cv2.perspectiveTransform(
            right, cv2.getPerspectiveTransform(unit, quad.astype(np.float32)))[0]
        cv2.fillConvexPoly(mask, np.rint(mapped).astype(np.int32), 255)
        result = tracker.update(self.scene.render(quad), mask)
        self.assert_tracking(result, quad)
        self.assertEqual(result["motion_mode"], "one_end_guarded")

    def test_blank_and_near_uniform_noise_cannot_initialize(self):
        rng = np.random.default_rng(41)
        for frame in (np.zeros((400, 640), dtype=np.uint8),
                      np.full((400, 640), 127, dtype=np.uint8),
                      rng.integers(126, 129, (400, 640), dtype=np.uint8)):
            with self.subTest(contrast=int(np.ptp(frame))):
                self.assert_lost(self.tracker.initialize(frame, self.scene.corners()))
                self.assertEqual(self.tracker.snapshot()["reason"], "insufficient_texture")

    def test_background_and_clustered_features_do_not_establish_board_support(self):
        rng = np.random.default_rng(43)
        background = rng.integers(0, 256, (400, 640), dtype=np.uint8)
        cv2.fillConvexPoly(background, self.scene.quad.astype(np.int32), 120)
        self.assert_lost(self.tracker.initialize(background, self.scene.corners()))
        for region in ((slice(175, 215), slice(275, 325)),
                       (slice(181, 192), slice(120, 500))):
            frame = np.full((400, 640), 120, dtype=np.uint8)
            frame[region] = self.scene.render()[region]
            self.assert_lost(self.tracker.initialize(frame, self.scene.corners()))
            self.assertEqual(self.tracker.snapshot()["reason"], "insufficient_spread")

    def test_low_texture_update_loses_calibration_and_loss_is_sticky(self):
        for value in (0, 127):
            with self.subTest(value=value):
                self.start()
                lost = self.tracker.update(np.full((400, 640), value, dtype=np.uint8))
                self.assert_lost(lost)
                for frame in (self.scene.render(), self.scene.render(self.scene.pose(1)), None):
                    self.assertEqual(self.tracker.update(frame), lost)
                self.start()

    def test_board_leaving_frame_invalidates_instead_of_clamping(self):
        self.start()
        lost = False
        for step in range(1, 25):
            quad = self.scene.quad + np.float32([8 * step, 0])
            result = self.tracker.update(self.scene.render(quad))
            if result["status"] == "lost":
                self.assert_lost(result)
                lost = True
            else:
                self.assertFalse(lost)
                self.assertLessEqual(quad[:, 0].max(), self.scene.width - 1)
                self.assert_tracking(result, quad)
        self.assertTrue(lost)
        self.assert_lost(self.tracker.update(self.scene.render()))

    def test_scene_cut_large_jump_and_major_occlusion_fail_closed(self):
        other = BoardScene(seed=240)
        frames = (other.render(),
                  self.scene.render(self.scene.quad + np.float32([140, -70])),
                  self.scene.render(self.scene.pose(1, angle=55, scale=0, dx=0, dy=0)),
                  self.scene.render(occluder=(95, 110, 370, 170)))
        for index, frame in enumerate(frames):
            with self.subTest(index=index):
                self.start()
                self.assert_lost(self.tracker.update(frame))
                self.assert_lost(self.tracker.update(self.scene.render()))

    def test_fast_motion_gate_allows_large_shape_preserving_translation(self):
        default = FretboardTracker()
        fast = FretboardTracker(fast_motion=True)
        frame = self.scene.render()
        for tracker in (default, fast):
            self.assert_tracking(tracker.initialize(frame, self.scene.corners()),
                                 self.scene.quad)
        translation = np.float32([[1, 0, 114], [0, 1, 0], [0, 0, 1]])
        quad = cv2.perspectiveTransform(
            self.scene.quad.reshape(-1, 1, 2), translation)[:, 0]
        self.assertFalse(default._plausible_motion(translation, quad))
        self.assertTrue(fast._plausible_motion(translation, quad))

    def test_invalid_calibration_clears_an_existing_track(self):
        good = self.scene.corners()
        invalid = (None, [], good[:3], [good[i] for i in (0, 2, 1, 3)],
                   [good[0]] * 4,
                   [{"x": 0, "y": 0}, {"x": .2, "y": .2},
                    {"x": .4, "y": .4}, {"x": .6, "y": .6}],
                   [{"x": .2, "y": .2}, {"x": .8, "y": .2},
                    {"x": .4, "y": .3}, {"x": .2, "y": .7}])
        for corners in invalid:
            with self.subTest(corners=corners):
                self.start()
                result = self.tracker.initialize(self.scene.render(), corners)
                self.assert_lost(result)
                self.assertEqual(result["reason"], "invalid_corners")
                self.assert_lost(self.tracker.update(self.scene.render()))
        for bad in (float("nan"), float("inf"), -.1, 1.1, "0.2", True, None):
            with self.subTest(bad=bad):
                corners = self.scene.corners()
                corners[0]["x"] = bad
                self.assert_lost(self.tracker.initialize(self.scene.render(), corners))

    def test_invalid_frames_and_resolution_changes_fail_closed(self):
        for frame in (None, [], np.zeros((0, 0), dtype=np.uint8),
                      np.zeros((400, 640, 3), dtype=np.uint8),
                      self.scene.render().astype(np.float32)):
            with self.subTest(shape=getattr(frame, "shape", None)):
                self.assert_lost(self.tracker.initialize(frame, self.scene.corners()))
                self.start()
                self.assert_lost(self.tracker.update(frame))
        self.start()
        result = self.tracker.update(cv2.resize(self.scene.render(), (320, 200)))
        self.assert_lost(result)
        self.assertEqual(result["reason"], "frame_size_changed")

    def test_readonly_strided_frames_are_unchanged_and_history_is_owned(self):
        original = self.scene.render()
        storage = np.zeros((400, 1280), dtype=np.uint8)
        storage[:, ::2] = original
        view = storage[:, ::2]
        view.flags.writeable = False
        self.start(view)
        np.testing.assert_array_equal(view, original)
        storage[:] = 0  # Caller buffer reuse must not overwrite tracker history.
        for step in range(1, 5):
            quad = self.scene.pose(step)
            frame = self.scene.render(quad)
            before = frame.copy()
            frame.flags.writeable = False
            self.assert_tracking(self.tracker.update(frame), quad)
            np.testing.assert_array_equal(frame, before)

    def test_snapshots_are_detached_and_manual_loss_requires_reinitialization(self):
        corners = self.scene.corners()
        result = self.start(corners=corners)
        result["corners"][0]["x"] = -100
        corners[0]["x"] = -200
        self.assert_tracking(self.tracker.snapshot(), self.scene.quad)
        self.assert_tracking(self.tracker.update(self.scene.render()), self.scene.quad)
        lost = self.tracker.invalidate("camera_stopped")
        self.assert_lost(lost)
        self.assertEqual(lost["reason"], "camera_stopped")
        self.assertEqual(self.tracker.update(self.scene.render()), lost)
        quad = self.scene.pose(10)
        self.assert_tracking(self.tracker.initialize(self.scene.render(quad),
                                                      self.scene.corners(quad)), quad)


if __name__ == "__main__":
    unittest.main()

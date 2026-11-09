import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from roi_fretboard import RoiFretboardTracker
from test_auto_fretboard import GuitarScene, CV_AVAILABLE

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipUnless(CV_AVAILABLE, "OpenCV/numpy unavailable")
class RoiFretboardTests(unittest.TestCase):
    @staticmethod
    def rectified_board(silver=True, occlusion=None, clutter=False):
        import cv2
        count = 12
        ratio = np.exp(-np.log(2.) / 12.)
        basis = ratio ** np.arange(count)
        xs = np.rint(8 + (basis - basis[0]) / (basis[-1] - basis[0]) * 304).astype(int)
        board = np.full((128, 320, 3), (48, 76, 108), np.uint8)
        grain = np.random.default_rng(41).integers(-5, 6, (128, 320, 1))
        board = np.clip(board.astype(np.int16) + grain, 0, 255).astype(np.uint8)
        for y in (16, 32, 48, 64, 80, 96):
            cv2.line(board, (0, y), (319, y), (22, 24, 27), 1)
        if silver:
            for x in xs:
                cv2.line(board, (int(x), 8), (int(x), 119), (205, 212, 218), 2)
        if occlusion is not None:
            x, width = occlusion
            cv2.rectangle(board, (x, 30), (x + width, 95), (88, 112, 150), -1)
        if clutter:
            cv2.rectangle(board, (132, 8), (158, 119), (224, 226, 228), -1)
            for x in (28, 74, 205, 276):
                cv2.line(board, (x, 8), (x, 119), (210, 212, 214), 2)
        return board, xs

    def test_silver_color_recovers_frets_when_supplied_gray_is_weak(self):
        import cv2
        self.tracker._dependencies()
        color, xs = self.rectified_board()
        weak_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        for x in xs:
            cv2.line(weak_gray, (int(x), 4), (int(x), 123), 88, 3)
        corners = [dict(x=0., y=0.), dict(x=1., y=0.),
                   dict(x=1., y=1.), dict(x=0., y=1.)]
        result = self.tracker._detect(weak_gray, corners, color)
        self.assertIsNotNone(result)
        found = np.array([line[0]["x"] * 319. for line in result[2]])
        np.testing.assert_allclose(found, xs, atol=2.2)
        self.assertGreaterEqual(self.tracker.timings["fret_color_support"], .45)

    def test_silver_spacing_rejects_broad_white_clutter(self):
        self.tracker._dependencies()
        color, _ = self.rectified_board(silver=False, clutter=True)
        response = self.tracker._silver_fret_response(color, 8, 119)
        self.assertIsNotNone(response)
        self.assertIsNone(self.tracker._color_fret_fit(response))

    def test_silver_sequence_survives_partial_hand_occlusion(self):
        self.tracker._dependencies()
        color, xs = self.rectified_board(occlusion=(105, 54))
        response = self.tracker._silver_fret_response(color, 8, 119)
        fit = self.tracker._color_fret_fit(response)
        self.assertIsNotNone(fit)
        np.testing.assert_allclose(fit[0], xs, atol=2.2)

    def test_hand_exclusion_mask_accepts_bgr_and_rgba(self):
        import cv2
        tracker = RoiFretboardTracker()
        rgb = np.zeros((120, 160, 3), np.uint8)
        rgb[30:80, 55:100] = (190, 140, 110)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        bgr_mask = tracker._hand_exclusion_mask(bgr, (60, 80))
        rgba_mask = tracker._hand_exclusion_mask(rgba, (60, 80))
        np.testing.assert_array_equal(bgr_mask, rgba_mask)
        self.assertGreater(np.count_nonzero(bgr_mask), 0)

    def test_tracking_loss_reacquires_from_color_in_the_same_frame(self):
        import cv2
        tracker = RoiFretboardTracker(
            fast_tracking=True, async_measurements=False)
        frame = self.scene.render()[0]
        color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        result = tracker.initialize(frame, self.corners(), color=color)
        self.assertEqual(result["roi_status"], "tracking")
        tracker._roi.invalidate("forced_loss")
        proposal = {"corners": self.corners(), "score": 100.}
        with patch("neck_color.find_color_neck_regions",
                   return_value=[proposal]), \
             patch.object(tracker, "_detect", return_value=None):
            result = tracker.update(frame, color=color)
        self.assertEqual(result["roi_status"], "tracking", result)
        self.assertTrue(result["flow"]["reacquired"])
        self.assertEqual(result["flow"]["reacquisitions"], 1)

    def test_real_candidate_color_path_preserves_fret_sequence(self):
        root = Path(__file__).resolve().parents[2]
        path = root / "diagnostics" / "actual_candidate_inspect.png"
        if not path.is_file():
            self.skipTest("real candidate diagnostic is not available")
        import cv2
        color = cv2.imread(str(path))
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        pixels = [(2, 303), (320, 219), (342, 295), (26, 391)]
        corners = [dict(x=x/(gray.shape[1]-1), y=y/(gray.shape[0]-1))
                   for x, y in pixels]
        result = self.tracker._detect(gray, corners, color)
        self.assertIsNotNone(result)
        self.assertEqual(len(result[1]), 6)
        self.assertGreaterEqual(len(result[2]), 13)
        self.assertGreaterEqual(self.tracker.timings["fret_color_support"], .35)

    def test_sparse_stable_neck_does_not_refresh_features_each_frame(self):
        tracker = RoiFretboardTracker(fast_tracking=True)
        tracker._dependencies()
        state = dict(status='tracking',corners=self.corners(),points=18,
                     inliers=18,inlier_ratio=1.)
        tracker._refresh_point_count = 18
        tracker._refresh_attempt_points = 18
        tracker._snapshot = tracker._empty('no_supported_lattice',self.corners(),'tracking')
        tracker._roi.update = Mock(return_value=state)
        with patch('roi_fretboard.FretboardTracker') as constructor, \
             patch.object(tracker,'_from_roi',side_effect=lambda *args: tracker._empty(
                 'no_supported_lattice',self.corners(),'tracking')):
            constructor.return_value.initialize.return_value = dict(status='lost')
            for _ in range(30):
                tracker.update(self.scene.render()[0])
            self.assertEqual(constructor.call_count,2)

    def test_depleted_neck_bounds_failed_feature_refresh_attempts(self):
        tracker = RoiFretboardTracker(fast_tracking=True)
        tracker._dependencies()
        tracker._refresh_point_count = 48
        tracker._refresh_attempt_points = 48
        tracker._snapshot = tracker._empty('no_supported_lattice',self.corners(),'tracking')
        tracker._roi.update = Mock(return_value=dict(status='tracking',corners=self.corners(),
            points=16,inliers=16,inlier_ratio=1.))
        with patch('roi_fretboard.FretboardTracker') as constructor, \
             patch.object(tracker,'_from_roi',side_effect=lambda *args: tracker._empty(
                 'no_supported_lattice',self.corners(),'tracking')):
            constructor.return_value.initialize.return_value = dict(status='lost')
            frame = self.scene.render()[0]
            for _ in range(14):
                tracker.update(frame)
            self.assertEqual(constructor.call_count,3)

    def test_sparse_attrition_can_refresh_before_minimum_point_loss(self):
        for baseline, counts in ((18,(18,17,16)),(48,(30,20,16))):
            tracker = RoiFretboardTracker(fast_tracking=True)
            tracker._dependencies()
            tracker._refresh_point_count = tracker._refresh_attempt_points = baseline
            tracker._snapshot = tracker._empty('no_supported_lattice',self.corners(),'tracking')
            tracker._roi.update = Mock(side_effect=[dict(status='tracking',corners=self.corners(),
                points=count,inliers=count,inlier_ratio=1.) for count in counts])
            with patch('roi_fretboard.FretboardTracker') as constructor, \
                 patch.object(tracker,'_from_roi',side_effect=lambda *args: tracker._empty(
                     'no_supported_lattice',self.corners(),'tracking')):
                constructor.return_value.initialize.return_value = dict(status='lost')
                frame = self.scene.render()[0]
                for _ in counts:
                    tracker.update(frame)
                self.assertEqual(constructor.call_count,2 if baseline==18 else 1)

    def test_async_measurement_does_not_block_flow_and_is_single_job(self):
        tracker = RoiFretboardTracker(fast_tracking=True,async_measurements=True)
        frame = self.scene.render()[0]
        with patch('roi_fretboard.threading.Thread') as thread, \
             patch.object(tracker,'_detect',side_effect=AssertionError('foreground measurement')):
            first = tracker.initialize(frame,self.corners())
            second = tracker.update(self.scene.render(dx=1)[0])
            self.assertEqual(first['roi_status'],'tracking')
            self.assertEqual(second['roi_status'],'tracking')
            self.assertEqual(first['strings'],[])
            self.assertEqual(thread.call_count,1)
            # Finish the captured job to release the globally bounded slot.
            thread.call_args.kwargs['target']()
        self.assertFalse(tracker._measurement_busy)

    def test_failed_background_measurement_backs_off_and_new_neck_resets(self):
        tracker = RoiFretboardTracker(fast_tracking=True,async_measurements=True)
        frame = self.scene.render()[0]
        with patch('roi_fretboard.threading.Thread') as thread, \
             patch.object(RoiFretboardTracker,'_detect',return_value=None), \
             patch('roi_fretboard.time.monotonic',return_value=10.):
            tracker._submit_measurement(frame,self.corners(),10.)
            thread.call_args.kwargs['target']()
            self.assertEqual(tracker._measurement_failures,1)
            tracker._submit_measurement(frame,self.corners(),10.75)
            self.assertEqual(thread.call_count,1)
            tracker._submit_measurement(frame,self.corners(),11.6)
            self.assertEqual(thread.call_count,2)
            thread.call_args.kwargs['target']()
        tracker.invalidate('new_neck')
        self.assertEqual(tracker._measurement_failures,0)

    def test_async_result_is_cleared_after_invalidation(self):
        tracker = RoiFretboardTracker(fast_tracking=True,async_measurements=True)
        with patch('roi_fretboard.threading.Thread') as thread:
            tracker.initialize(self.scene.render()[0],self.corners())
            target = thread.call_args.kwargs['target']
            tracker.invalidate('camera_changed')
            target()
        self.assertIsNone(tracker._measurement_result)
        self.assertFalse(tracker._measurement_busy)

    def test_async_result_requires_current_frame_line_support(self):
        tracker = RoiFretboardTracker(fast_tracking=True,async_measurements=True)
        frame = self.scene.render()[0]
        with patch('roi_fretboard.threading.Thread') as thread:
            tracker.initialize(frame,self.corners())
            thread.call_args.kwargs['target']()
        self.assertIsNotNone(tracker._measurement_result)
        state = tracker._roi.snapshot()
        blank = tracker._from_roi(np.zeros_like(frame),state,True)
        self.assertNotEqual(blank['status'],'tracking')
        self.assertEqual(blank['strings'],[])

    def test_async_valid_measurement_maps_to_the_current_flow_frame(self):
        tracker = RoiFretboardTracker(fast_tracking=True,async_measurements=True)
        frame = self.scene.render()[0]
        with patch('roi_fretboard.threading.Thread') as thread:
            tracker.initialize(frame,self.corners())
            thread.call_args.kwargs['target']()
        result = tracker.update(self.scene.render(dx=1)[0])
        self.assertEqual(result['roi_status'],'tracking')
        self.assertEqual(result['status'],'tracking')
        self.assertEqual(result['stage'],'tracked_lattice')
        self.assertEqual(result['string_count'],6)

    def test_partial_finger_occlusion_preserves_observed_fret_centers(self):
        self.tracker._dependencies()
        import cv2
        basis = np.exp(np.log(2)*np.arange(7)/12.)
        xs = np.rint((basis-basis[0])/(basis[-1]-basis[0])*290+10).astype(int)
        for sign in (1,-1):
            image = np.full((128,320),120,np.uint8)
            for x in xs:
                cv2.line(image,(int(x),8),(int(x),118),120+sign*70,2)
            cv2.rectangle(image,(85,40),(245,61),120,-1)
            result = self.tracker._ridge_frets(image,8,119)
            self.assertIsNotNone(result)
            np.testing.assert_allclose(result[0],xs,atol=1.)
        for regular in (True,False):
            image = np.full((128,320),120,np.uint8)
            for x in (np.linspace(10,300,7) if regular else np.delete(xs,3)):
                cv2.line(image,(int(x),8),(int(x),118),220,2)
            self.assertIsNone(self.tracker._ridge_frets(image,8,119))

    def test_short_fret_run_uses_scale_relative_curvature_gate(self):
        self.tracker._dependencies()
        basis = np.exp(-np.log(2)*np.arange(5)/12.)
        positions = (basis-basis[0])/(basis[-1]-basis[0])*260+20
        self.assertIsNotNone(self.tracker._fret_spacing(np.rint(positions)))
        self.assertIsNone(self.tracker._fret_spacing(np.linspace(20,280,5)))

    def test_string_lattice_ignores_stronger_unrelated_ridges(self):
        self.tracker._dependencies()
        wires = [(y,30.) for y in (18,34,50,66,82,98)]
        clutter = [(y,45.) for y in (26,57,72,110)]
        fit = self.tracker._six_strings(wires+clutter)
        self.assertIsNotNone(fit)
        np.testing.assert_allclose(fit[1],[18,34,50,66,82,98])

    def test_missing_string_is_not_filled_by_periodicity(self):
        self.tracker._dependencies()
        self.assertIsNone(self.tracker._six_strings([(y,30.) for y in (18,34,50,82,98)]))

    def test_faint_string_ridges_with_bright_frets_and_border(self):
        self.tracker._dependencies()
        import cv2
        for polarity in (1,-1):
            image = np.full((128,320),110,np.uint8)
            for x in self.scene.fret_x/1100*319:
                cv2.line(image,(int(x),2),(int(x),125),245,2)
            for y in (18,34,50,66,82,98):
                cv2.line(image,(0,y),(319,y),110+polarity*25,1)
            cv2.line(image,(0,115),(319,115),250,3)
            fit = self.tracker._ridge_strings(image)
            self.assertIsNotNone(fit,polarity)
            np.testing.assert_allclose(fit[1],[18,34,50,66,82,98],atol=1.)

    def test_hough_recovers_frets_when_global_profiles_lose_wires(self):
        import cv2
        self.tracker._dependencies()
        color, xs = self.rectified_board()
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        fit = self.tracker._hough_frets(gray, 3, 124)
        self.assertIsNotNone(fit)
        found = np.asarray(fit[0])
        self.assertGreaterEqual(len(found), 5)
        self.assertTrue(all(np.min(np.abs(xs-value)) <= 4. for value in found))


    def test_flow_point_configuration(self):
        from unittest.mock import patch
        for count in (24, 32, 48):
            with patch.dict("os.environ", {"FRETLENS_FLOW_POINTS": str(count)}):
                self.assertEqual(RoiFretboardTracker()._FLOW_POINTS, count)
        for invalid in ("0", "64", "bad"):
            with patch.dict("os.environ", {"FRETLENS_FLOW_POINTS": invalid}):
                with self.assertRaises(ValueError):
                    RoiFretboardTracker()

    def test_fast_path_motion_periodic_refresh_and_occlusion(self):
        from unittest.mock import patch
        tracker = RoiFretboardTracker(fast_tracking=True)
        with patch("roi_fretboard.time.monotonic", return_value=10):
            tracker.initialize(self.scene.render()[0], self.corners())
            stages = []
            for step in range(1,6):
                frame, expected, _ = self.scene.render(angle=step*.3, dx=step*.4)
                result = tracker.update(frame)
                self.assertEqual(result["status"], "tracking", result)
                self.assert_strings_near(result, expected)
                stages.append(result["stage"])
            self.assertIn("tracked_lattice", stages)
            lost = tracker.update(np.zeros_like(frame))
            self.assertEqual(lost["strings"], [])
            self.assertNotEqual(lost["status"], "tracking")

    def test_fast_searching_roi_tracks_flow_but_limits_lattice_measurement(self):
        tracker = RoiFretboardTracker(fast_tracking=True)
        tracker.initialize(self.scene.render()[0], self.corners())
        tracker._snapshot = tracker._empty("no_supported_lattice", tracker._snapshot["roi_corners"], "tracking")
        tracker._roi.update = Mock(return_value={"status": "tracking", "corners": tracker._snapshot["roi_corners"],
                                                  "points": 40, "inliers": 40, "inlier_ratio": 1.0})
        tracker._measure_at = time.monotonic() - 1
        with patch.object(tracker, "_detect", return_value=None) as detect:
            first = tracker.update(self.scene.render()[0])
            second = tracker.update(self.scene.render()[0])
        self.assertEqual(first["strings"], [])
        self.assertEqual(second["strings"], [])
        self.assertEqual(detect.call_count, 1)
        self.assertEqual(first["roi_status"], "tracking")
        self.assertEqual(second["roi_status"], "tracking")

    def test_fast_measurement_is_rate_limited_while_default_measures_each_update(self):
        fast = RoiFretboardTracker(fast_tracking=True)
        full = RoiFretboardTracker(fast_tracking=False)
        frame = self.scene.render()[0]
        corners = self.corners()
        fast.initialize(frame, corners)
        full.initialize(frame, corners)
        fast._measure_at = time.monotonic() - 1
        with patch.object(fast, "_detect", return_value=None) as fast_detect, \
             patch.object(full, "_detect", return_value=None) as full_detect:
            fast._snapshot = fast._empty("no_supported_lattice", fast._snapshot["roi_corners"], "tracking")
            fast._roi.update = Mock(return_value={"status": "tracking", "corners": fast._snapshot["roi_corners"],
                                                   "points": 40, "inliers": 40, "inlier_ratio": 1.0})
            full._roi.update = Mock(return_value={"status": "tracking", "corners": full._snapshot["roi_corners"],
                                                   "points": 40, "inliers": 40, "inlier_ratio": 1.0})
            fast.update(frame)
            fast.update(frame)
            full.update(frame)
            full.update(frame)
        self.assertEqual(fast_detect.call_count, 1)
        self.assertEqual(full_detect.call_count, 2)

    def setUp(self):
        self.scene = GuitarScene()
        self.tracker = RoiFretboardTracker()

    def corners(self, quad=None):
        quad = self.scene.quad if quad is None else quad
        return [{"x": float(x / (self.scene.width - 1)), "y": float(y / (self.scene.height - 1))}
                for x, y in quad]

    def assert_strings_near(self, result, expected):
        for actual, target in zip(result["strings"], expected):
            points = np.array([[actual[key]["x"] * (self.scene.width - 1),
                                actual[key]["y"] * (self.scene.height - 1)]
                               for key in ("start", "end")])
            vector = target[1] - target[0]
            distance = np.abs(vector[0] * (points[:, 1] - target[0, 1]) -
                              vector[1] * (points[:, 0] - target[0, 0])) / np.linalg.norm(vector)
            self.assertLess(float(distance.max()), 4.0, (actual, target))

    def test_initializes_and_tracks_current_lattice_after_motion(self):
        frame, expected, _ = self.scene.render()
        result = self.tracker.initialize(frame, self.corners())
        self.assertEqual(result["roi_status"], "tracking")
        self.assertIn(result["stage"], ("roi_tracking", "lattice"))
        for step in range(1, 4):
            frame, expected, _ = self.scene.render(angle=step * 2, dx=step * 2, dy=step)
            result = self.tracker.update(frame)
            self.assertEqual(result["roi_status"], "tracking", result)
            self.assertEqual(result["status"], "tracking", result)
            self.assertEqual(result["stage"], "lattice")
            self.assertEqual(result["string_count"], 6)
            self.assertGreaterEqual(result["fret_count"], 5)
            self.assertTrue(all(line["observed"] for line in result["strings"] + result["frets"]))
            self.assert_strings_near(result, expected)
            json.dumps(result, allow_nan=False)

    def test_unmeasured_strings_are_explicitly_estimated_six_lanes(self):
        tracker = RoiFretboardTracker()
        strings = [({"x": float(index / 5), "y": float(row / 5)},
                    {"x": float(index / 5 + .7), "y": float(row / 5)})
                   for row, index in enumerate(range(6))]
        frets = [({"x": float(index / 5), "y": 0.},
                  {"x": float(index / 5), "y": 1.}) for index in range(6)]
        result = tracker._lattice_result(
            (.8, strings, frets), self.corners(), True)
        self.assertEqual(result["status"], "tracking")
        self.assertEqual(result["string_observation"],
                         "estimated_live_neck_width")
        self.assertTrue(all(line["observed"] is False and line["estimated"] is True
                            for line in result["strings"]))
        self.assertTrue(all(line["observed"] is True
                            for line in result["frets"]))

    def test_expired_projection_is_rejected_without_losing_current_roi(self):
        tracker = RoiFretboardTracker(fast_tracking=True)
        tracker.initialize(self.scene.render()[0], self.corners())
        tracker._full_at = time.monotonic() - 2.
        with patch.object(tracker, "_detect", return_value=None):
            result = tracker.update(self.scene.render(dx=1)[0])
        self.assertEqual(result["status"], "searching", result)
        self.assertEqual(result["reason"], "no_supported_lattice", result)
        self.assertEqual(result["roi_status"], "tracking", result)
        self.assertEqual(len(result["roi_corners"]), 4, result)
        self.assertEqual(result["strings"], [])
        self.assertEqual(result["frets"], [])

    def test_recent_projection_keeps_roi_during_bounded_remeasurement_window(self):
        tracker = RoiFretboardTracker(
            fast_tracking=True, async_measurements=False)
        tracker.initialize(self.scene.render()[0], self.corners())
        tracker._full_at = time.monotonic() - 1.0
        with patch.object(tracker, "_detect", return_value=None), \
             patch.object(tracker, "_submit_measurement") as submit:
            result = tracker.update(self.scene.render(dx=1)[0])
        self.assertEqual(result["roi_status"], "tracking", result)
        self.assertEqual(result["status"], "tracking", result)
        self.assertEqual(result["stage"], "tracked_lattice", result)
        self.assertGreaterEqual(result["lattice_age_ms"], 950)
        self.assertEqual(result["lattice_ttl_ms"], 1750)
        submit.assert_not_called()

    def test_blank_update_loses_roi_and_geometry(self):
        self.tracker.initialize(self.scene.render()[0], self.corners())
        result = self.tracker.update(np.zeros((400, 640), np.uint8))
        self.assertEqual(result["status"], "searching")
        self.assertEqual(result["roi_status"], "lost")
        self.assertIsNone(result["roi_corners"])

    def test_long_motion_refreshes_features_and_still_rejects_occlusion(self):
        self.tracker.initialize(self.scene.render()[0], self.corners())
        for step in range(1, 33):
            frame = self.scene.render(angle=2 * np.sin(step / 8),
                                      dx=4 * np.sin(step / 7), dy=2 * np.sin(step / 9))[0]
            result = self.tracker.update(frame)
            self.assertEqual(result["roi_status"], "tracking", result)
        self.assertGreaterEqual(result["flow"]["feature_refreshes"], 2)
        self.assertLessEqual(result["flow"]["points"], 48)
        lost = self.tracker.update(np.zeros_like(frame))
        self.assertEqual(lost["roi_status"], "lost")
        self.assertEqual(lost["strings"], [])


if __name__ == "__main__":
    unittest.main()

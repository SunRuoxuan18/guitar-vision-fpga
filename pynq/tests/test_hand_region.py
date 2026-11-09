import sys
import json
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_region import HandRegionTracker


def neck():
    return dict(roi_status="locked", roi_corners=[dict(x=.18, y=.42), dict(x=.82, y=.42),
                                                     dict(x=.82, y=.58), dict(x=.18, y=.58)])


def scene(hand_x=None):
    image = np.full((240, 320, 4), (45, 75, 95, 255), np.uint8)
    # Neck texture gives sparse LK features without becoming a candidate.
    for x in range(58, 264, 13):
        cv2.line(image, (x, 101), (x, 139), (150, 105, 65, 255), 1)
    if hand_x is not None:
        cv2.ellipse(image, (hand_x, 120), (21, 16), 0, 0, 360, (190, 135, 80, 255), -1)
        for dx in (-10, 0, 10):
            cv2.circle(image, (hand_x + dx, 120), 3, (220, 165, 105, 255), -1)
    return image


class HandRegionTests(unittest.TestCase):
    def test_native_component_matches_reference_on_motion_and_distractors(self):
        from neck_region import native_backend
        if not hasattr(native_backend(), 'tracking_skin_component'):
            self.skipTest('board native component library unavailable')
        quad = np.float32([[28,50],[131,50],[131,70],[28,70]])
        for x in (None, 70, 100, 120):
            image = cv2.resize(scene(x*2 if x else None), (160,120))
            for prior in (None,np.float32([75,60])):
                for side in ('any','right'):
                    actual = HandRegionTracker._skin_candidate(image,quad,prior,side)
                    with patch('neck_region.native_backend',return_value=None), \
                         patch('tracking_native.library',return_value=None):
                        expected = HandRegionTracker._skin_candidate(image,quad,prior,side)
                    self.assertEqual(actual[0] is None,expected[0] is None)
                    if expected[0] is not None:
                        np.testing.assert_allclose(actual[0],expected[0],atol=1e-5)
                        np.testing.assert_allclose(actual[2],expected[2],atol=1e-5)
                        self.assertAlmostEqual(actual[1],expected[1],places=5)

    def test_batched_cv_skin_matches_reference_for_rotated_neck(self):
        from tracking_native import library
        if library() is None:
            self.skipTest('board batched OpenCV library unavailable')
        quad = np.float32([[28,50],[131,50],[131,70],[28,70]])
        for angle in (-25.,-10.,0.,20.):
            transform = cv2.getRotationMatrix2D((80,60),angle,1.)
            mapped = cv2.transform(quad[None],transform)[0]
            for x in (None,70,100,120):
                image=cv2.warpAffine(cv2.resize(scene(x*2 if x else None),(160,120)),
                    transform,(160,120),borderMode=cv2.BORDER_REFLECT)
                for prior in (None,np.float32([75,60])):
                    actual=HandRegionTracker._skin_candidate(image,mapped,prior,'right')
                    with patch('tracking_native.library',return_value=None), \
                         patch('neck_region.native_backend',return_value=None):
                        expected=HandRegionTracker._skin_candidate(image,mapped,prior,'right')
                    self.assertEqual(actual[0] is None,expected[0] is None,(angle,x))
                    if expected[0] is not None:
                        np.testing.assert_allclose(actual[0],expected[0],atol=1e-5)
                        np.testing.assert_allclose(actual[2],expected[2],atol=1e-5)
                        self.assertAlmostEqual(actual[1],expected[1],places=5)

    def test_fast_right_hand_reversals_use_current_evidence_without_lk(self):
        tracker = HandRegionTracker(target_side='right')
        with patch('hand_region.cv2.calcOpticalFlowPyrLK', side_effect=AssertionError('unnecessary LK')):
            for sequence, x in enumerate((180, 245, 175, 240, 185, 245)*3, 1):
                result = self.update(tracker, scene(x), sequence)
                self.assertEqual(result['status'], 'tracking', (sequence, result))
                self.assertAlmostEqual(self.center_x(result), x, delta=10)
        self.assertEqual(self.update(tracker, scene(None), 19)['regions'], [])

    def test_right_side_target_ignores_larger_left_hand(self):
        tracker = HandRegionTracker(target_side='right')
        for sequence,x in enumerate((205,225,240,215),1):
            image = scene(x)
            cv2.ellipse(image,(105,120),(32,25),0,0,360,(190,135,80,255),-1)
            result = self.update(tracker,image,sequence)
            self.assertEqual(result['status'],'tracking',result)
            self.assertAlmostEqual(self.center_x(result),x,delta=10)
        self.assertEqual(self.update(tracker,scene(105),5)['regions'],[])

    def test_padded_red_box_contains_hand_above_and_below_neck(self):
        image = scene(215)
        cv2.ellipse(image,(215,120),(25,30),0,0,360,(190,135,80,255),-1)
        result = self.update(HandRegionTracker(target_side='right'),image)
        self.assertEqual(result['status'],'tracking',result)
        box = np.float32([[p['x']*319,p['y']*239] for p in result['regions'][0]['corners']])
        for point in ((215.,90.),(215.,150.),(190.,120.),(240.,120.)):
            self.assertGreater(cv2.pointPolygonTest(box,point,True),0)

    def test_right_only_target_does_not_template_track_into_left_half(self):
        tracker = HandRegionTracker(target_side='right')
        self.assertEqual(self.update(tracker,scene(180),1)['status'],'tracking')
        image = scene(145)
        gray = cv2.cvtColor(image,cv2.COLOR_RGBA2GRAY)
        image[:,:,:3] = gray[:,:,None]
        self.assertEqual(self.update(tracker,image,2)['regions'],[])

    @staticmethod
    def center_x(result):
        return np.mean([p['x'] for p in result['regions'][0]['corners']])*319

    def test_fast_hand_jump_reacquires_without_neck_restart(self):
        tracker = HandRegionTracker()
        self.assertEqual(self.update(tracker, scene(96), 1)['status'], 'tracking')
        result = self.update(tracker, scene(224), 2)
        self.assertEqual(result['status'], 'tracking', result)
        self.assertAlmostEqual(self.center_x(result), 224, delta=8)

    def test_hand_moves_independently_of_stationary_neck_each_frame(self):
        tracker = HandRegionTracker()
        for sequence, x in enumerate([100, 120, 150, 185, 220, 195, 160, 125, 100]*3, 1):
            result = self.update(tracker, scene(x), sequence)
            self.assertEqual(result['status'], 'tracking', (sequence, result))
            self.assertAlmostEqual(self.center_x(result), x, delta=8)

    def test_current_texture_bridges_short_colour_miss_but_not_absence(self):
        tracker = HandRegionTracker()
        with patch('hand_region.time.monotonic', return_value=10.):
            self.update(tracker, scene(128), 1)
        for sequence, x in ((2, 140), (3, 152)):
            image = scene(x)
            mask = image[:, :, 0] > 180
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            image[mask, :3] = gray[mask, None]
            with patch('hand_region.time.monotonic', return_value=10.+sequence*.1):
                result = self.update(tracker, image, sequence)
            self.assertEqual(result['status'], 'tracking', result)
            self.assertEqual(result['regions'][0]['source'], 'ps_local_template')
            self.assertAlmostEqual(self.center_x(result), x, delta=8)
        self.assertEqual(self.update(tracker, scene(None), 4)['regions'], [])

    def test_colour_miss_cannot_keep_box_after_frame_gap(self):
        tracker = HandRegionTracker()
        with patch('hand_region.time.monotonic', return_value=10.):
            self.update(tracker, scene(128), 1)
        image = scene(128)
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        image[:, :, :3] = gray[:, :, None]
        with patch('hand_region.time.monotonic', return_value=10.8):
            self.assertEqual(self.update(tracker, image, 2)['regions'], [])

    def update(self, tracker, image, sequence=1, analysis=None, stream="camera-a", generation=1):
        return tracker.update(image, cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY),
            dict(stream_id=stream, generation=generation, sequence=sequence), neck() if analysis is None else analysis)

    def test_waits_sequentially_for_locked_neck(self):
        result = self.update(HandRegionTracker(), scene(130), analysis={"roi_status": "searching"})
        self.assertEqual(result["status"], "waiting_for_neck")
        self.assertEqual(result["regions"], [])

    def test_textured_handlike_motion_yields_neck_band_and_ps_label(self):
        tracker = HandRegionTracker()
        self.assertEqual(self.update(tracker, scene(128), 1)["status"], "tracking")
        result = self.update(tracker, scene(145), 2)
        self.assertEqual(result["status"], "tracking")
        self.assertEqual(result["inference_location"], "ps_linux")
        self.assertEqual(result["regions"][0]["source"], "ps_skin_motion")
        corners = result["regions"][0]["corners"]
        self.assertEqual(len(corners), 4)
        self.assertTrue(all(0. <= point[axis] <= 1. for point in corners for axis in ("x", "y")))
        self.assertLess(max(point["y"] for point in corners) - min(point["y"] for point in corners), .35)
        json.dumps(result)
        self.assertLess(result["processing_ms"], 100)

    def test_bright_dark_and_moving_background_do_not_make_boxes(self):
        for color in ((245, 245, 245, 255), (5, 5, 5, 255), (45, 75, 95, 255)):
            tracker = HandRegionTracker()
            image = np.full((240, 320, 4), color, np.uint8)
            if color[0] == 45:
                cv2.rectangle(image, (0, 0), (50, 239), (205, 145, 85, 255), -1)
            result = self.update(tracker, image)
            self.assertEqual(result["regions"], [], result)

    def test_warm_wood_colored_neck_is_not_a_hand_candidate(self):
        image = scene(None)
        cv2.rectangle(image, (58, 101), (262, 139), (190, 135, 80, 255), -1)
        result = self.update(HandRegionTracker(), image)
        self.assertEqual(result["regions"], [], result)

    def test_real_fretboard_fixture_selects_fretting_hand_not_left_clothes(self):
        fixture = Path(__file__).resolve().parents[2] / "diagnostics" / "user_fretboard_frame.png"
        self.assertTrue(fixture.is_file(), fixture)
        image = cv2.cvtColor(cv2.imread(str(fixture), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGBA)
        analysis = dict(roi_status="locked", roi_corners=[
            dict(x=0.0, y=.56), dict(x=.79, y=.35), dict(x=.79, y=.50), dict(x=0.0, y=.84)])
        result = self.update(HandRegionTracker(), image, analysis=analysis)
        self.assertEqual(result["status"], "tracking", result)
        center_x = sum(point["x"] for point in result["regions"][0]["corners"]) / 4
        self.assertGreater(center_x, .55, result)
        self.assertLess(center_x, .78, result)

    def test_automatic_roi_fixture_uses_polygon_contact(self):
        fixture = Path(__file__).resolve().parents[2] / "diagnostics" / "user_fretboard_frame.png"
        from neck_region import find_neck_regions
        image = cv2.cvtColor(cv2.resize(cv2.imread(str(fixture)),(640,480)), cv2.COLOR_BGR2RGBA)
        # Use the integer PL coefficients: OpenCV's rounded gray can change
        # borderline Canny segments and conceal an acquisition regression.
        rgb = image[:,:,:3].astype(np.uint32)
        gray = ((77*rgb[:,:,0]+150*rgb[:,:,1]+29*rgb[:,:,2]+128)>>8).astype(np.uint8)
        proposals = find_neck_regions(gray)
        self.assertTrue(proposals)
        from vision_pc_bridge import LatticeSearch
        analysis = dict(roi_corners=LatticeSearch.select_proposal(proposals,gray.shape))
        result = self.update(HandRegionTracker(), image, 45,
                             analysis=dict(roi_status="locked", roi_corners=analysis["roi_corners"]))
        self.assertEqual(result["status"], "tracking", result)
        region = result["regions"][0]
        center_x = sum(point["x"] for point in region["corners"]) / 4
        self.assertGreater(center_x, .55, result)
        self.assertLess(center_x, .78, result)
        roi = np.float32([[p["x"] * image.shape[1], p["y"] * image.shape[0]]
                          for p in analysis["roi_corners"]])
        # The center stays local to the neck, but the display now includes
        # the full hand outside the neck rather than clipping to its polygon.
        center = np.mean([[p['x']*image.shape[1],p['y']*image.shape[0]] for p in region['corners']],axis=0)
        self.assertGreaterEqual(cv2.pointPolygonTest(roi,tuple(center.astype(float)),True),-35.)

    def test_temporal_prediction_rejects_distant_skin_blob(self):
        tracker = HandRegionTracker()
        self.assertEqual(self.update(tracker, scene(145), 1)["status"], "tracking")
        moved = scene(160)
        cv2.ellipse(moved, (35, 120), (22, 15), 0, 0, 360, (220, 165, 105, 255), -1)
        result = self.update(tracker, moved, 2)
        if result["regions"]:
            center_x = sum(point["x"] for point in result["regions"][0]["corners"]) / 4
            self.assertGreater(center_x, .30, result)

    def test_stream_reset_and_occlusion_clear_state(self):
        tracker = HandRegionTracker()
        self.assertEqual(self.update(tracker, scene(130), 1)["status"], "tracking")
        cleared = self.update(tracker, scene(None), 2)
        self.assertEqual(cleared["regions"], [])
        reset = self.update(tracker, scene(150), 1, stream="camera-b", generation=2)
        self.assertEqual(reset["status"], "tracking")
        self.assertEqual(reset["stream_id"], "camera-b")


if __name__ == "__main__":
    unittest.main()

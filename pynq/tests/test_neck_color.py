"""Colour candidates remain proposals, and unrelated strips must not acquire."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neck_color import (
    HEADSTOCK_END_RAISE, _clip_quad_to_crossbars, _fit_borders, _headstock_side,
    _regular_crossbar_run, find_color_neck_regions)
from roi_fretboard import RoiFretboardTracker


def strip(crossbars=True):
    image = np.full((480, 640, 3), (140, 190, 210), np.uint8)
    cv2.rectangle(image, (35, 190), (605, 265), (45, 40, 50), -1)
    if crossbars:
        for x in (65, 105, 148, 194, 243, 295, 350, 409, 471, 537, 586):
            cv2.line(image, (x, 191), (x, 264), (210, 215, 215), 2)
    return image


class ColorNeckTests(unittest.TestCase):
    def test_invalid_inputs_and_limits(self):
        for value in (None, np.zeros((480, 640), np.uint8), np.zeros((20, 30, 3), np.uint8),
                      np.zeros((480, 640, 3), np.float32)):
            with self.assertRaises(ValueError):
                find_color_neck_regions(value)
        for limit in (0, 7, True, 1.5):
            with self.assertRaises(ValueError):
                find_color_neck_regions(strip(), limit)

    def test_blank_and_plain_strip_do_not_acquire(self):
        self.assertEqual(find_color_neck_regions(np.zeros((480, 640, 3), np.uint8)), [])
        self.assertEqual(find_color_neck_regions(strip(False)), [])

    def test_disconnected_collinear_contours_are_not_duplicates(self):
        lines = np.float32([[[0, 20], [60, 20]], [[160, 20], [210, 20]]])
        fitted = _fit_borders(lines, 240)
        self.assertEqual(len(fitted), 2)
        self.assertTrue(any(np.allclose(line, lines[1]) for line in fitted))

    def test_headstock_boundary_uses_widening_fret_spacing(self):
        rising = [.10, .16, .225, .295, .37, .45, .535, .625, .72, .82]
        falling = [.12, .22, .31, .39, .46, .525, .585, .64, .69, .735]
        self.assertEqual(_headstock_side(rising), 0)
        self.assertEqual(_headstock_side(falling), 1)
        quad = np.float32([[.1, .2], [1., .2], [1., .8], [.1, .8]])
        clipped = _clip_quad_to_crossbars(quad, rising)
        parameter = clipped[0, 0] / (quad[1, 0] - quad[0, 0])
        self.assertGreaterEqual(parameter, .0999)
        reversed_quad = np.float32([[0., .2], [.9, .2], [.9, .8], [0., .8]])
        clipped = _clip_quad_to_crossbars(reversed_quad, falling)
        parameter = (clipped[1, 0] - reversed_quad[0, 0]) / (
            reversed_quad[1, 0] - reversed_quad[0, 0])
        self.assertLessEqual(parameter, .7351)

    def test_only_headstock_crossbar_is_raised_slightly(self):
        falling = [.12, .22, .31, .39, .46, .525, .585, .64, .69, .735]
        quad = np.float32([[0., .20], [.90, .20], [.90, .80], [0., .80]])
        clipped = _clip_quad_to_crossbars(quad, falling)
        self.assertEqual(_headstock_side(falling), 1)
        self.assertAlmostEqual(float(clipped[1, 1]), .20 - HEADSTOCK_END_RAISE)
        self.assertAlmostEqual(float(clipped[2, 1]), .80 - HEADSTOCK_END_RAISE)
        self.assertAlmostEqual(float(clipped[0, 1]), .20)
        self.assertAlmostEqual(float(clipped[3, 1]), .80)

        rising = [.10, .16, .225, .295, .37, .45, .535, .625, .72, .82]
        quad = np.float32([[.10, .20], [1., .20], [1., .80], [.10, .80]])
        clipped = _clip_quad_to_crossbars(quad, rising)
        self.assertEqual(_headstock_side(rising), 0)
        self.assertAlmostEqual(float(clipped[0, 1]), .20 - HEADSTOCK_END_RAISE)
        self.assertAlmostEqual(float(clipped[3, 1]), .80 - HEADSTOCK_END_RAISE)
        self.assertAlmostEqual(float(clipped[1, 1]), .20)
        self.assertAlmostEqual(float(clipped[2, 1]), .80)

    def test_headstock_clutter_does_not_extend_the_nut_boundary(self):
        positions = [
            .0353, .0667, .0941, .1294, .1608, .1961, .2314, .2902,
            .3569, .4000, .4431, .4941, .5451, .7176, .7804, .8431, .9176]
        regular = _regular_crossbar_run(positions)
        self.assertIsNotNone(regular)
        self.assertAlmostEqual(float(regular[-1]), .5451, places=4)
        self.assertEqual(_headstock_side(positions), 1)
        quad = np.float32([[.20, .78], [.90, .44], [.92, .60], [.22, .89]])
        clipped = _clip_quad_to_crossbars(quad, positions)
        parameter = (clipped[1, 0] - quad[0, 0]) / (quad[1, 0] - quad[0, 0])
        self.assertGreaterEqual(parameter, .5450)
        self.assertLessEqual(parameter, .5452)

    def test_hand_occlusion_without_colour_nut_stops_at_confirmed_fret(self):
        positions = [
            .0353, .0667, .0941, .1294, .1608, .1961, .2314, .2902,
            .3569, .4000, .4431, .4941, .5451, .7176, .7804, .8431, .9176]
        regular = _regular_crossbar_run(positions)
        self.assertIsNotNone(regular)
        self.assertAlmostEqual(float(regular[-1]), .5451, places=4)
        self.assertEqual(_headstock_side(positions), 1)
        quad = np.float32([[0., .2], [1., .2], [1., .8], [0., .8]])
        clipped = _clip_quad_to_crossbars(quad, positions)
        parameter = clipped[1, 0]
        self.assertGreaterEqual(parameter, .5450)
        self.assertLessEqual(parameter, .5452)

    def test_full_width_colour_ridge_marks_nut_after_hand_gap(self):
        positions = [
            .10, .17, .235, .295, .35, .40, .445, .485, .52,
            .68, .74, .80, .86]
        support = [.70] * 9 + [.63, 1., 1., .47]
        strength = [30.] * 9 + [26., 50., 45., 22.]
        self.assertEqual(_headstock_side(positions), 1)
        quad = np.float32([[0., .2], [1., .2], [1., .8], [0., .8]])
        clipped = _clip_quad_to_crossbars(quad, positions, support, strength)
        parameter = clipped[1, 0]
        self.assertGreaterEqual(parameter, .7999)
        self.assertLessEqual(parameter, .8001)

    def test_live_headstock_roi_stops_before_locking_nut_hardware(self):
        root = Path(__file__).resolve().parents[2]
        image = cv2.imread(str(root / 'diagnostics/endpoint_tracking_20261030/live_headstock_issue.jpg'))
        self.assertIsNotNone(image)
        candidates = find_color_neck_regions(image, limit=1)
        self.assertTrue(candidates)
        polygon = np.float32([[point['x'] * (image.shape[1] - 1),
                               point['y'] * (image.shape[0] - 1)]
                              for point in candidates[0]['corners']])
        # The nut-side edge must include the last visible fret and stop before
        # the locking-nut studs and tuners.
        self.assertGreater(polygon[1, 0], 515.)
        self.assertLess(polygon[1, 0], 545.)
        self.assertLess(cv2.pointPolygonTest(polygon, (555., 205.), False), 0)

    def test_live_headstock_candidate_initializes_roi_flow(self):
        root = Path(__file__).resolve().parents[2]
        image = cv2.imread(str(root / 'diagnostics/endpoint_tracking_20261030/live_headstock_issue.jpg'))
        self.assertIsNotNone(image)
        candidates = find_color_neck_regions(image, limit=1)
        self.assertTrue(candidates)
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (320, 200))
        result = RoiFretboardTracker(fast_tracking=True).initialize(
            gray, candidates[0]['corners'], image)
        self.assertEqual(result['roi_status'], 'tracking', result)
        self.assertEqual(result['flow']['sampling'], 'endpoint_bands')
        self.assertGreaterEqual(result['flow']['points'], 16)

    def test_first_search_uses_segmented_contours_without_hough_or_canny(self):
        with patch('cv2.HoughLinesP', side_effect=AssertionError('Hough fallback')), \
                patch('cv2.Canny', side_effect=AssertionError('Canny fallback')), \
                patch('neck_color.cv2.findContours', wraps=cv2.findContours) as contours, \
                patch('neck_color.cv2.fitLine', wraps=cv2.fitLine) as fit:
            candidates = find_color_neck_regions(strip())
        self.assertTrue(candidates)
        self.assertTrue(contours.called)
        self.assertTrue(fit.called)
        self.assertEqual(candidates[0]['algorithm'], 'color_region_contour_fit')
        with patch('neck_color._region_borders', return_value=np.empty((0, 2, 2), np.float32)), \
                patch('cv2.HoughLinesP', side_effect=AssertionError('Hough fallback')):
            self.assertEqual(find_color_neck_regions(strip()), [])

    def test_crossbars_seed_bounded_candidates_without_asserting_lattice(self):
        for angle in (-90, -25, 0, 25, 90):
            image = cv2.warpAffine(strip(), cv2.getRotationMatrix2D((320, 240), angle, .9),
                                   (640, 480), borderMode=cv2.BORDER_REPLICATE)
            for mirrored in (False, True):
                frame = cv2.flip(image, 1) if mirrored else image
                candidates = find_color_neck_regions(frame, limit=1)
                self.assertEqual(len(candidates), 1, (angle, mirrored))
                self.assertFalse(candidates[0]['lattice_confirmed'])
                self.assertTrue(candidates[0]['crossbars']['supported'])
                for point in candidates[0]['corners']:
                    self.assertTrue(0 <= point['x'] <= 1 and 0 <= point['y'] <= 1)

    def test_saved_neck_edges_cover_fretboard_not_headstock_or_background(self):
        root = Path(__file__).resolve().parents[2]
        positive = cv2.imread(str(root / 'diagnostics/neck_color_probe_20261028/inputs/'
                                  'WIN_20261026_01_17_27_Pro_0.jpg'))
        self.assertIsNotNone(positive)
        candidates = find_color_neck_regions(positive)
        self.assertTrue(candidates)
        polygon = np.float32([[p['x'] * (positive.shape[1] - 1),
                               p['y'] * (positive.shape[0] - 1)]
                              for p in candidates[0]['corners']])
        for point in ((.10, .90), (.45, .72), (.64, .66)):
            self.assertGreaterEqual(cv2.pointPolygonTest(
                polygon, (point[0] * 639, point[1] * 479), False), 0, point)
        self.assertLess(cv2.pointPolygonTest(polygon, (.68 * 639, .43 * 479), False), 0)

    def test_saved_room_negatives_do_not_acquire(self):
        root = Path(__file__).resolve().parents[2]
        for name in ('board_neck_current_20261028_latest.jpg', 'current_neck_frame.jpg'):
            image = cv2.imread(str(root / 'diagnostics' / name))
            self.assertIsNotNone(image)
            self.assertEqual(find_color_neck_regions(image), [], name)
        fixture_root = root / 'diagnostics/neck_color_probe_20261028/inputs'
        for name in ('case_2.jpg', 'case_3.jpg'):
            image = cv2.imread(str(fixture_root / name))
            self.assertIsNotNone(image)
            self.assertEqual(find_color_neck_regions(image), [], name)

    def test_recorded_neck_frames_have_high_direct_acquisition_coverage(self):
        fixture_root = Path(__file__).resolve().parents[2] / 'diagnostics/neck_color_probe_20261028/inputs'
        frames = sorted(fixture_root.glob('WIN_*.jpg'))
        self.assertEqual(len(frames), 56)
        no_guitar = [path for path in frames if path.name.startswith('WIN_20261024_')]
        guitar = [path for path in frames if path not in no_guitar]
        self.assertEqual(len(no_guitar), 8)
        self.assertTrue(all(find_color_neck_regions(cv2.imread(str(path))) == []
                            for path in no_guitar))
        hits = sum(bool(find_color_neck_regions(cv2.imread(str(path)))) for path in guitar)
        self.assertGreaterEqual(hits, 40, f'{hits}/{len(guitar)} direct colour-boundary acquisitions')

    def test_compressed_recording_can_initialize_without_relaxing_spread_gates(self):
        from fretboard_tracker import FretboardTracker
        from vision_pl import reference_words
        fixture = Path(__file__).resolve().parents[2] / 'diagnostics/neck_color_probe_20261028/inputs/WIN_20261026_01_17_27_Pro_0.jpg'
        image = cv2.imread(str(fixture))
        self.assertIsNotNone(image)
        for quality in (70, 90):
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            self.assertTrue(ok)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            candidates = find_color_neck_regions(frame)
            self.assertTrue(candidates)
            polygon = np.float32([[p['x'] * 639, p['y'] * 479]
                                   for p in candidates[0]['corners']])
            self.assertLess(cv2.pointPolygonTest(polygon, (540., 245.), False), 0)
            gray = cv2.resize((reference_words(frame) & 255).astype(np.uint8), (320, 200))
            # Production uses endpoint-weighted sampling inside RoiFretboardTracker.
            result = FretboardTracker(endpoint_weighted=True).initialize(
                gray, candidates[0]['corners'])
            self.assertEqual(result['status'], 'tracking', result)
            self.assertEqual(result['sampling'], 'endpoint_bands', result)


if __name__ == '__main__':
    unittest.main()

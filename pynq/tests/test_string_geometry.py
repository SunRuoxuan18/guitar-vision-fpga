import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from string_geometry import estimate_strings, locate_string


class StringGeometryTests(unittest.TestCase):
    def setUp(self):
        self.quad = np.float32([[.1, .35], [.9, .18], [.86, .59], [.13, .52]])
        self.corners = [dict(x=float(x), y=float(y)) for x, y in self.quad]

    def test_centers_and_boundaries_divide_current_visible_width(self):
        model = estimate_strings(self.corners)
        self.assertEqual(model['status'], 'estimated')
        self.assertFalse(model['string_numbers_known'])
        for i, line in enumerate(model['strings']):
            endpoints = np.float32([[line[k]['x'], line[k]['y']] for k in ('start', 'end')])
            fraction = (i+.5)/6
            np.testing.assert_allclose(endpoints, self.quad[[0, 1]] +
                fraction*(self.quad[[3, 2]]-self.quad[[0, 1]]), atol=1e-6)
            self.assertFalse(line['observed'])
        np.testing.assert_allclose([b['board_y'] for b in model['boundaries']],
                                  np.arange(7)/6)
        self.assertEqual(model['width_source'], 'current_tracked_image_edges')
        self.assertIsNone(model['margin_fraction'])

    def test_each_lane_and_ambiguous_boundary(self):
        model = estimate_strings(self.corners)
        forward = cv2.getPerspectiveTransform(np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), self.quad)
        def point(x, y):
            ends = cv2.perspectiveTransform(np.float32([[[x, 0], [x, 1]]]), forward)[0]
            p = ends[0] + y*(ends[1]-ends[0])
            return dict(x=float(p[0]), y=float(p[1]))
        for x in (.05, .45, .95):
            for i in range(6):
                lane = locate_string(model, point(x, (i+.5)/6))
                self.assertEqual(lane['string_line'], i)
                self.assertAlmostEqual(lane['board_y'], (i+.5)/6, places=5)
                self.assertFalse(lane['ambiguous'])
                self.assertIsNone(lane['pressed'])
        self.assertTrue(locate_string(model, point(.45, 2/6))['ambiguous'])
        self.assertIsNone(locate_string(model, point(.45, 1.1)))

    def test_invalid_corners_do_not_create_lines(self):
        for corners in (None, [], self.corners[:3], [self.corners[0]]*4,
                        [self.corners[i] for i in (0, 2, 1, 3)],
                        [dict(x=float('nan'), y=.5)]*4):
            self.assertIsNone(estimate_strings(corners))

    def test_motion_changes_estimates_without_reusing_old_geometry(self):
        before = estimate_strings(self.corners)
        shifted = [dict(x=p['x']+.02, y=p['y']+.03) for p in self.corners]
        after = estimate_strings(shifted)
        for old, new in zip(before['strings'], after['strings']):
            for end in ('start', 'end'):
                self.assertAlmostEqual(new[end]['x']-old[end]['x'], .02, places=6)
                self.assertAlmostEqual(new[end]['y']-old[end]['y'], .03, places=6)

    def test_changing_width_updates_each_end_independently(self):
        moved = self.quad.copy()
        moved[3] = moved[0] + .65*(moved[3]-moved[0])
        moved[2] = moved[1] + 1.15*(moved[2]-moved[1])
        model = estimate_strings([dict(x=float(x), y=float(y)) for x, y in moved])
        for line in model['strings']:
            y = line['board_y']
            for key, top, bottom in (('start', 0, 3), ('end', 1, 2)):
                expected = moved[top] + y*(moved[bottom]-moved[top])
                np.testing.assert_allclose([line[key]['x'], line[key]['y']], expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()

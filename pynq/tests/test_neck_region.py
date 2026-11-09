import unittest
from pathlib import Path
import sys
from unittest.mock import patch
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neck_region import find_neck_region, find_neck_regions, find_neck_regions_fast
from vision_pc_bridge import FrameTracker


class NeckRegionTests(unittest.TestCase):
    def test_fast_negative_uses_only_one_reduced_resolution_search(self):
        image = np.zeros((480,640), np.uint8)
        with patch('neck_region.cv2.HoughLinesP', return_value=None) as search:
            self.assertEqual(find_neck_regions_fast(image), [])
        self.assertEqual(search.call_count, 1)
        self.assertEqual(search.call_args.args[0].shape, (336,448))

    def test_fast_border_acquisition_does_not_require_fret_wires(self):
        image = np.full((480,640),35,np.uint8)
        cv2.line(image,(45,180),(590,180),230,2)
        cv2.line(image,(45,250),(590,250),230,2)
        # Two crossbars shape the proposal, but are insufficient to confirm
        # a fret lattice. The long neck borders should still seed tracking.
        for x in (80,180):
            cv2.line(image,(x,180),(x,250),180,2)
        self.assertEqual(find_neck_regions_fast(image), [])
        proposals = find_neck_regions_fast(image, require_frets=False)
        self.assertTrue(proposals)
        points = np.array([[p['x']*639,p['y']*479] for p in proposals[0]])
        self.assertGreater(np.linalg.norm(points[1]-points[0]),500)

    def test_visible_long_borders_survive_occluded_fret_extent(self):
        from neck_region import _find_neck_regions
        image = np.zeros((480, 640), np.uint8)
        # Borders are visible through the hand zone; only three transverse
        # wires are visible at the left. Do not replace 40..580 by 60..260.
        lines = np.int32([[40,180,580,180], [40,250,580,250],
                          [60,180,60,250], [150,180,150,250], [260,180,260,250]])[:,None,:]
        with patch('neck_region.cv2.HoughLinesP', return_value=lines), \
             patch('neck_region.native_backend', return_value=None):
            proposals = _find_neck_regions(image,6,image,max_segments=384,require_frets=True)
        self.assertTrue(proposals)
        self.assertGreater(max(p['x']*639 for p in proposals[0]), 575)
        self.assertLess(min(p['x']*639 for p in proposals[0]), 45)

    def test_fast_acquisition_finds_recorded_neck_and_rejects_negative(self):
        root = Path(__file__).resolve().parents[2] / 'diagnostics'
        positive = cv2.imread(str(root/'plgray_neck_fixture.png'),0)
        self.assertTrue(find_neck_regions_fast(positive))
        negative = cv2.imread(str(root/'current_neck_frame.jpg'),0)
        self.assertEqual(find_neck_regions_fast(negative),[])

    def test_fast_acquisition_includes_hand_on_first_recorded_colour_frame(self):
        from hand_region import HandRegionTracker
        from vision_pc_bridge import LatticeSearch
        fixture = Path(__file__).resolve().parents[2]/'diagnostics/user_fretboard_frame.png'
        image = cv2.cvtColor(cv2.resize(cv2.imread(str(fixture)),(640,480)),cv2.COLOR_BGR2RGBA)
        rgb = image[:,:,:3].astype(np.uint32)
        gray = ((77*rgb[:,:,0]+150*rgb[:,:,1]+29*rgb[:,:,2]+128)>>8).astype(np.uint8)
        proposals = find_neck_regions_fast(gray)
        self.assertTrue(proposals)
        quad = LatticeSearch.select_proposal(proposals,gray.shape)
        result = HandRegionTracker().update(image,gray,{},dict(roi_status='tracking',roi_corners=quad))
        self.assertEqual(result['status'],'tracking',result)
        self.assertGreater(np.mean([p['x'] for p in result['regions'][0]['corners']]),.55)

    def test_alternate_fast_pass_recovers_jpeg_neck_without_accepting_grid(self):
        root = Path(__file__).resolve().parents[2]/'diagnostics'
        image = cv2.resize(cv2.imread(str(root/'user_fretboard_frame.png')),(640,480))
        for quality in (95,80,70):
            _, data = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,quality])
            rgb = cv2.imdecode(data,cv2.IMREAD_COLOR)[:,:,::-1].astype(np.uint32)
            gray = ((77*rgb[:,:,0]+150*rgb[:,:,1]+29*rgb[:,:,2]+128)>>8).astype(np.uint8)
            self.assertTrue(find_neck_regions_fast(gray,alternate=True),quality)
        negative = cv2.imread(str(root/'current_neck_frame.jpg'),0)
        self.assertEqual(find_neck_regions_fast(negative,alternate=True),[])
        grid = np.zeros((480,640),np.uint8)
        for x in range(20,640,30):
            cv2.line(grid,(x,20),(x,460),255,2)
        for y in range(20,480,30):
            cv2.line(grid,(20,y),(620,y),255,2)
        self.assertEqual(find_neck_regions_fast(grid,alternate=True),[])

    def test_edge_retry_compares_extent_independent_of_frame_position(self):
        short = [dict(x=.75,y=.4), dict(x=.99,y=.4), dict(x=.99,y=.6), dict(x=.75,y=.6)]
        whole = [dict(x=.02,y=.4), dict(x=.65,y=.4), dict(x=.65,y=.6), dict(x=.02,y=.6)]
        for mirrored in (False, True):
            first = [dict(x=1-p['x'] if mirrored else p['x'], y=p['y']) for p in short]
            second = [dict(x=1-p['x'] if mirrored else p['x'], y=p['y']) for p in whole]
            with patch('neck_region._find_neck_regions', side_effect=[[first],[second]]):
                self.assertEqual(find_neck_regions(np.zeros((480,640), np.uint8)), [second])

    def test_pl_gray_fixture_keeps_visible_hand_side_neck_extent(self):
        fixture = Path(__file__).resolve().parents[2] / 'diagnostics' / 'plgray_neck_fixture.png'
        image = cv2.imread(str(fixture), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)
        corners = find_neck_region(image)
        self.assertIsNotNone(corners)
        points = np.array([[point['x'], point['y']] for point in corners])
        # The right-side neck is observed past the fretting hand. Do not crop
        # this to the left-side Hough fragments or the embedded preview.
        self.assertGreater(points[:, 0].max(), .70, points)
        self.assertLess(points[:, 0].min(), .15, points)
        self.assertLess(points[:, 1].min(), .42, points)
        self.assertGreater(points[:, 1].max(), .78, points)

    def test_candidates_are_bounded_distinct_and_keep_legacy_first_proposal(self):
        image = np.full((480,640),35,np.uint8)
        for y in (190,260):
            cv2.line(image,(60,y),(340,y),230,2)
        for x in (65,85,107,132,160,191,226,265,309,358,413,475):
            cv2.line(image,(x,185),(x,265),180,2)
        candidates = find_neck_regions(image)
        self.assertGreater(len(candidates),1)
        self.assertLessEqual(len(candidates),6)
        self.assertEqual(candidates[0],find_neck_region(image))
        for i,quad in enumerate(candidates):
            points=np.array([[p['x']*639,p['y']*479] for p in quad])
            for old in candidates[:i]:
                prior=np.array([[p['x']*639,p['y']*479] for p in old])
                self.assertGreater(np.linalg.norm(points-prior,axis=1).mean(),7)
        for limit in (0,7,True):
            with self.assertRaises(ValueError):
                find_neck_regions(image,limit)

    def test_disjoint_border_segments_across_hand_occlusion(self):
        image = np.full((480,640),35,np.uint8)
        cv2.line(image,(40,180),(240,180),230,2)
        cv2.line(image,(300,250),(590,250),230,2)
        for x in (65,116,170,227,287,351,419,491,568):
            cv2.line(image,(x,180),(x,250),180,2)
        corners = find_neck_region(image)
        self.assertIsNotNone(corners)
        self.assertLess(min(p['x']*639 for p in corners),80)
        self.assertGreater(max(p['x']*639 for p in corners),550)

    def test_border_only_candidate_survives_moderate_hand_occlusion(self):
        image = np.full((480,640),35,np.uint8)
        for y in (180,250):
            cv2.line(image,(45,y),(590,y),230,2)
        # A hand hides the middle fret run, leaving only the validated,
        # elongated board borders for the cheap acquisition proposal.
        cv2.rectangle(image,(230,155),(410,275),35,-1)
        corners = find_neck_region(image)
        self.assertIsNotNone(corners)
        points = np.array([[p['x']*639,p['y']*479] for p in corners])
        self.assertLess(points[:,0].min(),65)
        self.assertGreater(points[:,0].max(),570)
        self.assertGreater(np.linalg.norm(points[1]-points[0]),400)

    def test_parallel_background_lines_without_board_extent_do_not_propose(self):
        image = np.zeros((480,640),np.uint8)
        # Long parallels without a closed, in-frame board extent are not a
        # neck candidate even though they are elongated.
        cv2.line(image,(-100,180),(260,180),230,2)
        cv2.line(image,(380,250),(740,250),230,2)
        self.assertIsNone(find_neck_region(image))

    def test_transverse_frets_bound_neck_when_long_lines_are_inner_strings(self):
        image = np.full((480,640),35,np.uint8)
        for y in (220,240):
            cv2.line(image,(50,y),(590,y),230,2)
        for x in (65,120,172,221,267,310,350,388,424,458,490,520,548,574):
            cv2.line(image,(x,180),(x,260),180,2)
        corners = find_neck_region(image)
        self.assertIsNotNone(corners)
        quad = np.float32([[p['x']*639,p['y']*479] for p in corners])
        # All six independently specified string rows must be inside the ROI,
        # even though only two inner strings supplied long Hough segments.
        for y in np.linspace(188,252,6):
            self.assertGreaterEqual(cv2.pointPolygonTest(quad,(320.,float(y)),False),0)

    def test_repeated_frets_extend_beyond_fragmented_long_borders(self):
        image = np.full((480,640),35,np.uint8)
        for y in (190,260):
            cv2.line(image,(60,y),(340,y),230,2)
        for x in (65,85,107,132,160,191,226,265,309,358,413,475):
            cv2.line(image,(x,185),(x,265),180,2)
        corners = find_neck_region(image)
        self.assertIsNotNone(corners)
        self.assertGreater(max(p['x']*639 for p in corners),460)

    def test_five_visible_frets_use_short_scale_progression(self):
        image = np.zeros((480,640),np.uint8)
        for y in (180,250):
            cv2.line(image,(40,y),(580,y),210,2)
        for x in (65,116,170,227,287):
            cv2.line(image,(x,180),(x,250),220,2)
        self.assertIsNotNone(find_neck_region(image))

    def test_blank_and_regular_grid_do_not_initialize_neck(self):
        for grid in (False, True):
            image = np.zeros((480, 640), np.uint8)
            if grid:
                for x in range(20, 640, 30):
                    cv2.line(image, (x, 20), (x, 460), 255, 2)
                for y in range(20, 480, 30):
                    cv2.line(image, (20, y), (620, y), 255, 2)
            self.assertIsNone(find_neck_region(image))
            self.assertEqual(find_neck_regions_fast(image),[])

    def test_neck_region_does_not_assert_six_strings(self):
        image = np.zeros((480, 640), np.uint8)
        cv2.line(image, (50, 180), (590, 180), 210, 2)
        cv2.line(image, (50, 250), (590, 250), 210, 2)
        for x in (65, 120, 172, 221, 267, 310, 350, 388, 424, 458, 490, 520, 548, 574):
            cv2.line(image, (x, 180), (x, 250), 220, 2)
        self.assertIsNotNone(find_neck_region(image))
        result = FrameTracker(fast_tracking=True).track(image,
            dict(stream='test', generation=1, sequence=1, enabled=True))
        self.assertNotEqual(result['status'], 'tracking')
        self.assertEqual(result['strings'], [])

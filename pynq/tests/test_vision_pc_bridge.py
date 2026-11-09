"""Wire contract and real-driver boundary; mock output is never deployment evidence."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import threading
import zlib
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_pc_bridge import Bridge, FrameTracker, LatticeSearch, make_server, decode_frame, StreamBusy, analysis_header
from vision_pl import VisionPLError
from test_auto_fretboard import GuitarScene
from auto_fretboard import AutoFretboardTracker
from vision_pl import reference_words
from work_priority import WorkPriority


class BridgeTests(unittest.TestCase):
    def test_estimated_strings_follow_live_roi_without_claiming_a_lattice(self):
        tracker = FrameTracker(fast_tracking=True)
        tracker.identity = ('string-model', 1)
        tracker.last_gray_at = None
        tracker.seed_source = 'manual'
        tracker.tracker = Mock()
        corners = [dict(x=.1,y=.3),dict(x=.9,y=.3),dict(x=.9,y=.6),dict(x=.1,y=.6)]
        tracker.tracker.update.return_value = dict(status='searching', roi_status='tracking',
            roi_corners=corners, strings=[], frets=[], reason='no_supported_lattice')
        tracker.tracker.timings = {}
        options = dict(stream='string-model',generation=1,sequence=1,enabled=True,corners=corners)
        result = tracker.track(np.zeros((480,640),np.uint8), options)
        self.assertEqual(result['status'], 'searching')
        self.assertEqual(result['strings'], [])
        self.assertEqual(len(result['estimated_strings']['strings']), 6)
        self.assertEqual(result['estimated_strings']['sequence'], 1)
        tracker.tracker.update.return_value = dict(status='searching', roi_status='lost',
            roi_corners=None, strings=[], frets=[], reason='roi_tracking_lost')
        result = tracker.track(np.zeros((480,640),np.uint8), dict(options,sequence=2))
        self.assertIsNone(result['estimated_strings'])

    def test_large_optional_preview_cannot_break_coordinate_headers(self):
        import json
        preview = dict(jpeg_base64='a' * 18000, width=208, height=208)
        analysis = dict(sequence=3, stream_id='camera', generation=1,
            roi_status='tracking', fingertip_neural=dict(status='stale',
                last_result=dict(preview=preview, points=dict(index=dict(x=100., y=200.)))),
            fingertip_tracking=dict(status='tracking', points=dict(index=dict(x=102., y=200.))))
        encoded = analysis_header(analysis)
        self.assertLess(len(encoded), 14 * 1024)
        compact = json.loads(encoded)
        self.assertNotIn('preview', compact['fingertip_neural']['last_result'])
        self.assertEqual(compact['fingertip_tracking'], analysis['fingertip_tracking'])
        self.assertEqual(compact['fingertip_neural']['last_result']['points'],
                         analysis['fingertip_neural']['last_result']['points'])
        self.assertIs(analysis['fingertip_neural']['last_result']['preview'], preview)

    def test_small_preview_survives_coordinate_header(self):
        import json
        analysis = dict(fingertip_neural=dict(last_result=dict(preview=dict(jpeg_base64='abc'))))
        self.assertEqual(json.loads(analysis_header(analysis)), analysis)

    def test_note_calibration_is_validated_at_request_boundary(self):
        options = dict(stream='camera', generation=1, sequence=1, enabled=True)
        options['note_calibration'] = dict(stream_id='camera', generation=1, lattice_epoch=1,
            first_boundary_fret=0, strings_reversed=False, frets_reversed=False,
            tuning=[40,45,50,55,59,64])
        Bridge.validate_options(options)
        options['note_calibration']['tuning'] = [40,45]
        with self.assertRaises(ValueError):
            Bridge.validate_options(options)

    def test_fingertip_roi_is_bounded_and_source_identity_is_forwarded(self):
        bridge = Bridge(Mock())
        bridge.fingertip_worker = Mock()
        bridge.fingertip_worker.snapshot.return_value = {'status': 'running'}
        bridge.analysis = {'hand_region': {'status': 'tracking', 'regions': [
            {'corners': [dict(x=.5,y=.25),dict(x=.75,y=.25),
                         dict(x=.75,y=.75),dict(x=.5,y=.75)]}]}}
        frame = np.zeros((480,640,4),np.uint8)
        options = dict(stream='camera',generation=3,sequence=19)
        bridge.update_fingertips(frame,options)
        args = bridge.fingertip_worker.submit.call_args.args
        self.assertIs(args[0],frame)
        self.assertEqual(args[1:],((320,120,160,240),'camera',3,19))
        self.assertEqual(bridge.analysis['fingertip_neural'],{'status':'running'})

    def test_fingertip_does_not_infer_on_missing_transport_pixels(self):
        bridge = Bridge(Mock())
        bridge.fingertip_worker = Mock()
        bridge.analysis = {'hand_region': {'status': 'tracking', 'regions': [
            {'corners': [dict(x=.5,y=.25),dict(x=.75,y=.25),
                         dict(x=.75,y=.75),dict(x=.5,y=.75)]}]}}
        bridge.update_fingertips(np.zeros((480,640,4),np.uint8),
            dict(stream='camera',generation=3,sequence=20,
                 transport_roi=dict(x=0,y=0,width=100,height=100)))
        bridge.fingertip_worker.submit.assert_not_called()
        bridge.fingertip_worker.snapshot.assert_called_once_with('camera',3,20)

    def test_production_hand_tracker_targets_image_right(self):
        bridge = Bridge(Mock())
        self.assertEqual(bridge.hand_region_tracker.target_side,'right')

    def test_geometry_refresh_never_starts_neural_or_lattice_search(self):
        detector = Mock()
        worker = LatticeSearch(detector)
        with patch('vision_pc_bridge.find_neck_regions_fast', return_value=[]), \
             patch('vision_pc_bridge.find_neck_regions', side_effect=AssertionError('slow search')), \
             patch('vision_pc_bridge.AutoFretboardTracker', side_effect=AssertionError('lattice search')), \
             patch('vision_pc_bridge.threading.Thread') as thread:
            worker.submit(np.zeros((480,640),np.uint8),None,('camera',1),geometry_only=True)
            thread.call_args.kwargs['target']()
        self.assertFalse(worker.model_busy)
        detector.detect.assert_not_called()

    def test_initial_search_accepts_long_borders_before_lattice_confirmation(self):
        worker = LatticeSearch()
        gray = np.zeros((480,640),np.uint8)
        proposals = [[dict(x=.05,y=.4),dict(x=.95,y=.4),
                      dict(x=.95,y=.6),dict(x=.05,y=.6)]]
        with patch('vision_pc_bridge.find_neck_regions_fast',return_value=proposals) as fast, \
             patch('vision_pc_bridge.threading.Thread') as thread:
            worker.submit(gray,None,('camera',1))
            thread.call_args.kwargs['target']()
        self.assertFalse(fast.call_args.kwargs['require_frets'])
        result = worker.take(('camera',1))[2]
        self.assertEqual(result['acquisition'],'neck_border_candidate')
        self.assertFalse(result['lattice_confirmed'])

    def test_initial_acquisition_is_fast_without_a_model_and_retries_promptly(self):
        worker = LatticeSearch()
        with patch('vision_pc_bridge.find_neck_regions_fast', return_value=[]) as fast, \
             patch('vision_pc_bridge.find_neck_regions', side_effect=AssertionError('slow search')), \
             patch('vision_pc_bridge.AutoFretboardTracker', side_effect=AssertionError('lattice search')), \
             patch('vision_pc_bridge.time.monotonic', return_value=10.), \
             patch('vision_pc_bridge.threading.Thread') as thread:
            worker.submit(np.zeros((480,640),np.uint8),None,('camera',1))
            thread.call_args.kwargs['target']()
        fast.assert_called_once()
        self.assertEqual(worker.take(('camera',1))[2]['reason'], 'no_fast_neck_proposal')
        self.assertLessEqual(worker.next_at, 10.2)

    def test_initial_search_uses_yolo_without_running_global_geometry(self):
        detector = Mock()
        detector.status.return_value = {'status': 'ready'}
        worker = LatticeSearch(detector)
        with patch('vision_pc_bridge.find_neck_regions_fast', side_effect=AssertionError('global geometry')), \
             patch('vision_pc_bridge.AutoFretboardTracker', side_effect=AssertionError('lattice search')), \
             patch('vision_pc_bridge.threading.Thread') as thread:
            worker.submit(np.zeros((480,640),np.uint8),None,('camera',1),
                          bgr=np.zeros((480,640,3),np.uint8))
            thread.call_args.kwargs['target']()
        self.assertEqual(thread.call_count, 2)
        self.assertTrue(worker.status()['model_primary_enabled'])
        self.assertFalse(worker.status()['model_fallback_enabled'])
        self.assertEqual(worker.status()['acquisition_mode'], 'yolo_first_then_flow')
        self.assertEqual(worker.take(('camera',1))[2]['reason'], 'yolo_detection_pending')
        detector.detect.assert_not_called()
        self.assertTrue(worker.status()['model_busy'])

    def test_color_mode_acquires_without_waiting_for_yolo_or_full_lattice(self):
        detector = Mock()
        detector.status.return_value = {'status': 'ready'}
        corners = [dict(x=.05,y=.4),dict(x=.95,y=.4),dict(x=.95,y=.6),dict(x=.05,y=.6)]
        worker = LatticeSearch(detector, acquisition_mode='color_border')
        with patch('neck_color.find_color_neck_regions', return_value=[dict(corners=corners)]) as color, \
             patch('vision_pc_bridge.find_neck_regions_fast', side_effect=AssertionError('legacy search')), \
             patch('vision_pc_bridge.AutoFretboardTracker', side_effect=AssertionError('full lattice')), \
             patch('vision_pc_bridge.threading.Thread') as thread:
            worker.submit(np.zeros((480,640),np.uint8), None, ('camera',1),
                          bgr=np.zeros((480,640,3),np.uint8))
            thread.call_args.kwargs['target']()
        color.assert_called_once()
        detector.detect.assert_not_called()
        self.assertEqual(thread.call_count, 1)
        result = worker.take(('camera',1))[2]
        self.assertEqual(result['corners'], corners)
        self.assertFalse(result['lattice_confirmed'])
        self.assertEqual(FrameTracker.seed_source_for(result), 'color_border_candidate')
        self.assertEqual(worker.status()['acquisition_mode'], 'color_border_then_flow')
        self.assertIn('color_region_contour_fit', worker.status()['fast_acquisition']['backend'])
        self.assertFalse(worker.status()['model_primary_enabled'])

    def test_color_mode_negative_retries_without_publishing_geometry_or_starting_yolo(self):
        detector = Mock()
        worker = LatticeSearch(detector, acquisition_mode='color_border')
        with patch('neck_color.find_color_neck_regions', return_value=[]), \
             patch('vision_pc_bridge.time.monotonic', return_value=10.), \
             patch('vision_pc_bridge.threading.Thread') as thread:
            worker.submit(np.zeros((480,640),np.uint8), None, ('camera',1),
                          bgr=np.zeros((480,640,3),np.uint8))
            thread.call_args.kwargs['target']()
        result = worker.take(('camera',1))[2]
        self.assertEqual(result['status'], 'searching')
        self.assertNotIn('corners', result)
        self.assertEqual(result['strings'], [])
        self.assertEqual(result['frets'], [])
        self.assertLessEqual(worker.next_at, 10.2)
        detector.detect.assert_not_called()

    def test_acquisition_tries_bounded_alternative_after_invalid_primary(self):
        gray = np.zeros((200, 320), np.uint8)
        first = [dict(x=.1, y=.4), dict(x=.9, y=.4), dict(x=.9, y=.42), dict(x=.1, y=.42)]
        second = [dict(x=.1, y=.4), dict(x=.9, y=.4), dict(x=.9, y=.6), dict(x=.1, y=.6)]
        tracker = FrameTracker()
        with patch('vision_pc_bridge.RoiFretboardTracker') as constructor:
            proposed = constructor.return_value
            proposed.initialize.side_effect = [dict(roi_status='lost', reason='invalid_corners'),
                                               dict(status='tracking', roi_status='tracking', reason=None)]
            _, result = tracker.initialize_search_result(gray, gray,
                dict(corners=first, alternatives=[second], acquisition='color_border_candidate'))
        self.assertEqual(result['roi_status'], 'tracking')
        self.assertEqual(proposed.initialize.call_count, 2)
        self.assertEqual(proposed.initialize.call_args.args[1], second)
        self.assertFalse(constructor.call_args.kwargs['async_measurements'])

    def test_fast_acquisition_keeps_lattice_measurement_async(self):
        gray = np.zeros((200, 320), np.uint8)
        first = [dict(x=.1, y=.4), dict(x=.9, y=.4), dict(x=.9, y=.42), dict(x=.1, y=.42)]
        tracker = FrameTracker(fast_tracking=True)
        with patch('vision_pc_bridge.RoiFretboardTracker') as constructor:
            proposed = constructor.return_value
            proposed.initialize.return_value = dict(
                status='searching', roi_status='tracking',
                reason='no_supported_lattice')
            _, result = tracker.initialize_search_result(gray, gray,
                dict(corners=first, acquisition='color_border_candidate'))
        self.assertEqual(result['roi_status'], 'tracking')
        self.assertEqual(proposed.initialize.call_count, 1)
        self.assertTrue(constructor.call_args.kwargs['async_measurements'])
        self.assertTrue(proposed.async_measurements)

    def test_color_mode_requires_color_pixels_and_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            LatticeSearch(acquisition_mode='unknown')
        worker = LatticeSearch(acquisition_mode='color_border')
        with patch('vision_pc_bridge.threading.Thread') as thread:
            worker.submit(np.zeros((480,640),np.uint8), None, ('camera',1))
            thread.call_args.kwargs['target']()
        self.assertEqual(worker.take(('camera',1))[2]['reason'], 'color_frame_required')

    def test_fast_negative_alternates_single_pass_and_new_stream_resets_it(self):
        worker = LatticeSearch()
        gray = np.zeros((480,640),np.uint8)
        with patch('vision_pc_bridge.find_neck_regions_fast',return_value=[]) as fast, \
             patch('vision_pc_bridge.threading.Thread') as thread:
            for identity in (('camera',1),('camera',1),('camera',1),('new',1)):
                worker.next_at = float('-inf')
                self.assertTrue(worker.submit(gray,None,identity))
                thread.call_args.kwargs['target']()
            self.assertEqual(fast.call_count,4)
            self.assertEqual(fast.call_args_list[0].kwargs,
                             dict(alternate=False,require_frets=False))
            self.assertEqual(fast.call_args_list[1].kwargs,
                             dict(alternate=True,require_frets=False))
            self.assertEqual(fast.call_args_list[2].kwargs,
                             dict(alternate=False,require_frets=False))
            self.assertEqual(fast.call_args_list[3].kwargs,
                             dict(alternate=False,require_frets=False))

    def test_short_live_lock_extends_only_to_overlapping_observed_neck(self):
        old = [dict(x=.2,y=.4),dict(x=.55,y=.4),dict(x=.55,y=.6),dict(x=.2,y=.6)]
        whole = [dict(x=.1,y=.4),dict(x=.9,y=.4),dict(x=.9,y=.6),dict(x=.1,y=.6)]
        elsewhere = [dict(x=.1,y=.1),dict(x=.9,y=.1),dict(x=.9,y=.3),dict(x=.1,y=.3)]
        gray = np.zeros((480,640),np.uint8)
        for candidate, accepted in ((whole,True),(elsewhere,False),(old,False)):
            worker = Mock()
            worker.take.return_value = (('camera',1),gray,
                dict(status='tracking',corners=candidate,acquisition='neck_border_candidate'))
            tracker = FrameTracker(search_worker=worker)
            original = tracker.tracker = Mock()
            result = dict(status='searching',roi_status='tracking',roi_corners=old)
            replacement = dict(result,roi_corners=candidate)
            proposed = Mock()
            with patch.object(tracker,'align_candidate',return_value=candidate), \
                 patch.object(tracker,'initialize_search_result',return_value=(proposed,replacement)) as initialize:
                actual = tracker.refresh_neck_extent(gray,None,result,2.)
            if accepted:
                self.assertIs(tracker.tracker,proposed)
                self.assertEqual(actual['coverage'],'observed_neck_extended')
            else:
                self.assertIs(tracker.tracker,original)
                initialize.assert_not_called()
            tracker.refresh_neck_extent(gray,None,result,2.1)
            self.assertEqual(worker.submit.call_count,1)

    def test_stable_extent_search_backs_off_but_retries_deferred_submission(self):
        gray = np.zeros((480,640),np.uint8)
        worker = Mock()
        tracker = FrameTracker(search_worker=worker)
        result = dict(roi_status='tracking',roi_corners=[])
        worker.take.return_value = (('camera',1),gray,dict(status='searching'))
        worker.submit.return_value = True
        tracker.refresh_neck_extent(gray,None,result,0.)
        self.assertEqual(tracker.extent_interval,4.)
        tracker.refresh_neck_extent(gray,None,result,1.)
        self.assertEqual(tracker.extent_interval,8.)
        worker.take.return_value = None
        tracker.refresh_neck_extent(gray,None,result,7.9)
        self.assertEqual(worker.submit.call_count,1)
        worker.submit.return_value = False
        tracker.refresh_neck_extent(gray,None,result,8.)
        self.assertEqual(tracker.last_extent_at,0.)
        worker.submit.return_value = True
        tracker.refresh_neck_extent(gray,None,result,8.1)
        self.assertEqual(tracker.last_extent_at,8.1)

    def test_extent_backoff_never_delays_lost_roi_acquisition(self):
        tracker = FrameTracker(search_worker=Mock())
        tracker.extent_interval = 8.
        gray = np.zeros((480,640),np.uint8)
        result = dict(roi_status='lost')
        self.assertIs(tracker.refresh_neck_extent(gray,None,result,1.),result)
        tracker.search_worker.take.assert_not_called()
        with patch.object(tracker,'full_search',return_value=(gray,dict(status='searching'))) as search:
            tracker.discover(gray,None)
        search.assert_called_once()

    def test_failed_recent_relocalization_bounds_expensive_retries(self):
        source = GuitarScene().render()[0]
        tracker = FrameTracker(fast_tracking=True,search_worker=Mock())
        tracker.neck_reference = (10.,source,[], 'neck_border_candidate')
        with patch.object(tracker,'align_candidate',return_value=None) as align:
            for now in (11.,11.05,11.24,11.25):
                with patch('vision_pc_bridge.time.monotonic',return_value=now):
                    self.assertIsNone(tracker.relocalize_recent(source))
            self.assertEqual(align.call_count,2)

    def test_recent_neck_relocalization_requires_current_image_and_expires(self):
        scene = GuitarScene()
        source = scene.render()[0]
        corners = [dict(x=float(x/(scene.width-1)),y=float(y/(scene.height-1))) for x,y in scene.quad]
        tracker = FrameTracker(fast_tracking=True,search_worker=Mock())
        tracker.neck_reference = (10.,source,corners,'neck_border_candidate')
        with patch('vision_pc_bridge.time.monotonic',return_value=11.):
            self.assertIsNone(tracker.relocalize_recent(np.zeros_like(source)))
            with patch.object(tracker,'initialize_search_result',return_value=(Mock(),
                     dict(status='searching',roi_status='tracking',roi_corners=corners))):
                result = tracker.relocalize_recent(scene.render(dx=8)[0])
            self.assertEqual(result['recovery'],'verified_recent_neck')
            self.assertEqual(tracker.seed_source,'neck_border_candidate')
        with patch('vision_pc_bridge.time.monotonic',return_value=14.):
            self.assertIsNone(tracker.relocalize_recent(source))

    def test_recent_recovery_prefers_verified_rotation_over_translation(self):
        gray = GuitarScene().render()[0]
        corners = [dict(x=.1,y=.4),dict(x=.9,y=.4),dict(x=.9,y=.6),dict(x=.1,y=.6)]
        rotated = [dict(x=.12,y=.45),dict(x=.88,y=.35),dict(x=.9,y=.55),dict(x=.14,y=.65)]
        with patch.object(FrameTracker,'align_features',return_value=rotated) as align, \
             patch('vision_pc_bridge.cv2.matchTemplate',side_effect=AssertionError('translation used')):
            actual = FrameTracker.align_candidate(gray,gray,corners,prefer_features=True)
        self.assertEqual(actual,rotated)
        align.assert_called_once_with(gray,gray,corners)
        with patch.object(FrameTracker,'align_features',return_value=None) as align:
            actual = FrameTracker.align_candidate(gray,gray,corners,prefer_features=True)
        self.assertIsNotNone(actual)
        self.assertEqual(align.call_count,1)

    def test_lost_flow_can_recover_in_current_frame_only_after_revalidation(self):
        gray = GuitarScene().render()[0]
        options = dict(stream='camera',generation=1,sequence=3,enabled=True)
        corners = [dict(x=.1,y=.4),dict(x=.9,y=.4),dict(x=.9,y=.6),dict(x=.1,y=.6)]
        for accepted in (False,True):
            tracker = FrameTracker(fast_tracking=True,search_worker=Mock())
            tracker.identity = ('camera',1)
            tracker.tracker = Mock()
            tracker.tracker.update.return_value = dict(status='searching',roi_status='lost',
                reason='roi_tracking_lost',flow_reason='insufficient_spread',strings=[],frets=[])
            restored = dict(status='searching',roi_status='tracking',roi_corners=corners,
                reason='no_supported_lattice',strings=[],frets=[],recovery='verified_recent_neck')
            with patch.object(tracker,'relocalize_recent',return_value=restored if accepted else None) as recover, \
                 patch.object(tracker,'refresh_neck_extent',side_effect=lambda g,e,r,n:r), \
                 patch.object(tracker,'recover_lattice',side_effect=lambda g,e,r,n:r):
                result = tracker.track(gray,options)
            recover.assert_called_once_with(gray, force=True)
            self.assertEqual(result['roi_status'],'tracking' if accepted else 'lost')
            if accepted:
                self.assertEqual(result['recovery'],'verified_recent_neck')
            else:
                self.assertIsNone(tracker.tracker)

    def test_large_motion_fast_tracking_recovers_without_full_image_search(self):
        scene = GuitarScene()
        corners = [dict(x=float(x/(scene.width-1)),y=float(y/(scene.height-1)))
                   for x,y in scene.quad]
        tracker = FrameTracker(fast_tracking=True,search_worker=Mock())
        options = dict(stream='camera',generation=1,sequence=1,enabled=True,
                       fast_tracking=True,corners=corners)
        frame = scene.render()[0]
        initial = tracker.track(frame,options,frame)
        self.assertEqual(initial['roi_status'],'tracking')
        options = dict(options,sequence=2,corners=None)
        current = scene.render(dx=45)[0]
        result = tracker.track(current,options,current)
        self.assertEqual(result['roi_status'],'tracking',result)
        if result.get('recovery') is not None:
            self.assertEqual(result['recovery'],'verified_recent_neck',result)
            self.assertEqual(result['recovery_detail'],'masked_motion_template',result)
        tracker.search_worker.submit.assert_not_called()

    def test_recovery_cannot_occupy_the_neck_acquisition_worker(self):
        search, recovery = Mock(), Mock()
        search.take.return_value = recovery.take.return_value = None
        tracker = FrameTracker(search_worker=search,recovery_worker=recovery)
        gray = np.zeros((20,30),np.uint8)
        tracker.full_search(gray,None,lattice_only=True)
        recovery.submit.assert_called_once()
        search.submit.assert_not_called()
        tracker.full_search(gray,None)
        search.submit.assert_called_once()

    def test_discovery_consumes_completed_result_on_the_next_frame(self):
        worker = Mock()
        worker.take.return_value = None
        tracker = FrameTracker(search_worker=worker)
        tracker.last_search_at = __import__('time').monotonic()
        result = tracker.discover(np.zeros((20,30),np.uint8),None)
        worker.take.assert_called_once()
        self.assertEqual(result['status'],'searching')

    def test_yolo_acquisition_does_not_publish_global_geometry_candidate(self):
        detector = Mock()
        worker = LatticeSearch(detector)
        corners = [dict(x=.1,y=.4),dict(x=.9,y=.4),dict(x=.9,y=.6),dict(x=.1,y=.6)]
        with patch('vision_pc_bridge.find_neck_regions_fast',side_effect=AssertionError('global geometry')), \
             patch('vision_pc_bridge.threading.Thread') as thread:
            worker.submit(np.zeros((480,640),np.uint8),None,('camera',1),
                          bgr=np.zeros((480,640,3),np.uint8))
            thread.call_args.kwargs['target']()
        detected = worker.take(('camera',1))[2]
        self.assertEqual(detected['reason'],'yolo_detection_pending')
        self.assertTrue(worker.model_busy)
        detector.detect.assert_not_called()

    def test_delayed_neural_result_cannot_replace_new_camera_stream(self):
        detector = Mock()
        detector.detect.return_value = dict(status='tracking',corners=[])
        worker = LatticeSearch(detector)
        worker.identity = ('old',1)
        with patch('vision_pc_bridge.threading.Thread') as thread:
            worker.submit_model(np.zeros((20,30),np.uint8),np.zeros((20,30,3),np.uint8),('old',1),[])
            worker.identity = ('new',1)
            thread.call_args.kwargs['target']()
        self.assertIsNone(worker.completed)
        self.assertFalse(worker.model_busy)

    def test_seed_source_is_preserved_and_cleared_for_identity_disable_and_loss(self):
        tracker = FrameTracker(search_worker=Mock())
        tracker.identity = ("camera", 1)
        tracker.tracker = Mock()
        tracker.seed_source = "neck_model_geometry_refined"
        options = dict(stream="camera", generation=1, sequence=1, enabled=False,
                       preview_encoding="jpeg")
        result = tracker.track(np.zeros((20, 20), np.uint8), options)
        self.assertIsNone(tracker.seed_source)
        self.assertEqual(result["status"], "disabled")

        tracker.seed_source = "neck_border_candidate"
        tracker.identity = ("camera", 1)
        tracker.tracker = Mock()
        tracker.tracker.update.return_value = {"status": "searching", "roi_status": "lost"}
        options.update(sequence=2, enabled=True)
        tracker.tracker.invalidate.return_value = {"status": "searching", "roi_status": "lost"}
        with patch("vision_pc_bridge.time.monotonic", side_effect=[10., 13., 13.]):
            result = tracker.track(np.zeros((20, 20), np.uint8), options)
        self.assertIsNone(tracker.seed_source)

        tracker.seed_source = "neck_model_geometry_refined"
        options.update(stream="new-camera", generation=2, sequence=1, enabled=False)
        tracker.track(np.zeros((20, 20), np.uint8), options)
        self.assertIsNone(tracker.seed_source)

    def test_seed_source_from_successful_initialize_is_reported(self):
        tracker = FrameTracker(search_worker=Mock())
        tracker.identity = ("camera", 1)
        tracker.initialize_search_result = Mock(return_value=(Mock(),
            {"status": "tracking", "roi_status": "tracking", "strings": [], "frets": [],
             "roi_corners": []}))
        tracker.full_search = Mock(return_value=(np.zeros((20, 20), np.uint8),
            {"status": "tracking", "acquisition": "neck_model_geometry_refined",
             "corners": [{"x": 0, "y": 0}] * 4}))
        result = tracker.discover(np.zeros((20, 20), np.uint8), None)
        self.assertEqual(tracker.seed_source, "neck_model_geometry_refined")
    def test_neck_acquisition_keeps_long_extent_without_background_envelope(self):
        wide = [dict(x=.1,y=.2),dict(x=.9,y=.2),dict(x=.9,y=.8),dict(x=.1,y=.8)]
        tight = [dict(x=.1,y=.4),dict(x=.89,y=.4),dict(x=.89,y=.6),dict(x=.1,y=.6)]
        short = [dict(x=.1,y=.45),dict(x=.5,y=.45),dict(x=.5,y=.55),dict(x=.1,y=.55)]
        self.assertEqual(LatticeSearch.select_proposal([wide,tight,short],(480,640)),tight)

    def test_delayed_acquisition_aligns_translation_but_rejects_scene_cut(self):
        scene = GuitarScene()
        gray = scene.render()[0]
        current = scene.render(dx=12)[0]
        corners = [dict(x=float(x/(scene.width-1)),y=float(y/(scene.height-1)))
                   for x,y in scene.quad]
        aligned = FrameTracker.align_candidate(gray,current,corners)
        self.assertIsNotNone(aligned)
        shift = np.mean([p['x']-q['x'] for p,q in zip(aligned,corners)])*(scene.width-1)
        self.assertAlmostEqual(shift,12,delta=3)
        self.assertIsNone(FrameTracker.align_candidate(gray,np.zeros_like(gray),corners))

    def test_delayed_neck_uses_spatially_supported_orb_for_rotation(self):
        rng = np.random.RandomState(48)
        # Keep texture resolved at the tracker's analysis scale on ARM and x86.
        source = cv2.resize(rng.randint(0,256,(120,160),dtype=np.uint8),
                            (640,480),interpolation=cv2.INTER_CUBIC)
        transform = cv2.getRotationMatrix2D((320,240),5.,1.)
        current = cv2.warpAffine(source,transform,(640,480))
        corners = [dict(x=.15,y=.3),dict(x=.85,y=.3),dict(x=.85,y=.7),dict(x=.15,y=.7)]
        aligned = FrameTracker.align_features(source,current,corners)
        self.assertIsNotNone(aligned)
        pixels = np.float32([[p['x']*639,p['y']*479] for p in corners])
        expected = cv2.transform(pixels.reshape(-1,1,2),transform)[:,0]
        actual = np.float32([[p['x']*639,p['y']*479] for p in aligned])
        self.assertLess(float(np.linalg.norm(actual-expected,axis=1).max()),6.)
        self.assertIsNone(FrameTracker.align_features(source,np.zeros_like(source),corners))

    def test_masked_motion_relocalization_handles_large_one_frame_translation(self):
        scene = GuitarScene()
        source = scene.render()[0]
        current = scene.render(dx=45)[0]
        corners = [dict(x=float(x/(scene.width-1)),y=float(y/(scene.height-1)))
                   for x,y in scene.quad]
        aligned = FrameTracker.align_motion(source,current,corners)
        self.assertIsNotNone(aligned)
        shift = np.mean([p['x']-q['x'] for p,q in zip(aligned,corners)])*(scene.width-1)
        self.assertAlmostEqual(shift,45,delta=3)
        self.assertIsNone(FrameTracker.align_motion(source,np.zeros_like(source),corners))

    def test_masked_motion_relocalization_rejects_unrelated_board_texture(self):
        scene = GuitarScene()
        source = scene.render()[0]
        other = GuitarScene()
        other.wood = np.random.default_rng(8).integers(
            20, 230, other.wood.shape, dtype=np.uint8)
        corners = [dict(x=float(x/(scene.width-1)),y=float(y/(scene.height-1)))
                   for x,y in scene.quad]
        self.assertIsNone(FrameTracker.align_motion(source,other.render()[0],corners))

    def test_delayed_border_proposal_initializes_on_current_verified_frame(self):
        scene = GuitarScene()
        source, current = scene.render()[0], scene.render(dx=8)[0]
        corners = [dict(x=float(x/(scene.width-1)),y=float(y/(scene.height-1)))
                   for x,y in scene.quad]
        tracker = FrameTracker(search_worker=Mock())
        current_color = np.dstack((current,current,current,np.full_like(current,255)))
        with patch('vision_pc_bridge.RoiFretboardTracker') as factory:
            factory.return_value.initialize.return_value = dict(roi_status='tracking',status='searching')
            _,result = tracker.initialize_search_result(
                source,current,dict(corners=corners),color=current_color)
            self.assertEqual(result['roi_status'],'tracking')
            self.assertIs(factory.return_value.initialize.call_args.args[0],current)
            self.assertIs(factory.return_value.initialize.call_args.args[2],current_color)
            factory.return_value.update.assert_not_called()

    def test_recovery_cannot_shrink_neck_to_short_lattice_patch(self):
        tracker = FrameTracker(search_worker=Mock())
        tracker.tracker = original = Mock()
        corners = [dict(x=.1,y=.3),dict(x=.9,y=.3),dict(x=.9,y=.6),dict(x=.1,y=.6)]
        short = [dict(x=.1,y=.3),dict(x=.5,y=.3),dict(x=.5,y=.6),dict(x=.1,y=.6)]
        result = dict(status='searching',roi_status='tracking',roi_corners=corners)
        gray = np.zeros((480,640),np.uint8)
        tracker.unsupported_since = 0.
        with patch.object(tracker,'full_search',return_value=(gray,dict(status='tracking'))), \
                patch.object(tracker,'initialize_search_result',return_value=(Mock(),
                    dict(status='tracking',roi_corners=short))):
            recovered = tracker.recover_lattice(gray,None,result,2.)
        self.assertIs(tracker.tracker,original)
        self.assertEqual(recovered['roi_corners'],corners)
        self.assertEqual(recovered['recovery'],'short_lattice_patch_rejected')

    def test_status_reports_missing_and_aging_analysis_without_advancing_frames(self):
        import json
        import time
        driver = Mock()
        driver.fingerprint = {}
        bridge = Bridge(driver)
        server = make_server(bridge,host='127.0.0.1',port=0)
        worker = threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        try:
            route = 'http://127.0.0.1:'+str(server.server_port)+'/api/vision/pc/status'
            with urlopen(route,timeout=2) as response:
                payload = json.load(response)
            self.assertIsNone(payload['analysis_age_seconds'])
            self.assertTrue(payload['tracking_priority_enabled'])
            self.assertFalse(payload['neural_stream_available'])
            self.assertEqual(payload['neural_stream_calls'], 0)
            self.assertEqual(payload['neural_stream_kernel_ms'], 0.)
            bridge.last_analysis = {'sequence':17}
            bridge.last_processed_at = time.monotonic()-.5
            with urlopen(route,timeout=2) as response:
                payload = json.load(response)
            self.assertGreaterEqual(payload['analysis_age_seconds'],.5)
            self.assertEqual(payload['analysis']['sequence'],17)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(2)

    def test_status_reports_optional_neural_stream_telemetry(self):
        import json
        driver = Mock()
        driver.fingerprint = {}
        bridge = Bridge(driver)
        bridge.neural = SimpleNamespace(stream=SimpleNamespace(calls=7, kernel_ms=12.5))
        server = make_server(bridge,host='127.0.0.1',port=0)
        worker = threading.Thread(target=server.serve_forever,daemon=True)
        worker.start()
        try:
            route = 'http://127.0.0.1:'+str(server.server_port)+'/api/vision/pc/status'
            with urlopen(route,timeout=2) as response:
                payload = json.load(response)
            self.assertTrue(payload['neural_stream_available'])
            self.assertEqual(payload['neural_stream_calls'], 7)
            self.assertEqual(payload['neural_stream_kernel_ms'], 12.5)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(2)

    def test_pl_preview_planes_preserve_source_pixels_and_skip_encoding(self):
        gray = np.arange(480*640,dtype=np.uint8).reshape(480,640)
        edge = np.where(gray > 127,255,0).astype(np.uint8)
        driver = Mock()
        driver.process.return_value = dict(gray=gray,edge=edge)
        options = dict(stream='planes',generation=1,sequence=1,enabled=False,
                       preview_scale=2,preview_encoding='planes')
        with patch('vision_pc_bridge.cv2.imencode',side_effect=AssertionError('unnecessary encode')):
            body,_ = Bridge(driver).process(bytes(640*480*4),options)
        self.assertEqual(body,gray[::2,::2].tobytes()+edge[::2,::2].tobytes())
        with self.assertRaises(ValueError):
            FrameTracker.validate_options(dict(options,preview_encoding='unknown'))

    def test_background_search_is_bounded_and_discards_foreign_results(self):
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        worker = LatticeSearch()
        gray = np.zeros((480,640),np.uint8)
        def search(source, edge=None, **_kwargs):
            entered.set()
            release.wait(2)
            done.set()
            return [[dict(x=.1,y=.4),dict(x=.9,y=.4),dict(x=.9,y=.6),dict(x=.1,y=.6)]]
        try:
            with patch('vision_pc_bridge.find_neck_regions_fast', side_effect=search):
                self.assertTrue(worker.submit(gray,None,('first',1)))
                self.assertTrue(entered.wait(1))
                self.assertIsNone(worker.take(('first',1)))
                self.assertFalse(worker.submit(gray,None,('second',1)))
                release.set()
                self.assertTrue(done.wait(1))
            # Wait for the result publication lock, not an assumed sleep.
            import time
            deadline = time.monotonic()+1
            while worker.busy and time.monotonic()<deadline:
                time.sleep(.001)
            self.assertFalse(worker.busy)
            self.assertIsNone(worker.take(('second',1)))
            self.assertIsNone(worker.completed)
        finally:
            release.set()
            worker.stopping = True

    def test_background_search_does_not_start_while_priority_gate_is_closed(self):
        worker = LatticeSearch()
        gray = np.zeros((480,640),np.uint8)
        self.assertFalse(worker.submit(gray,None,('stream',1),can_start=lambda:False))
        self.assertFalse(worker.busy)
        self.assertIsNone(worker.completed)

    def test_new_camera_stream_does_not_inherit_old_search_cooldown(self):
        worker = LatticeSearch()
        worker.identity = ('old-camera', 1)
        worker.next_at = float('inf')
        gray = np.zeros((480, 640), np.uint8)
        with patch('vision_pc_bridge.threading.Thread') as thread:
            self.assertFalse(worker.submit(gray, None, ('old-camera', 1)))
            self.assertTrue(worker.submit(gray, None, ('new-camera', 1)))
            thread.return_value.start.assert_called_once()
            self.assertFalse(worker.submit(gray, None, ('third-camera', 1)))

    def test_lattice_recovery_requests_independent_measurement(self):
        worker = Mock()
        worker.take.return_value = None
        worker.submit.return_value = True
        tracker = FrameTracker(search_worker=worker)
        tracker.identity = ('camera', 1)
        gray = np.zeros((480,640),np.uint8)
        tracker.full_search(gray, None, lattice_only=True)
        worker.submit.assert_called_once_with(gray, None, ('camera',1), None, lattice_only=True)

    def test_lattice_measurement_does_not_wait_for_border_search_cooldown(self):
        worker = LatticeSearch()
        worker.identity = ('camera',1)
        worker.next_at = float('inf')
        with patch('vision_pc_bridge.threading.Thread') as thread:
            self.assertTrue(worker.submit(np.zeros((480,640),np.uint8), None,
                                          ('camera',1), lattice_only=True))
            thread.return_value.start.assert_called_once()
            self.assertFalse(worker.submit(np.zeros((480,640),np.uint8), None,
                                           ('camera',1), lattice_only=True))

    def test_hand_priority_defers_discovery_only_for_a_bounded_window(self):
        now = [10.]
        priority = WorkPriority(clock=lambda:now[0],hand_grace_seconds=.2)
        hand = Mock(busy=True)
        self.assertFalse(priority.allow_background_discovery(hand))
        hand.busy = False
        priority.observe_hand_update({'scheduler':{'request_reason':'reacquire', 'inference_busy':True}})
        self.assertFalse(priority.allow_background_discovery(hand))
        now[0] += .21
        self.assertTrue(priority.allow_background_discovery(hand))

    def test_busy_hand_cannot_starve_discovery_indefinitely(self):
        now = [10.]
        priority = WorkPriority(clock=lambda:now[0],maximum_deferral_seconds=.5)
        hand = Mock(busy=True)
        self.assertFalse(priority.allow_background_discovery(hand))
        now[0] += .49
        self.assertFalse(priority.allow_background_discovery(hand))
        now[0] += .02
        self.assertTrue(priority.allow_background_discovery(hand))

    def test_due_hand_refresh_defers_discovery_before_worker_reports_busy(self):
        now = [10.]
        priority = WorkPriority(clock=lambda:now[0],maximum_deferral_seconds=.5)
        hand = SimpleNamespace(busy=False, track=None, next_detection_at=float('inf'))
        self.assertFalse(priority.allow_background_discovery(hand))
        now[0] += .51
        self.assertTrue(priority.allow_background_discovery(hand))

    def test_default_tracker_owns_an_async_search_worker(self):
        tracker = FrameTracker()
        gray = np.zeros((480,640),np.uint8)
        try:
            with patch.object(AutoFretboardTracker, 'update', side_effect=AssertionError('foreground search')):
                _, result = tracker.full_search(gray, None)
            self.assertIsInstance(tracker.search_worker, LatticeSearch)
            self.assertEqual(result['reason'], 'background_lattice_search')
        finally:
            tracker.search_worker.stopping = True

    def test_background_geometry_requires_present_frame_validation(self):
        tracker = FrameTracker(search_worker=Mock())
        scene = GuitarScene()
        source = scene.render()[0]
        current = np.zeros_like(source)
        corners = [dict(x=float(x/(scene.width-1)),y=float(y/(scene.height-1)))
                   for x,y in scene.quad]
        proposed,result = tracker.initialize_search_result(source,current,{'corners':corners})
        self.assertEqual(result['status'],'searching')
        self.assertEqual(result['roi_status'],'lost')
        self.assertIsNone(result['roi_corners'])
        self.assertEqual(result['reason'],'acquisition_frame_mismatch')

    def test_unsupported_region_does_not_run_foreground_full_lattice_search(self):
        tracker = FrameTracker(fast_tracking=True)
        tracker.tracker = original = Mock()
        gray, edge = np.zeros((480,640),np.uint8), np.zeros((480,640),np.uint8)
        unsupported = {'status':'searching','roi_status':'tracking','strings':[],'frets':[]}
        for now in (0., .5, 1.):
            tracker.recover_lattice(gray,edge,unsupported,now)
        with patch.object(tracker, 'full_search', side_effect=AssertionError('full search in fast mode')):
            self.assertEqual(tracker.recover_lattice(gray,edge,unsupported,1.6)['recovery'],
                             'local_lattice_measurement')
        tracker.recover_lattice(gray,edge,unsupported,2.)
        self.assertIs(tracker.tracker,original)
        self.assertIsNotNone(tracker.unsupported_since)

    def test_native_jpeg_transport_stays_on_ps_and_skips_bgr_copy(self):
        image = np.full((480,640,4), [10,20,30,255], np.uint8)
        with patch('vision_pc_bridge._JPEG_DECODER', return_value=image) as decoder, \
                patch('vision_pc_bridge.cv2.imdecode', side_effect=AssertionError('redundant decode')):
            self.assertEqual(decode_frame(b'\xff\xd8test', 'jpeg'), image.tobytes())
            decoder.assert_called_once()

    def test_busy_and_foreign_streams_skip_expensive_decode(self):
        from http.client import HTTPConnection
        import json
        driver = Mock()
        state = Bridge(driver)
        state.active_stream, state.active_at = ('owner', 1), float('inf')
        server = make_server(state, '127.0.0.1', 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        headers = {'Content-Type':'application/octet-stream', 'X-Vision-Token':'admission',
            'X-Vision-Encoding':'deflate', 'X-Vision-Options':json.dumps(dict(
                stream='competitor', generation=1, sequence=2, enabled=True))}
        try:
            for busy, expected in ((False, 409), (True, 409)):
                connection = HTTPConnection('127.0.0.1', server.server_port, timeout=2)
                if busy:
                    state.lock.acquire()
                try:
                    with patch('vision_pc_bridge.decode_frame', side_effect=AssertionError('discarded frame decoded')) as decode:
                        connection.request('POST', '/api/vision/pc/process', b'invalid', headers)
                        response = connection.getresponse()
                        self.assertEqual(response.status, expected)
                        response.read()
                        decode.assert_not_called()
                finally:
                    connection.close()
                    if busy:
                        state.lock.release()
            driver.process.assert_not_called()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_competing_camera_stream_does_not_reset_model_repeatedly(self):
        driver = Mock()
        driver.process.return_value = {'gray':np.zeros((480,640),np.uint8),
                                      'edge':np.zeros((480,640),np.uint8)}
        bridge = Bridge(driver)
        first = dict(stream='one',generation=0,sequence=1,enabled=False,coordinates_only=True)
        with patch('vision_pc_bridge.time.monotonic',return_value=10.):
            bridge.process(bytes(640*480*4),first)
            bridge.process(bytes(640*480*4),dict(first,stream='two'))
            calls = driver.process.call_count
            with self.assertRaises(StreamBusy):
                bridge.process(bytes(640*480*4),dict(first,sequence=2))
            self.assertEqual(driver.process.call_count,calls)
        with patch('vision_pc_bridge.time.monotonic',return_value=14.):
            bridge.process(bytes(640*480*4),dict(first,sequence=3))
            self.assertEqual(bridge.active_stream,('one',0,None))

    def test_rejected_body_drain_is_bounded_when_client_does_not_send_payload(self):
        from http.client import HTTPConnection
        import time
        server = make_server(Bridge(Mock()),'127.0.0.1',0)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        connection = HTTPConnection('127.0.0.1',server.server_port,timeout=1)
        try:
            started = time.monotonic()
            connection.request('POST','/api/vision/pc/process',body=None,
                headers={'Content-Length':'4096'})
            response = connection.getresponse()
            self.assertEqual(response.status,400)
            response.read()
            self.assertLess(time.monotonic()-started,.8)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_http_reuses_connection_and_closes_rejected_unread_body(self):
        from http.client import HTTPConnection
        driver = Mock()
        driver.process.return_value = {"gray": np.zeros((480,640),np.uint8),
                                       "edge": np.zeros((480,640),np.uint8)}
        server = make_server(Bridge(driver), "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            headers = {"Content-Type":"application/octet-stream", "X-Vision-Token":"keep-1",
                       "X-Vision-Encoding":"deflate"}
            body = zlib.compress(bytes(640*480*4),1)
            original = None
            for _ in range(2):
                connection.request("POST", "/api/vision/pc/process", body, headers)
                response = connection.getresponse()
                self.assertEqual(response.status,200)
                response.read()
                self.assertIsNotNone(connection.sock)
                if original is not None:
                    self.assertIs(connection.sock, original)
                original = connection.sock
            self.assertEqual(driver.process.call_count,2)
            connection.request("POST", "/api/vision/pc/process", b"unread", {})
            response = connection.getresponse()
            self.assertEqual(response.status,400)
            self.assertEqual(response.getheader("Connection"),"close")
            response.read()
            self.assertIsNone(connection.sock)
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_direct_rgba_driver_skips_intermediate_color_conversion(self):
        class Driver:
            def process_rgba(self, frame):
                self.frame = frame
                return {"gray": np.zeros((480,640),np.uint8),
                        "edge": np.zeros((480,640),np.uint8)}
        driver = Driver()
        rgba = np.full((480,640,4), [11,22,33,99], np.uint8)
        with patch("vision_pc_bridge.cv2.cvtColor", side_effect=AssertionError("redundant conversion")), \
                patch.object(FrameTracker, "track", return_value={}):
            Bridge(driver).process(rgba.tobytes(), dict(stream="direct",sequence=1,generation=1,
                enabled=False,corners=None,coordinates_only=True))
        np.testing.assert_array_equal(driver.frame, rgba)

    def test_roi_transport_reconstructs_global_coordinates_and_exact_color(self):
        driver=Mock()
        driver.process.return_value={"gray":np.zeros((480,640),np.uint8),"edge":np.zeros((480,640),np.uint8)}
        options=dict(stream="crop",sequence=1,generation=1,enabled=True,corners=None,
                     coordinates_only=True,transport_roi=dict(x=100,y=60,width=80,height=40))
        cropped=np.full((40,80,4),[11,22,33,255],np.uint8)
        compressed=zlib.compress(cropped.tobytes(),1)
        raw=decode_frame(compressed,"deflate",cropped.nbytes)
        Bridge(driver).process(raw,options)
        image=driver.process.call_args[0][0]
        self.assertEqual(image.shape,(480,640,3))
        np.testing.assert_array_equal(image[60:100,100:180],np.full((40,80,3),[33,22,11],np.uint8))
        self.assertEqual(int(image[:60].sum()),0)
        for roi in (dict(x=-1,y=0,width=1,height=1),dict(x=639,y=0,width=2,height=1),
                    dict(x=True,y=0,width=1,height=1),dict(x=1,y=1,width=0,height=1)):
            with self.assertRaises(ValueError):
                FrameTracker.validate_options(dict(options,transport_roi=roi))

    def test_coordinates_only_skips_jpeg_and_keeps_full_tracking_frame(self):
        driver = Mock()
        driver.process.return_value = {"gray": np.zeros((480,640), np.uint8),
                                       "edge": np.zeros((480,640), np.uint8)}
        options = dict(stream="coords", sequence=1, generation=1, enabled=True,
                       corners=None, coordinates_only=True, fast_tracking=True)
        with patch.object(FrameTracker, "track", return_value={}) as tracking, \
                patch("vision_pc_bridge.cv2.imencode") as encode:
            body, _ = Bridge(driver).process(bytes(640*480*4), options)
        self.assertEqual(body, b"")
        encode.assert_not_called()
        self.assertEqual(tracking.call_args[0][0].shape, (480,640))
        for key in ("coordinates_only", "fast_tracking"):
            with self.assertRaises(ValueError):
                FrameTracker.validate_options(dict(options, **{key: 1}))

    def test_small_preview_does_not_downsample_tracking(self):
        driver = Mock()
        driver.process.return_value = {"gray": np.full((480,640), 70, np.uint8),
                                       "edge": np.full((480,640), 255, np.uint8)}
        bridge = Bridge(driver)
        options = dict(stream="preview", sequence=1, generation=1, enabled=True,
                       corners=None, preview_scale=2)
        with patch.object(FrameTracker, "track", return_value={}) as tracking:
            jpeg, _ = bridge.process(bytes(640*480*4), options)
        self.assertEqual(tracking.call_args[0][0].shape, (480,640))
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(image.shape, (240,640))
        self.assertAlmostEqual(float(image[:,:320].mean()), 70, delta=2)
        self.assertGreater(float(image[:,320:].mean()), 250)
        for scale in (True, 0, 3, "2"):
            with self.assertRaises(ValueError):
                FrameTracker.validate_options(dict(options, preview_scale=scale))

    def test_current_four_tip_projection_skips_redundant_legacy_projection(self):
        driver = Mock()
        driver.process.return_value = {"gray": np.zeros((480,640), np.uint8),
                                       "edge": np.zeros((480,640), np.uint8)}
        for mode in ({"coordinates_only": True}, {"preview_encoding": "planes"}, {}):
            with self.subTest(mode=mode):
                bridge = Bridge(driver)
                projection = {"status": "note_candidates", "positions": [{"midi_note": 47}]}
                bridge.fingertip_notes = Mock()
                bridge.fingertip_notes.project.return_value = {"status": "current", "projection": projection}
                with patch.object(FrameTracker, "track", return_value={}), \
                        patch("vision_pc_bridge.project_positions", side_effect=AssertionError("duplicate projection")), \
                        patch("vision_pc_bridge.time.monotonic", side_effect=(i/1000 for i in range(1000))):
                    bridge.process(bytes(640*480*4), dict(stream="projection", sequence=1,
                        generation=1, enabled=True, corners=None, **mode))
                self.assertIs(bridge.analysis["visual_positions"], projection)
                stages = bridge.analysis["bridge_stages_ms"]
                self.assertGreater(stages["visual_projection_ms"], 0)
                self.assertGreaterEqual(stages["preview_encode_ms"], 0)
                if mode.get("coordinates_only"):
                    self.assertEqual(stages["preview_encode_ms"], 0)

    def test_historical_four_tip_projection_retains_legacy_current_evidence(self):
        driver = Mock()
        driver.process.return_value = {"gray": np.zeros((480,640), np.uint8),
                                       "edge": np.zeros((480,640), np.uint8)}
        bridge = Bridge(driver)
        historical = {"status": "note_candidates", "positions": [{"midi_note": 47}]}
        bridge.fingertip_notes = Mock()
        bridge.fingertip_notes.project.return_value = {"status": "historical", "projection": historical}
        current = {"status": "unlocalized", "positions": []}
        with patch.object(FrameTracker, "track", return_value={}), \
                patch("vision_pc_bridge.project_positions", return_value=current) as project:
            bridge.process(bytes(640*480*4), dict(stream="projection", sequence=2,
                generation=1, enabled=True, corners=None, coordinates_only=True))
        project.assert_called_once_with(bridge.analysis)
        self.assertIs(bridge.analysis["visual_positions"], current)
        self.assertIs(bridge.analysis["fingertip_notes"]["projection"], historical)

    def test_lossless_transport_is_exact_and_bounded(self):
        raw = np.random.default_rng(7).integers(0, 256, 640*480*4, dtype=np.uint8).tobytes()
        self.assertEqual(decode_frame(raw), raw)
        patterned = bytes(range(256)) * (640*480*4//256)
        compressed = zlib.compress(patterned, 1)
        self.assertEqual(decode_frame(compressed, "deflate"), patterned)
        for invalid in (compressed[:-1], compressed+b"trailing", b"invalid",
                        zlib.compress(patterned+b"extra"), zlib.compress(b"short")):
            with self.assertRaises(ValueError):
                decode_frame(invalid, "deflate")
        with self.assertRaises(ValueError):
            decode_frame(compressed, "gzip")

    def test_jpeg_transport_decodes_color_and_enforces_frame_size(self):
        image = np.full((480,640,3),[30,80,150],np.uint8)
        jpeg = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,92])[1].tobytes()
        rgba = np.frombuffer(decode_frame(jpeg,'jpeg'),np.uint8).reshape(480,640,4)
        np.testing.assert_allclose(rgba[100,100],[150,80,30,255],atol=2)
        for body in (b'invalid',jpeg[:12],cv2.imencode('.jpg',image[:240])[1].tobytes()):
            with self.assertRaises(ValueError):
                decode_frame(body,'jpeg')
        with self.assertRaises(ValueError):
            decode_frame(jpeg,'jpeg',100)

    def test_compressed_http_reaches_driver_with_exact_color(self):
        driver = Mock()
        driver.process.return_value = {"gray": np.zeros((480,640), np.uint8),
                                       "edge": np.zeros((480,640), np.uint8)}
        state = Bridge(driver)
        server = make_server(state, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        rgba = np.full((480,640,4), [11,22,33,255], np.uint8)
        try:
            request = Request("http://127.0.0.1:{}/api/vision/pc/process".format(server.server_port),
                data=zlib.compress(rgba.tobytes(), 1), headers={
                    "Content-Type": "application/octet-stream", "X-Vision-Token": "lossless-1",
                    "X-Vision-Encoding": "deflate"})
            with urlopen(request, timeout=3) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())
            np.testing.assert_array_equal(driver.process.call_args[0][0][0,0], [33,22,11])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)

    def test_hardware_tracking_fault_latches_before_next_dma(self):
        driver = Mock()
        driver.process.return_value = {"gray": np.zeros((480, 640), np.uint8),
                                       "edge": np.zeros((480, 640), np.uint8)}
        accelerator = Mock(fault=True)
        bridge = Bridge(driver, flow_backend=accelerator)
        options = {"stream": "fault", "sequence": 1, "generation": 1,
                   "enabled": True, "corners": None}
        with patch.object(FrameTracker, "track", side_effect=TimeoutError("core timeout")):
            with self.assertRaises(TimeoutError):
                bridge.process(bytes(640 * 480 * 4), options)
        self.assertEqual(bridge.fault, "core timeout")
        with self.assertRaises(VisionPLError):
            bridge.process(bytes(640 * 480 * 4), dict(options, sequence=2))
        self.assertEqual(driver.process.call_count, 1)

    def test_automatic_discovery_motion_loss_and_reacquisition_without_corners(self):
        scene = GuitarScene()
        options = {"stream": "automatic", "sequence": 0, "generation": 1,
                   "enabled": True, "corners": None}
        frames = []
        for step in range(11):
            image = (np.zeros((400, 640), np.uint8) if step == 5 else
                     scene.render(angle=0 if step > 5 else max(0, step - 1),
                                  dx=0 if step > 5 else max(0, step - 1))[0])
            image = cv2.resize(image, (640, 480))
            words = reference_words(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)).reshape(480, 640)
            frames.append(((words & 255).astype(np.uint8), ((words >> 16) & 255).astype(np.uint8)))

        class InjectedAsyncSearch:
            """Publishes worker-produced snapshots on the following frame."""
            def __init__(self, completed):
                self.completed = list(completed)
                self.pending = None

            def submit(self, gray, edge, identity, can_start=None):
                if self.pending is not None or can_start is not None and not can_start():
                    return False
                self.pending = self.completed.pop(0) if self.completed else None
                return self.pending is not None

            def take(self, identity):
                result, self.pending = self.pending, None
                return result if result is not None and result[0] == identity else None

        identity = (options['stream'], options['generation'])
        completed = []
        for gray, edge in frames:
            detected = AutoFretboardTracker().update(gray, edge=edge)
            completed.append((identity, gray.copy(), detected))
        tracker = FrameTracker(search_worker=InjectedAsyncSearch(completed))
        results = []
        # No seed geometry enters this pipeline, including after occlusion.
        for step, (gray, edge) in enumerate(frames):
            options["sequence"] += 1
            with patch("vision_pc_bridge.time.monotonic", return_value=step * .6):
                result = tracker.track(gray, options, edge)
            results.append(result)
            self.assertEqual(result["acquisition"], "rulesfallback")
        self.assertEqual(results[0]["reason"], "background_lattice_search", results[0])
        self.assertTrue(any(r["status"] == "tracking" for r in results[1:5]),
                        [(r["status"], r.get("reason")) for r in results])
        self.assertEqual(results[5]["strings"], [])
        self.assertIsNone(results[5].get("roi_corners"))
        self.assertTrue(any(r["status"] == "tracking" for r in results[6:]), results)
        self.assertEqual(results[-1]["discovery_backend"], "rulesfallback")

    def test_motion_lattice_and_occlusion_with_current_pl_gray(self):
        class GoldenProcessor:
            # CPU golden only for host contract testing, never used by service.
            def process(self, frame):
                words = reference_words(frame)
                return {"gray": (words & 255).astype(np.uint8).reshape(480, 640),
                        "edge": ((words >> 16) & 255).astype(np.uint8).reshape(480, 640)}
        scene = GuitarScene()
        bridge = Bridge(GoldenProcessor())
        corners = [{"x": float(x / (scene.width - 1)), "y": float(y / (scene.height - 1))}
                   for x, y in scene.quad]
        options = {"stream": "test-camera", "sequence": 0, "generation": 1,
                   "enabled": True, "corners": corners}
        for step in range(4):
            gray = scene.render(angle=step * 2, dx=step * 2, dy=step)[0]
            rgba = cv2.cvtColor(cv2.resize(gray, (640, 480)), cv2.COLOR_GRAY2RGBA)
            options["sequence"] += 1
            with patch("vision_pc_bridge.time.monotonic", return_value=step * .1):
                bridge.process(rgba.tobytes(), options)
            result = bridge.analysis
            self.assertEqual(result["roi_status"], "tracking", result)
        self.assertEqual(result["status"], "tracking", result)
        self.assertEqual(len(result["strings"]), 6)
        self.assertGreaterEqual(len(result["frets"]), 5)
        self.assertEqual(result["sequence"], 4)
        options["sequence"] += 1
        with patch("vision_pc_bridge.time.monotonic", return_value=.4):
            bridge.process(bytes(640 * 480 * 4), options)
        self.assertNotEqual(bridge.analysis["status"], "tracking")
        self.assertEqual(bridge.analysis["strings"], [])

    def test_invalid_tracking_metadata_prevents_dma(self):
        driver = Mock()
        for options in ({}, {"stream": "a", "sequence": True, "generation": 0, "enabled": True}):
            with self.assertRaises(ValueError):
                Bridge(driver).process(bytes(640 * 480 * 4), options)
        driver.process.assert_not_called()

    def test_stream_owner_prevents_a_copied_tab_from_sharing_the_same_identity(self):
        bridge = Bridge(Mock())
        bridge.active_stream = ("camera", 7, "owner-a")
        bridge.active_at = 10.
        with patch("vision_pc_bridge.time.monotonic", return_value=10.1):
            with self.assertRaises(StreamBusy):
                bridge.check_stream(dict(stream="camera", generation=7, owner="owner-b", sequence=2))
        bridge.check_stream(dict(stream="camera", generation=7, owner="owner-b", sequence=1))
        bridge.check_stream(dict(stream="camera", generation=7, owner="owner-a", sequence=2))

    def test_stream_owner_is_bounded_and_optional_for_compatibility(self):
        base = dict(stream="camera", generation=1, sequence=1, enabled=True)
        Bridge.validate_options(base)
        for owner in ("", "short", "bad_owner", "x" * 65, 7):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                Bridge.validate_options(dict(base, owner=owner))
        Bridge.validate_options(dict(base, owner="12345678"))

    def test_rgba_preserves_color_and_returns_processor_planes(self):
        driver = Mock()
        driver.process.return_value = {"gray": np.full((480, 640), 90, np.uint8),
                                       "edge": np.full((480, 640), 255, np.uint8)}
        rgba = np.empty((480, 640, 4), np.uint8)
        rgba[:] = [11, 22, 33, 255]
        jpeg, elapsed = Bridge(driver).process(rgba.tobytes())
        np.testing.assert_array_equal(driver.process.call_args[0][0][0, 0], [33, 22, 11])
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(image.shape, (480, 1280))
        self.assertAlmostEqual(float(image[:, :640].mean()), 90, delta=2)
        self.assertGreater(float(image[:, 640:].mean()), 250)
        self.assertGreaterEqual(elapsed, 0)

    def test_bad_length_never_dispatches(self):
        driver = Mock()
        with self.assertRaises(ValueError):
            Bridge(driver).process(b"short")
        driver.process.assert_not_called()

    def test_dma_failure_disables_further_dispatch(self):
        driver = Mock()
        driver.process.side_effect = VisionPLError("DMA fault")
        state = Bridge(driver)
        body = bytes(640 * 480 * 4)
        for _ in range(2):
            with self.assertRaises(VisionPLError):
                state.process(body)
        self.assertEqual(driver.process.call_count, 1)

    def test_http_rejects_malformed_or_busy_requests_without_dispatch(self):
        driver = Mock()
        state = Bridge(driver)
        server = make_server(state, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = "http://127.0.0.1:{}/api/vision/pc/process".format(server.server_port)
        try:
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(url, data=b"bad"), timeout=3)
            self.assertEqual(error.exception.code, 400)
            state.lock.acquire()
            try:
                request = Request(url, data=bytes(640 * 480 * 4), headers={
                    "Content-Type": "application/octet-stream", "X-Vision-Token": "test-1"})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=3)
                self.assertEqual(error.exception.code, 429)
            finally:
                state.lock.release()
            driver.process.assert_not_called()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(2)


if __name__ == "__main__":
    unittest.main()

import sys
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hand_tracking import HandModel, HandPipeline


class FakeModel:
    def detect(self, rgba, proposal=None):
        return [{'landmarks':np.float32([(x,y) for y in np.linspace(.3,.6,3)
                      for x in np.linspace(.3,.6,7)]), 'score':.9}]


def cpu_flow(before, after, points):
    return cv2.calcOpticalFlowPyrLK(before,after,points,None,winSize=(15,15),maxLevel=2)


class HandTrackingTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = HandPipeline(FakeModel(),cpu_flow, keypoint_mode='full')
        self.gray = np.random.default_rng(2).integers(0,256,(480,640),dtype=np.uint8)
        self.rgba = np.zeros((480,640,4),np.uint8)

    def tearDown(self):
        self.pipeline.close()

    def options(self,sequence,stream='test'):
        return dict(stream=stream,generation=1,sequence=sequence)

    def ready(self):
        self.pipeline.update(self.rgba,self.gray,self.options(1))
        for _ in range(100):
            with self.pipeline.lock:
                done = self.pipeline.completed is not None
            if done:
                break
            time.sleep(.005)
        return self.pipeline.update(self.rgba,self.gray,self.options(2))

    def test_background_result_is_propagated_to_current_identity(self):
        result = self.ready()
        self.assertEqual(result['inference_location'],'ps_linux')
        self.assertEqual(result['status'],'tracking')
        self.assertEqual(result['hands'][0]['tracked_sequence'],2)
        self.assertFalse(result['hands'][0]['landmarks_observed'])

    def test_stable_current_joints_bound_neural_refresh_without_delaying_recovery(self):
        result = self.ready()
        self.assertEqual(result['status'], 'tracking')
        with self.pipeline.lock:
            self.assertFalse(self.pipeline.busy)
            self.assertIsNone(self.pipeline.pending)
        # Losing current flow must trigger reacquisition immediately, even
        # inside the idle refresh interval.
        self.pipeline.flow = lambda before,after,points: (points.copy(),
            np.zeros((len(points),1),np.uint8),np.zeros((len(points),1)))
        result = self.pipeline.update(self.rgba, np.zeros_like(self.gray), self.options(3))
        self.assertEqual(result['hands'], [])
        self.assertTrue(self.pipeline.busy or self.pipeline.completed is not None)

    def test_stream_change_cannot_reuse_old_hand(self):
        self.ready()
        result = self.pipeline.update(self.rgba,self.gray,self.options(3,'other'))
        self.assertEqual(result['hands'],[])

    def test_verified_track_replenishes_features_after_normal_attrition(self):
        self.ready()
        self.pipeline.close()
        track = self.pipeline.track
        track['features'] = track['features'][:10].copy()
        count = len(track['features'])
        def attrition(before, after, points):
            good = np.ones((len(points),1), np.uint8)
            good[8:count] = 0
            return points.copy(), good, np.zeros((len(points),1))
        advanced = self.pipeline._advance(track, (3, track['gray']), flow=attrition)
        self.assertIsNotNone(advanced)
        self.assertGreater(len(advanced['features']), 8)
        self.assertLessEqual(len(advanced['features']), self.pipeline.feature_limit)

    def test_delayed_negative_detection_preserves_verified_newer_track(self):
        self.ready()
        self.pipeline.close()
        with self.pipeline.lock:
            self.pipeline.completed = (('test', 1), 1, [], time.monotonic(),
                                       2000., self.pipeline.epoch)
        result = self.pipeline.update(self.rgba, self.gray, self.options(3))
        self.assertEqual(result['status'], 'tracking')
        self.assertEqual(result['hands'][0]['tracked_sequence'], 3)
        self.pipeline.flow = lambda before, after, points: (None, None, None)
        self.assertEqual(self.pipeline.update(self.rgba, np.zeros_like(self.gray),
                                             self.options(4))['hands'], [])

    def test_invalid_flow_drops_hand_instead_of_holding_old_box(self):
        self.ready()
        def failed(before,after,points):
            return points.copy(),np.zeros((len(points),1),np.uint8),np.zeros((len(points),1))
        self.pipeline.flow = failed
        result = self.pipeline.update(self.rgba,np.zeros_like(self.gray),self.options(3))
        self.assertEqual(result['hands'],[])

    def test_loss_reuses_last_accepted_model_roi_not_drifted_track(self):
        self.ready()
        expected = self.pipeline._model_proposal.copy()
        self.pipeline.track['landmarks'] += [.1, 0]
        drifted = self.pipeline.track['landmarks'].copy()
        self.pipeline.close()
        self.pipeline.stopping = False
        self.pipeline._catchup = lambda track, record: None
        self.pipeline.update(self.rgba,np.zeros_like(self.gray),self.options(3))
        with self.pipeline.lock:
            proposal = self.pipeline.pending[4]
        np.testing.assert_allclose(proposal, expected)
        self.assertFalse(np.array_equal(proposal, drifted))

    def test_expired_recovery_roi_falls_back_to_full_palm_search(self):
        self.ready()
        self.pipeline.close()
        self.pipeline.stopping = False
        self.pipeline.track = None
        self.pipeline._model_proposal_last_valid_at = (
            time.monotonic()-self.pipeline._RECOVERY_PROPOSAL_TTL-.01)
        self.pipeline.update(self.rgba,self.gray,self.options(3))
        with self.pipeline.lock:
            self.assertIsNone(self.pipeline.pending[4])

    def test_stream_change_cannot_reuse_recovery_roi(self):
        self.ready()
        self.pipeline.close()
        self.pipeline.stopping = False
        self.pipeline.update(self.rgba,self.gray,self.options(3,'other'))
        with self.pipeline.lock:
            self.assertIsNone(self.pipeline.pending[4])

    def test_generation_change_cannot_reuse_old_hand(self):
        self.ready()
        options = dict(self.options(3),generation=2)
        self.assertEqual(self.pipeline.update(self.rgba,self.gray,options)['hands'],[])

    def test_old_detector_result_outside_history_is_not_drawn(self):
        self.ready()
        self.pipeline.close()
        self.pipeline.track = None
        with self.pipeline.lock:
            self.pipeline.completed = (('test',1),0,FakeModel().detect(self.rgba),time.monotonic(),10.,self.pipeline.epoch)
        self.assertEqual(self.pipeline.update(self.rgba,self.gray,self.options(3))['hands'],[])

    def test_reused_identity_rejects_previous_epoch(self):
        self.ready()
        self.pipeline.close()
        old_epoch = self.pipeline.epoch
        self.pipeline.update(self.rgba,self.gray,self.options(1,'other'))
        self.pipeline.update(self.rgba,self.gray,self.options(1))
        with self.pipeline.lock:
            self.pipeline.completed = (('test',1),1,FakeModel().detect(self.rgba),time.monotonic(),10.,old_epoch)
        self.assertEqual(self.pipeline.update(self.rgba,self.gray,self.options(2))['hands'],[])

    def test_fingertip_can_move_independently_of_palm_transform(self):
        self.pipeline.close()
        gray = cv2.resize(self.gray,(320,200),interpolation=cv2.INTER_AREA)
        hand = FakeModel().detect(self.rgba)[0]
        track = self.pipeline._initialize((1,gray),hand,time.monotonic(),10.)
        point = np.rint(hand['landmarks'][8]*[320,200]).astype(int)
        x,y = point
        moved = gray.copy()
        patch = gray[y-7:y+8,x-7:x+8].copy()
        moved[y-7:y+8,x-7:x+8] = 0
        moved[y-7:y+8,x-1:x+14] = patch
        calls = []
        def articulated_flow(before,after,points):
            result = points.copy()
            result[len(track['features'])+8,0,0] += 6 if not calls else -6
            calls.append(len(points))
            return result,np.ones((len(points),1),np.uint8),np.zeros((len(points),1))
        self.pipeline.flow = articulated_flow
        advanced = self.pipeline._advance(track,(2,moved))
        self.assertEqual(len(calls),2)
        self.assertLessEqual(max(calls),64)
        self.assertTrue(advanced['joint_valid'][8])
        self.assertAlmostEqual(advanced['landmarks'][8,0]-track['landmarks'][8,0],6/320,places=5)
        self.assertAlmostEqual(advanced['landmarks'][0,0],track['landmarks'][0,0],places=5)

    def test_bad_local_joint_is_not_promoted_by_good_rigid_hand_track(self):
        self.pipeline.close()
        gray = cv2.resize(self.gray,(320,200),interpolation=cv2.INTER_AREA)
        track = self.pipeline._initialize((1,gray),FakeModel().detect(self.rgba)[0],time.monotonic(),10.)
        count = len(track['features'])
        def bad_joint(before,after,points):
            good = np.ones((len(points),1),np.uint8)
            good[count+8] = 0
            return points.copy(),good,np.zeros_like(good,dtype=float)
        self.pipeline.flow = bad_joint
        occluded = gray.copy()
        x,y = np.int32(track['landmarks'][8]*[320,200])
        occluded[y-8:y+9,x-8:x+9] = 0
        advanced = self.pipeline._advance(track,(2,occluded))
        self.assertIsNotNone(advanced)
        self.assertFalse(advanced['joint_valid'][8])

    def test_static_slow_inference_uses_one_verified_direct_step(self):
        self.ready()
        self.pipeline.close()
        gray = cv2.resize(self.gray,(320,200),interpolation=cv2.INTER_AREA)
        self.pipeline.history.clear()
        self.pipeline.history.extend((i,gray) for i in range(1,31))
        with self.pipeline.lock:
            self.pipeline.completed = (('test',1),1,FakeModel().detect(self.rgba),
                time.monotonic(),100.,self.pipeline.epoch)
        calls = []
        advance = self.pipeline._advance
        def measured(track,record):
            if record[0] > track['sequence']:
                calls.append(record[0])
            return advance(track,record)
        self.pipeline._advance = measured
        result = self.pipeline.update(self.rgba,self.gray,self.options(31))
        self.assertEqual(calls,[31])
        self.assertEqual(calls[-1],31)
        self.assertEqual(result['hands'][0]['tracked_sequence'],31)
        self.assertTrue(result['catchup_diagnostics']['reached_current'])
        self.assertEqual(result['catchup_diagnostics']['attempted_steps'],1)
        self.assertEqual(result['catchup_diagnostics']['adjacent_steps'],0)
        self.assertEqual(result['catchup_diagnostics']['max_sequence_gap'],30)
        self.assertEqual(result['catchup_diagnostics']['mode'],'direct_verified')

    def test_dense_catchup_tracks_known_translation_to_current_frame(self):
        self.pipeline.close()
        self.pipeline.identity = ('test',1)
        rng = np.random.default_rng(13)
        seed = rng.integers(0,256,(200,320),dtype=np.uint8)
        frames = []
        for sequence in range(1,32):
            matrix = np.float32([[1,0,sequence-1],[0,1,0]])
            frames.append((sequence,cv2.warpAffine(seed,matrix,(320,200),borderMode=cv2.BORDER_REFLECT)))
        self.pipeline.history.clear()
        self.pipeline.history.extend(frames[:-1])
        catchup = self.pipeline._catchup
        calls = []
        def adjacent_only(track,record):
            if record[0] > track['sequence']:
                calls.append((track['sequence'],record[0]))
            return None if record[0]-track['sequence'] > 1 else catchup(track,record)
        self.pipeline._catchup = adjacent_only
        with self.pipeline.lock:
            self.pipeline.completed = (('test',1),1,FakeModel().detect(self.rgba),
                time.monotonic(),4930.,self.pipeline.epoch)
        result = self.pipeline.update(self.rgba,frames[-1][1],self.options(31))
        self.assertEqual(result['status'],'tracking')
        self.assertEqual(result['hands'][0]['tracked_sequence'],31)
        self.assertTrue(result['catchup_diagnostics']['reached_current'])
        self.assertEqual(result['catchup_diagnostics']['adjacent_steps'],30)
        self.assertEqual(result['catchup_diagnostics']['attempted_steps'],31)
        self.assertEqual(result['catchup_diagnostics']['mode'],'dense_adjacent')
        self.assertEqual(calls,[(1,31)]+[(i,i+1) for i in range(1,31)])
        self.assertAlmostEqual(result['hands'][0]['landmarks'][0]['x'],.3+30/320,places=2)

    def test_dense_catchup_rejects_occlusion_instead_of_drawing_old_detection(self):
        self.pipeline.close()
        self.pipeline.identity = ('test',1)
        gray = cv2.resize(self.gray,(320,200),interpolation=cv2.INTER_AREA)
        self.pipeline.history.clear()
        self.pipeline.history.extend((sequence,gray.copy()) for sequence in range(1,31))
        occluded = np.zeros_like(gray)
        with self.pipeline.lock:
            self.pipeline.completed = (('test',1),1,FakeModel().detect(self.rgba),
                time.monotonic(),4930.,self.pipeline.epoch)
        result = self.pipeline.update(self.rgba,occluded,self.options(31))
        self.assertEqual(result['hands'],[])
        self.assertEqual(result['acceptance_reason'],'detection_catchup_failed')
        self.assertFalse(result['catchup_diagnostics']['reached_current'])
        self.assertEqual(result['catchup_diagnostics']['reason'],'geometric_check_failed')

    def test_dense_catchup_stops_at_deadline_without_returning_partial_track(self):
        self.pipeline.close()
        gray = cv2.resize(self.gray,(320,200),interpolation=cv2.INTER_AREA)
        self.pipeline.history.clear()
        self.pipeline.history.extend((sequence,gray) for sequence in range(1,31))
        candidate = self.pipeline._initialize((1,gray),FakeModel().detect(self.rgba)[0],time.monotonic(),10.)
        def slow_verified_step(track,record):
            time.sleep(.07)
            if record[0]-track['sequence'] > 1:
                return None
            return dict(track,sequence=record[0],gray=record[1])
        self.pipeline._catchup = slow_verified_step
        self.assertIsNone(self.pipeline._catchup_detection(candidate,list(self.pipeline.history),10.))
        telemetry = self.pipeline.catchup_diagnostics
        self.assertEqual(telemetry['reason'],'time_budget')
        self.assertFalse(telemetry['reached_current'])
        self.assertLess(telemetry['attempted_steps'],30)
        self.assertEqual(telemetry['history_capacity'],36)

    def test_failed_old_positive_catchup_does_not_discard_good_current_track(self):
        self.ready()
        self.pipeline.close()
        current = self.pipeline.track
        with self.pipeline.lock:
            self.pipeline.completed = (('test',1),1,FakeModel().detect(self.rgba),
                time.monotonic(),100.,self.pipeline.epoch)
        advance = self.pipeline._advance
        self.pipeline._advance = lambda track, record, **kwargs: (None if track is not current else advance(track,record,**kwargs))
        result = self.pipeline.update(self.rgba,self.gray,self.options(3))
        self.assertEqual(result['status'],'tracking')
        self.assertEqual(result['hands'][0]['tracked_sequence'],3)

    def test_current_negative_detection_clears_previously_good_track(self):
        self.ready()
        self.pipeline.close()
        with self.pipeline.lock:
            self.pipeline.completed = (('test',1),2,[],time.monotonic(),100.,self.pipeline.epoch)
        self.assertEqual(self.pipeline.update(self.rgba,self.gray,self.options(3))['hands'],[])

    def test_ps_catchup_recovers_when_pl_flow_rejects_identical_real_texture(self):
        self.ready()
        self.pipeline.close()
        track = self.pipeline.track
        self.pipeline.flow = lambda a,b,p: (p.copy(), np.zeros((len(p),1),np.uint8), None)
        advanced = self.pipeline._catchup(track, (3,track['gray'].copy()))
        self.assertIsNotNone(advanced)
        self.assertEqual(advanced['ps_catchup_steps'], 1)
        self.assertEqual(advanced['sequence'], 3)
        self.assertTrue(advanced['ps_tracking_fallback'])
        motion = self.pipeline._observe_motion(track, advanced, time.monotonic())
        self.assertEqual(motion['source'], 'current_ps_joint_flow_recovery')

    def test_ps_catchup_does_not_accept_occluded_hand(self):
        self.ready()
        self.pipeline.close()
        track = self.pipeline.track
        self.pipeline.flow = lambda a,b,p: (p.copy(), np.zeros((len(p),1),np.uint8), None)
        self.assertIsNone(self.pipeline._catchup(track, (3,np.zeros_like(track['gray']))))

    def test_sparse_tracking_reduces_points_without_promoting_untracked_joints(self):
        self.pipeline.close()
        self.pipeline = HandPipeline(FakeModel(), cpu_flow)
        gray = cv2.resize(self.gray,(320,200),interpolation=cv2.INTER_AREA)
        track = self.pipeline._initialize((1,gray),FakeModel().detect(self.rgba)[0],time.monotonic(),10.)
        calls = []
        def measured(before, after, points):
            calls.append(len(points))
            return cpu_flow(before, after, points)
        self.pipeline.flow = measured
        advanced = self.pipeline._advance(track,(2,gray))
        self.assertIsNotNone(advanced)
        self.assertEqual(len(calls), 2)
        self.assertLessEqual(max(calls), 29)
        self.assertTrue(advanced['joint_valid'][[0,1,2,5,6,7,8,9,12,13,16,17,20]].all())
        self.assertFalse(advanced['joint_valid'][[3,4,10,11,14,15,18,19]].any())

    def test_sparse_independent_finger_motion_is_not_rigid_palm_motion(self):
        self.pipeline.close()
        self.pipeline = HandPipeline(FakeModel(), cpu_flow)
        self.pipeline.close()
        gray = cv2.resize(self.gray,(320,200),interpolation=cv2.INTER_AREA)
        track = self.pipeline._initialize((1,gray),FakeModel().detect(self.rgba)[0],time.monotonic(),10.)
        x,y = np.rint(track['landmarks'][8]*[320,200]).astype(int)
        moved = gray.copy()
        patch = gray[y-7:y+8,x-7:x+8].copy()
        moved[y-7:y+8,x-7:x+8] = 0
        moved[y-7:y+8,x-1:x+14] = patch
        calls = []
        def flow(before, after, points):
            result = points.copy()
            offset = list(self.pipeline.tracked_joint_ids).index(8)
            result[len(track['features'])+offset,0,0] += 6 if not calls else -6
            calls.append(len(points))
            return result,np.ones((len(points),1),np.uint8),np.zeros((len(points),1))
        self.pipeline.flow = flow
        advanced = self.pipeline._advance(track,(2,moved))
        self.assertTrue(advanced['joint_valid'][8])
        self.assertAlmostEqual(advanced['landmarks'][8,0]-track['landmarks'][8,0],6/320,places=5)
        self.assertTrue(__import__('hand_tracking').classify_hand_motion(track,advanced,2)['finger_change'])

    def test_stable_schedule_has_deadline_and_does_not_postpone_recovery(self):
        result = self.ready()
        self.assertEqual(result['scheduler']['request_reason'], 'stable_tracking')
        self.assertFalse(result['scheduler']['inference_busy'])
        self.assertGreater(result['scheduler']['next_refresh_ms'], 750)
        self.pipeline.track['joint_valid'][8] = False
        self.pipeline.track['joint_valid'][12] = False
        result = self.pipeline.update(self.rgba,self.gray,self.options(3))
        self.assertEqual(result['scheduler']['request_reason'], 'insufficient_fingertips')

    def test_neural_anchor_deadline_overrides_stable_refresh_wait(self):
        self.ready()
        self.pipeline.next_detection_at = time.monotonic()+10
        self.pipeline.track['detected_at'] = time.monotonic()-4.1
        self.pipeline.track['refreshed_at'] = time.monotonic()-4.1
        result = self.pipeline.update(self.rgba,self.gray,self.options(3))
        self.assertEqual(result['scheduler']['request_reason'], 'anchor_refresh')

    def test_slow_detection_acceptance_starts_cooldown_but_keeps_original_ttl(self):
        self.ready()
        detected_at = time.monotonic()-6.7
        with self.pipeline.lock:
            self.assertFalse(self.pipeline.busy)
            self.pipeline.completed = (('test',1),1,FakeModel().detect(self.rgba),
                detected_at,6700.,self.pipeline.epoch)
        requests = self.pipeline.detection_requests
        result = self.pipeline.update(self.rgba,self.gray,self.options(3))
        self.assertEqual(result['status'],'tracking')
        self.assertEqual(result['scheduler']['request_reason'],'stable_tracking')
        self.assertFalse(result['scheduler']['inference_busy'])
        self.assertEqual(self.pipeline.detection_requests,requests)
        self.assertGreater(result['scheduler']['next_refresh_ms'],1400.)
        self.assertEqual(self.pipeline.track['detected_at'],detected_at)
        self.assertGreater(self.pipeline.track['refreshed_at'],detected_at+6.)
        self.assertGreater(result['hands'][0]['detector_age_ms'],6600.)
        # A recent verified refresh cannot extend the original detector TTL.
        self.pipeline.detector_ttl = 6.6
        result = self.pipeline.update(self.rgba,self.gray,self.options(4))
        self.assertEqual(result['hands'],[])
        self.assertEqual(result['scheduler']['request_reason'],'reacquire')

    def test_anchor_refresh_uses_capture_time_when_refresh_time_is_missing(self):
        self.ready()
        self.pipeline.track.pop('refreshed_at')
        self.pipeline.track['detected_at'] = time.monotonic()-4.1
        self.pipeline.next_detection_at = time.monotonic()+10
        result = self.pipeline.update(self.rgba,self.gray,self.options(3))
        self.assertEqual(result['scheduler']['request_reason'],'anchor_refresh')

    def test_verified_whole_hand_shift_does_not_force_pose_reinference(self):
        self.ready()
        def translated(track, record):
            return dict(track, sequence=record[0], gray=record[1], landmarks=track['landmarks']+[.02,0])
        self.pipeline._advance = translated
        result = self.pipeline.update(self.rgba,self.gray,self.options(3))
        self.assertTrue(result['motion']['hand_shift'])
        self.assertFalse(result['motion']['finger_change'])
        self.assertEqual(result['scheduler']['request_reason'], 'stable_tracking')
        self.assertFalse(result['scheduler']['inference_busy'])

    def test_independent_finger_change_still_requests_pose_confirmation(self):
        self.ready()
        def articulated(track, record):
            points = track['landmarks'].copy()
            points[8] += [.02,0]
            return dict(track,sequence=record[0],gray=record[1],landmarks=points)
        self.pipeline._advance = articulated
        result = self.pipeline.update(self.rgba,self.gray,self.options(3))
        self.assertTrue(result['motion']['finger_change'])
        self.assertEqual(result['scheduler']['request_reason'], 'finger_motion')


class HandModelRecoveryTests(unittest.TestCase):
    def detector(self):
        model = HandModel.__new__(HandModel)
        raw = np.zeros((1,2,18),np.float32)
        raw[0,:,2:4] = 20
        logits = np.full((1,2,1),2.,np.float32)
        model.palm = Mock()
        model.palm.getUnconnectedOutLayersNames.return_value = []
        model.palm.forward.return_value = [raw,logits]
        model.anchors = np.float32([[.2,.2],[.8,.8]])
        model.roi_detections = 0
        model.diagnostics = {}
        model._snapshot_saved = True
        return model

    def test_rejected_roi_continues_to_full_palm_search(self):
        model = self.detector()
        pose = {'landmarks':np.full((21,2),.5,np.float32), 'score':.9}
        model._pose = Mock(side_effect=[None,pose])
        proposal = np.full((21,2),.5,np.float32)
        with patch('hand_tracking.cv2.dnn.NMSBoxes', return_value=np.array([0])):
            hands = model.detect(np.zeros((100,100,4),np.uint8),proposal)
        self.assertEqual(model._pose.call_count,2)
        self.assertEqual(hands[0]['detection_mode'],'full_palm_search')

    def test_second_palm_pose_runs_only_after_first_is_rejected(self):
        model = self.detector()
        pose = {'landmarks':np.full((21,2),.5,np.float32), 'score':.9}
        model._pose = Mock(return_value=pose)
        with patch('hand_tracking.cv2.dnn.NMSBoxes', return_value=np.array([0,1])):
            hands = model.detect(np.zeros((100,100,4),np.uint8))
        self.assertEqual(model._pose.call_count,1)
        self.assertEqual(len(hands),1)

    def test_second_palm_pose_recovers_after_first_rejection(self):
        model = self.detector()
        pose = {'landmarks':np.full((21,2),.5,np.float32), 'score':.9}
        model._pose = Mock(side_effect=[None,pose])
        with patch('hand_tracking.cv2.dnn.NMSBoxes', return_value=np.array([0,1])):
            hands = model.detect(np.zeros((100,100,4),np.uint8))
        self.assertEqual(model._pose.call_count,2)
        self.assertEqual(len(hands),1)

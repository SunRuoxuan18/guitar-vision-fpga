"""Small per-frame motion must not disappear when camera FPS increases."""
import sys
from pathlib import Path
from collections import deque
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_tracking import HandPipeline, classify_hand_motion


def track(sequence, tip_delta=0., palm_delta=0.):
    points = np.full((21,2),.5,np.float32)
    points[:,0] += palm_delta
    points[8,0] += tip_delta
    return dict(sequence=sequence, landmarks=points, joint_valid=np.ones(21,bool))


class MotionWindowTests(unittest.TestCase):
    def pipeline(self):
        pipeline = HandPipeline.__new__(HandPipeline)
        pipeline.motion_samples = deque(maxlen=16)
        return pipeline

    def test_gradual_finger_change_triggers_within_three_frames(self):
        pipeline = self.pipeline()
        before = track(0)
        for step in range(1,4):
            after = track(step,tip_delta=step*.005)
            self.assertFalse(classify_hand_motion(before,after,step)['changing'])
            observed = pipeline._observe_motion(before,after,step/30.)
            before = after
        self.assertTrue(observed['finger_change'])
        self.assertLessEqual(observed['comparison_span_ms'],120)

    def test_gradual_whole_hand_shift_and_stationary_decay(self):
        pipeline = self.pipeline()
        before = track(0)
        for step in range(1,5):
            after = track(step,palm_delta=step*.005)
            observed = pipeline._observe_motion(before,after,step/30.)
            before = after
        self.assertTrue(observed['hand_shift'])
        self.assertFalse(observed['finger_change'])
        for step in range(5,12):
            after = track(step,palm_delta=.02)
            observed = pipeline._observe_motion(before,after,step/30.)
            before = after
        self.assertFalse(observed['changing'])

    def test_invalid_fingertip_is_not_motion_evidence(self):
        pipeline = self.pipeline()
        before, after = track(0), track(1,tip_delta=.1)
        after['joint_valid'][8] = False
        self.assertFalse(pipeline._observe_motion(before,after,0.)['finger_change'])

    def test_frame_stall_discards_expired_motion_baseline(self):
        pipeline = self.pipeline()
        pipeline._observe_motion(track(0),track(1,tip_delta=.02),0.)
        observed = pipeline._observe_motion(track(1,tip_delta=.02),track(2,tip_delta=.02),2.)
        self.assertFalse(observed['changing'])
        self.assertLessEqual(observed['comparison_span_ms'],120)


if __name__ == '__main__':
    unittest.main()

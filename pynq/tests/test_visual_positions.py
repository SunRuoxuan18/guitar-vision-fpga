import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from visual_positions import project_positions


class VisualPositionTests(unittest.TestCase):
    def frame(self):
        def point(x,y):
            return dict(x=x,y=y)
        def line(x0,y0,x1,y1):
            return dict(start=point(x0,y0),end=point(x1,y1),observed=True)
        landmarks = [point(0,0) for _ in range(21)]
        landmarks[8] = point(.4,.32)
        hand = dict(score=.9,tracked_sequence=4,landmarks=landmarks,joint_valid=[True]*21,
                    fresh_for_position_candidates=True,keypoint_source='optical_flow')
        return dict(status='tracking',sequence=4,stage='lattice',
            corners=[point(.1,.2),point(.9,.2),point(.9,.8),point(.1,.8)],
            strings=[line(.1,.2+i*.12,.9,.2+i*.12) for i in range(6)],
            frets=[line(.1+i*.2,.2,.1+i*.2,.8) for i in range(5)],
            hand_detection=dict(hands=[hand]))

    def test_current_joint_maps_to_relative_cell_on_ps(self):
        result = project_positions(self.frame())
        self.assertEqual(result['status'],'relative_positions')
        position = result['positions'][0]
        self.assertEqual((position['string_line'],position['fret_interval']),(2,2))
        self.assertFalse(position['ambiguous'])
        self.assertIsNone(position['pressed'])
        self.assertIsNone(position['sounding'])
        self.assertNotIn('midi_note',position)
        self.assertFalse(result['absolute_frets_known'])

    def test_old_invalid_or_affine_only_joint_does_not_become_evidence(self):
        for change in ('stale','invalid_joint','wrong_sequence','no_lattice'):
            frame = self.frame()
            hand = frame['hand_detection']['hands'][0]
            if change == 'stale': hand['fresh_for_position_candidates'] = False
            elif change == 'invalid_joint': hand['joint_valid'][8] = False
            elif change == 'wrong_sequence': hand['tracked_sequence'] = 3
            else: frame['status'] = 'searching'
            self.assertEqual(project_positions(frame)['positions'],[],change)

    def test_propagated_wires_require_recent_verified_lattice(self):
        frame = self.frame()
        for line in frame['strings']+frame['frets']:
            line['observed'] = False
        self.assertEqual(project_positions(frame)['positions'],[])
        frame.update(stage='tracked_lattice',lattice_age_ms=200,flow=dict(inlier_ratio=.9))
        self.assertEqual(len(project_positions(frame)['positions']),1)
        for stale_age in (601,-1):
            old = copy.deepcopy(frame)
            old['lattice_age_ms'] = stale_age
            self.assertEqual(project_positions(old)['positions'],[])

    def test_hand_outside_measured_frets_explains_absent_positions(self):
        frame = self.frame()
        hand = frame['hand_detection']['hands'][0]
        hand['landmarks'][8] = dict(x=.95,y=.32)
        hand['joint_valid'] = [i == 8 for i in range(21)]
        result = project_positions(frame)
        self.assertEqual(result['positions'], [])
        self.assertEqual(result['reason'], 'hand_outside_measured_lattice')
        self.assertEqual(result['diagnostics']['outside_measured_lattice'], 1)

    def test_current_verified_joint_is_candidate_without_claiming_new_neural_result(self):
        frame = self.frame()
        hand = frame['hand_detection']['hands'][0]
        hand.update(fresh_for_position_candidates=False,current_for_position_candidates=True,
            joint_tracking='pl_lk_per_joint_fb_patch',detector_age_ms=3000,flow_kernel_ms=15.)
        result = project_positions(frame)
        self.assertEqual(len(result['positions']),1)
        self.assertEqual(result['positions'][0]['keypoint_source'],'optical_flow')
        self.assertEqual(result['positions'][0]['neural_age_ms'],3000)
        self.assertIsNone(result['positions'][0]['pressed'])
        for field,value in (('detector_age_ms',8001),('joint_tracking','rigid_affine'),
                            ('flow_kernel_ms',0),('tracked_sequence',3)):
            changed = copy.deepcopy(frame)
            changed['hand_detection']['hands'][0][field] = value
            self.assertEqual(project_positions(changed)['positions'],[])
        hand['joint_valid'][8] = False
        self.assertEqual(project_positions(frame)['positions'],[])


if __name__ == '__main__':
    unittest.main()

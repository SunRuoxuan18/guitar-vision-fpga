from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fingertip_notes import FingertipNotes
from visual_positions import project_positions, validate_note_calibration
import test_visual_positions


class FingertipNoteTests(unittest.TestCase):
    def setUp(self):
        self.frame = test_visual_positions.VisualPositionTests().frame()
        self.frame.update(stream_id='test', generation=2, lattice_epoch=3)
        self.calibration = dict(stream_id='test', generation=2, lattice_epoch=3,
            first_boundary_fret=0, strings_reversed=False, frets_reversed=False,
            tuning=[40, 45, 50, 55, 59, 64])

    def project(self, **changes):
        calibration = dict(self.calibration, **changes)
        return project_positions(self.frame, tips=dict(index=dict(x=.4, y=.32)), calibration=calibration)

    def test_string_fret_midi_note_frequency(self):
        result = self.project()
        point = result['positions'][0]
        self.assertEqual(result['status'], 'note_candidates')
        self.assertEqual((point['string'], point['fret'], point['midi_note'], point['note_name']),
                         (5, 2, 47, 'B2'))
        self.assertAlmostEqual(point['frequency_hz'], 123.471, places=3)
        self.assertIsNone(point['pressed'])
        self.assertIsNone(point['sounding'])

    def test_reversed_strings_and_frets_are_independent(self):
        p = self.project(strings_reversed=True)['positions'][0]
        self.assertEqual((p['string'], p['midi_note'], p['note_name']), (2, 61, 'C#4'))
        p = self.project(first_boundary_fret=7, frets_reversed=True)['positions'][0]
        self.assertEqual((p['fret'], p['midi_note']), (6, 51))
        p = self.project(first_boundary_fret=5)['positions'][0]
        self.assertEqual((p['fret'], p['midi_note']), (7, 52))

    def test_all_strings_and_four_named_fingers(self):
        for row, open_midi in enumerate(self.calibration['tuning']):
            result = project_positions(self.frame,
                tips={f: dict(x=.2, y=.2+row*.12) for f in ('index', 'middle', 'ring', 'little')},
                calibration=self.calibration)
            self.assertEqual(len(result['positions']), 4)
            self.assertTrue(all(p['midi_note'] == open_midi+1 for p in result['positions']))

    def test_perspective_and_nonuniform_frets(self):
        transform = cv2.getPerspectiveTransform(np.float32([[0,0],[1,0],[1,1],[0,1]]),
            np.float32([[.05,.1],[.95,.02],[.85,.92],[.12,.8]]))
        def mapped(p):
            x,y = cv2.perspectiveTransform(np.float32([[[p['x'],p['y']]]]), transform)[0,0]
            return dict(x=float(x), y=float(y))
        for i, wire in enumerate(self.frame['frets']):
            wire['start']['x'] = wire['end']['x'] = [.1,.25,.46,.7,.9][i]
        self.frame['corners'] = [mapped(p) for p in self.frame['corners']]
        for wire in self.frame['strings']+self.frame['frets']:
            for key in ('start', 'end'):
                wire[key] = mapped(wire[key])
        result = project_positions(self.frame, tips=dict(index=mapped(dict(x=.36,y=.32))),
                                   calibration=self.calibration)
        self.assertEqual(result['positions'][0]['midi_note'], 47)

    def test_boundary_and_between_strings_remain_ambiguous(self):
        for tip in (dict(x=.3,y=.32), dict(x=.4,y=.38)):
            result = project_positions(self.frame, tips=dict(index=tip), calibration=self.calibration)
            self.assertTrue(result['positions'])
            self.assertTrue(all(p['ambiguous'] for p in result['positions']))
        result = project_positions(self.frame, tips=dict(index=dict(x=.98,y=.32)), calibration=self.calibration)
        self.assertEqual(result['positions'], [])

    def test_calibration_does_not_survive_grid_or_stream_change(self):
        for key,value in (('stream_id','other'),('generation',3),('lattice_epoch',4)):
            result = self.project(**{key:value})
            self.assertFalse(result['absolute_frets_known'])
            self.assertNotIn('midi_note', result['positions'][0])
        result = self.project(first_boundary_fret=23)
        self.assertEqual(result['calibration_reason'], 'fret_numbers_out_of_range')
        with self.assertRaises(ValueError):
            validate_note_calibration(dict(self.calibration, first_boundary_fret=True))

    def test_history_uses_source_geometry_and_never_becomes_current(self):
        cache = FingertipNotes()
        cache.remember(self.frame, self.calibration)
        current = dict(self.frame, sequence=9, status='searching', strings=[], frets=[], corners=None)
        cache.remember(current, None)
        result = dict(stream_id='test', generation=2, sequence=4, age_ms=2300, fault=None,
                      points=dict(index=dict(x=256., y=153.6, raw_score=-2.)))
        output = cache.project(dict(status='stale', last_result=result), current)
        self.assertEqual(output['status'], 'historical')
        self.assertEqual(output['source_sequence'], 4)
        self.assertEqual(output['projection']['positions'][0]['note_name'], 'B2')
        result['age_ms'] = 9000
        self.assertEqual(cache.project(dict(last_result=result), current)['status'], 'unavailable')

    def test_current_points_and_missing_source_and_invalid_tips(self):
        cache = FingertipNotes(max_frames=1)
        cache.remember(self.frame, self.calibration)
        neural = dict(status='ok', stream_id='test', generation=2, sequence=4, age_ms=100,
            points=dict(index=dict(x=256., y=153.6), little=dict(x=250., y=150., visible=False),
                        middle=dict(x=float('nan'), y=100.)))
        output = cache.project(neural, self.frame)
        self.assertEqual(output['status'], 'current')
        self.assertEqual(len(output['projection']['positions']), 1)
        later = dict(self.frame, sequence=5)
        cache.remember(later, self.calibration)
        self.assertEqual(cache.project(neural, later)['reason'], 'source_frame_geometry_unavailable')
        changed = dict(later, stream_id='new')
        cache.remember(changed, None)
        self.assertEqual(cache.project(neural, changed)['reason'], 'neural_source_mismatch')

    def test_tracked_partial_tips_project_through_current_geometry(self):
        cache = FingertipNotes()
        current = dict(self.frame, sequence=9)
        cache.remember(current, self.calibration)
        neural = dict(status='stale', last_result=dict(stream_id='test', generation=2,
            sequence=4, age_ms=1000, points=dict(index=dict(x=100., y=100.))))
        tracking = dict(status='tracking', stream_id='test', generation=2, sequence=9,
            source_sequence=4, keypoint_source='optical_flow', detector_age_ms=1200.,
            points=dict(index=dict(x=256.,y=153.6),little=dict(x=250.,y=150.,visible=False),
                        middle=dict(x='bad',y=100.),ring=dict(x=900.,y=150.)))
        output = cache.project(neural, current, tracking)
        self.assertEqual((output['status'],output['source_sequence'],output['detector_sequence']),
                         ('current',9,4))
        positions = output['projection']['positions']
        self.assertEqual(len(positions),1)
        self.assertEqual(positions[0]['note_name'],'B2')
        self.assertEqual(positions[0]['keypoint_source'],'optical_flow')
        self.assertEqual(positions[0]['neural_age_ms'],1200.)
        for key, value in (('sequence',8),('generation',3),('stream_id','other'),
                           ('detector_age_ms',4001),('status','lost'),('keypoint_source','affine')):
            invalid = dict(tracking,**{key:value})
            self.assertNotEqual(cache.project(None,current,invalid)['status'],'current')
        self.assertEqual(cache.project(dict(neural,status='fault'),current,tracking)['reason'],'neural_fault')

    def test_current_flow_note_can_be_supported_by_live_audio(self):
        from av_fusion import fuse, publish_visual, read_visual
        import tempfile
        cache = FingertipNotes()
        cache.remember(self.frame, self.calibration)
        tracking = dict(status='tracking',stream_id='test',generation=2,sequence=4,
            source_sequence=1,keypoint_source='optical_flow',detector_age_ms=1000.,
            points=dict(index=dict(x=256.,y=153.6)))
        current = dict(self.frame,visual_positions=cache.project(None,self.frame,tracking)['projection'],
                       processing_completed_monotonic=10.05)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'visual.json'
            publish_visual(path,current,10.)
            visual = read_visual(path)
            audio = dict(status='recognized',stale=False,fft={'analysis_test_mode':0},
                chord={'label':'B'},observed_monotonic=10.02,
                notes=[dict(midi_note=47,confidence=.9)])
            output = fuse(audio,visual,10.1)
            self.assertEqual(output['status'],'audio_visual_supported')
            self.assertEqual(output['matches'][0]['midi_note'],47)
            self.assertFalse(output['pressed_strings_known'])


if __name__ == '__main__':
    unittest.main()

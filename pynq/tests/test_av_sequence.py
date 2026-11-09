"""Synthetic musical/geometry integration, separate from camera/model accuracy."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_fft import FFTSnapshot
from audio_music_server import MusicState
from av_fusion import publish_visual
from visual_positions import project_positions
from test_audio_music import synthesize


class SequenceTests(unittest.TestCase):
    def geometry(self, sequence, fingers):
        def point(x, y):
            return dict(x=float(x), y=float(y))
        def line(a, b):
            return dict(start=a, end=b, observed=True)
        xs = .1 + .8 * (1-2.**(-np.arange(9)/12))/(1-2.**(-8/12))
        ys = np.linspace(.2, .8, 6)
        landmarks = [point(0, 0) for _ in range(21)]
        for finger, string, fret in fingers:
            landmarks[finger] = point((xs[fret-1]+xs[fret])/2, ys[string-1])
        frame = dict(status='tracking', sequence=sequence, stream_id='synthetic-sequence', generation=1,
            lattice_epoch=1,
            stage='lattice', corners=[point(.1,.2),point(.9,.2),point(.9,.8),point(.1,.8)],
            strings=[line(point(xs[0],y),point(xs[-1],y)) for y in ys],
            frets=[line(point(x,ys[0]),point(x,ys[-1])) for x in xs],
            hand_detection=dict(hands=[dict(score=.99, tracked_sequence=sequence,
                fresh_for_position_candidates=True, keypoint_source='fixture',
                landmarks=landmarks, joint_valid=[True]*21)]))
        frame['visual_positions'] = project_positions(frame)
        return frame

    def test_c_am_f_g_waveform_and_independent_position_sequence(self):
        cases = (
            ('C', (48,52,55,60,64), ((16,2,3),(12,3,2),(8,5,1))),
            ('Am', (45,52,57,60,64), ((16,4,2),(12,3,2),(8,5,1))),
            ('F', (41,48,53,57,60,65), ((16,2,3),(20,3,3),(12,4,2),(8,6,1))),
            ('G', (43,47,50,55,59,67), ((8,2,2),(12,1,3),(16,6,3))))
        source = type('Source', (), {'read_snapshot':lambda obj:obj.snapshot})()
        with tempfile.TemporaryDirectory() as directory:
            visual_path = Path(directory)/'visual.json'
            state = MusicState(source, source_kind='synthetic_fixture', visual_path=visual_path)
            sequence = 0
            for label, notes, fingers in cases:
                power = synthesize(notes)
                for _ in range(5):
                    sequence += 1
                    now = 10 + sequence*.1
                    geometry = self.geometry(sequence, fingers)
                    self.assertEqual(len(geometry['visual_positions']['positions']), len(fingers))
                    publish_visual(visual_path, geometry, now-.025)
                    source.snapshot = FFTSnapshot(sequence, power)
                    with patch('audio_music_server.time.monotonic', return_value=now):
                        state.update()
                        result = state.snapshot()
                self.assertEqual(result['stable_chord']['label'], label, result['stable_chord'])
                fusion = result['fusion']
                self.assertIn(fusion['status'], ('mapping_candidate','mapping_ambiguous','audio_visual_aligned'))
                if label != 'C':
                    self.assertEqual(fusion['status'],'audio_visual_aligned')
                self.assertTrue(any(not h['reversed_strings'] and not h['reversed_frets']
                    and h['first_fret']==0 for h in fusion['mapping_hypotheses']), fusion)
                self.assertFalse(fusion['pressed_strings_known'])
                self.assertFalse(fusion['sounding_strings_known'])
                self.assertFalse(fusion['absolute_frets_known'])
                json.dumps(result, allow_nan=False)
            with patch('audio_music_server.time.monotonic', return_value=now+.4):
                self.assertEqual(state.snapshot()['fusion']['reason'], 'vision_stale')


if __name__ == '__main__':
    unittest.main()

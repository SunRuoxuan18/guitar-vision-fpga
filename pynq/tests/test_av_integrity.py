"""Adversarial timing and evidence tests independent of board hardware."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_fft import FFTSnapshot
from audio_music import MusicConfig, recognize_chord
from audio_music_server import MusicState
from av_fusion import fuse, visual_transition


def audio_frame(observed=10.):
    notes = [dict(midi_note=p, confidence=.99) for p in (48, 52, 55)]
    return dict(status='recognized', signal_active=True, stale=False, sequence=2,
                fft={'analysis_test_mode': 0}, notes=notes, chord=recognize_chord(notes),
                observed_monotonic=observed)


def visual_frame(received=10.):
    return dict(received_monotonic=received, processing_completed_monotonic=received+.05,
        sequence=8, positions=[dict(finger='index', midi_note=48, relative=False),
                               dict(finger='middle', midi_note=52, relative=False)],
        hand_motion=dict(sequence=8, changing=True, source='current_pl_joint_flow'))


class FusionIntegrityTests(unittest.TestCase):
    def test_fresh_but_differently_timed_modalities_cannot_form_joint_verdict(self):
        result = fuse(audio_frame(9.5), visual_frame(10.), 10.1)
        self.assertEqual(result['reason'], 'audio_visual_time_mismatch')
        self.assertFalse(result['joint_decision']['visual_participation'])
        self.assertEqual(result['matches'], [])

    def test_future_or_nonfinite_audio_observation_cannot_fuse(self):
        for observed in (11., float('nan'), float('inf'), True, '10'):
            result = fuse(audio_frame(observed), visual_frame(), 10.1)
            self.assertEqual(result['reason'], 'audio_visual_time_mismatch')

    def test_confident_tones_only_define_joint_chord(self):
        audio = audio_frame()
        for extra in (dict(midi_note=61, confidence=.2),
                      dict(midi_note=61, confidence=.99, octave_ambiguous=True)):
            with self.subTest(extra=extra):
                candidate = copy.deepcopy(audio)
                candidate['notes'].append(extra)
                result = fuse(candidate, visual_frame(), 10.1)
                self.assertEqual(result['voicing_midi_notes'], [48, 52, 55])
                self.assertEqual(result['joint_decision']['status'], 'audio_visual_consistent')

    def test_motion_after_audio_observation_cannot_accelerate_that_audio(self):
        self.assertFalse(visual_transition(visual_frame(), 10.1, audio_observed_at=9.5))
        self.assertTrue(visual_transition(visual_frame(), 10.1, audio_observed_at=10.))

    def test_same_finger_multiple_cells_never_count_as_two_fingers(self):
        visual = visual_frame()
        visual['positions'][1]['finger'] = 'index'
        result = fuse(audio_frame(), visual, 10.1)
        self.assertFalse(result['joint_decision']['visual_participation'])

    def test_two_matches_cannot_hide_a_third_conflicting_finger(self):
        visual = visual_frame()
        visual['positions'].append(dict(finger='ring', midi_note=61, relative=False))
        result = fuse(audio_frame(), visual, 10.1)
        self.assertEqual(result['joint_decision']['status'], 'visual_audio_conflict')
        self.assertFalse(result['joint_decision']['visual_participation'])


class AudioPublicationIntegrityTests(unittest.TestCase):
    def state(self):
        class Recognizer:
            config = MusicConfig()

            def analyze(self, power, sequence):
                result = audio_frame()
                result.update(sequence=sequence, spectrum={'rms': .1})
                return result

        source = type('Source', (), {'read_snapshot': lambda obj: obj.snapshot})()
        return MusicState(source, recognizer=Recognizer()), source

    def test_delayed_audio_packet_cannot_replace_newer_publication(self):
        state, source = self.state()
        for sequence, now in ((9, 10.), (10, 10.1), (9, 10.2), (8, 10.3)):
            source.snapshot = FFTSnapshot(sequence, np.zeros(2049))
            with patch('audio_music_server.time.monotonic', return_value=now):
                state.update()
                result = state.snapshot()
        self.assertEqual(result['sequence'], 10)
        self.assertEqual(result['age_seconds'], .2)

    def test_observation_timestamp_survives_slow_recognition(self):
        state, source = self.state()
        source.snapshot = FFTSnapshot(1, np.zeros(2049))
        clock = [10.]
        analyze = state.recognizer.analyze

        def slow_analyze(*args):
            clock[0] = 10.4
            return analyze(*args)

        state.recognizer.analyze = slow_analyze
        with patch('audio_music_server.time.monotonic', side_effect=lambda: clock[0]):
            state.update()
            result = state.snapshot()
        self.assertEqual(result['observed_monotonic'], 10.)
        self.assertEqual(result['timing']['observation_to_result_ms'], 400.)

    def test_audio_sequence_wrap_remains_forward(self):
        state, source = self.state()
        for sequence, now in ((0xffffffff, 10.), (0, 10.1)):
            source.snapshot = FFTSnapshot(sequence, np.zeros(2049))
            with patch('audio_music_server.time.monotonic', return_value=now):
                state.update()
                result = state.snapshot()
        self.assertEqual(result['sequence'], 0)
        self.assertEqual(result['stable_chord']['label'], 'C')


if __name__ == '__main__':
    unittest.main()

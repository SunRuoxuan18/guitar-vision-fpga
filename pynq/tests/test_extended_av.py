"""Register-independent chord grammar and causal movement/fusion regressions."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_music import CHORD_TYPES, MusicConfig, NOTE_NAMES, recognize_chord
from av_fusion import FretMappingTracker, attach_joint_decision, fuse, visual_transition
from hand_tracking import classify_hand_motion
from music_tracking import MusicTracker
from test_music_tracking import observation


class ExtendedAVTests(unittest.TestCase):
    def test_all_chord_roots_registers_and_inversions_keep_intended_interpretation(self):
        for quality, suffix, intervals in CHORD_TYPES:
            for root in range(12):
                for base in (48, 60, 72):
                    pitches = sorted(base+root+step for step in intervals)
                    for inversion in range(len(pitches)):
                        ordered = pitches[inversion:]+[p+12 for p in pitches[:inversion]]
                        notes = [dict(midi_note=p, confidence=.95) for p in ordered]
                        result = recognize_chord(notes)
                        choices = [result]+result['alternatives']
                        with self.subTest(quality=quality, root=root, base=base, inversion=inversion):
                            self.assertTrue(any(c['quality'] == quality and c['root'] == NOTE_NAMES[root]
                                                for c in choices), result)
                            self.assertEqual(result['bass'], NOTE_NAMES[min(ordered) % 12])

    def test_default_range_includes_every_standard_string_through_24th_fret(self):
        config = MusicConfig()
        self.assertLessEqual(config.min_midi, 40)
        self.assertGreaterEqual(config.max_midi, 88)

    def test_octave_shift_preserves_name_but_not_audio_register(self):
        low = [dict(midi_note=p, confidence=.95) for p in (48, 52, 55, 58, 62)]
        high = [dict(n, midi_note=n['midi_note']+12) for n in low]
        self.assertEqual(recognize_chord(low)['label'], 'C9')
        self.assertEqual(recognize_chord(high)['label'], 'C9')
        self.assertNotEqual([n['midi_note'] for n in low], [n['midi_note'] for n in high])

    def test_high_fret_mapping_is_not_forced_down_an_octave(self):
        for start in (0, 5, 12, 16):
            for reversed_strings in (False, True):
                visual = dict(received_monotonic=10., fret_count=8, positions=[
                    dict(finger='index', hand_index=0, string_line=2, fret_interval=1, relative=True),
                    dict(finger='middle', hand_index=0, string_line=3, fret_interval=2, relative=True),
                    dict(finger='ring', hand_index=0, string_line=5, fret_interval=3, relative=True)])
                tuning = (40, 45, 50, 55, 59, 64)
                midis = [tuning[6-p['string_line'] if reversed_strings else p['string_line']-1]
                         +start+p['fret_interval'] for p in visual['positions']]
                audio = dict(status='recognized', stale=False, fft={'analysis_test_mode': 0},
                             notes=[dict(midi_note=p, confidence=.99) for p in midis])
                audio['chord'] = recognize_chord(audio['notes'])
                result = fuse(audio, visual, 10.1)
                candidates = result['mapping_hypotheses']
                self.assertTrue(any(c['first_fret'] == start and c['reversed_strings'] == reversed_strings
                                    and not c['reversed_frets'] for c in candidates), result)
                self.assertEqual(result['voicing_midi_notes'], sorted(set(midis)))
                self.assertFalse(result['absolute_frets_known'])

    def test_duplicate_finger_projections_cannot_vote_twice(self):
        visual = dict(received_monotonic=10., fret_count=8, positions=[
            dict(finger='index', string_line=2, fret_interval=1, relative=True),
            dict(finger='index', string_line=3, fret_interval=2, relative=True)])
        audio = dict(status='recognized', notes=[dict(midi_note=p, confidence=.99) for p in (46, 52)])
        self.assertFalse(fuse(audio, visual, 10.1)['mapping_hypotheses'])

    def track(self):
        return dict(landmarks=np.full((21, 2), .5, dtype=np.float32), joint_valid=np.ones(21, dtype=bool))

    def test_one_verified_finger_change_triggers_without_moving_palm(self):
        before = self.track()
        after = copy.deepcopy(before)
        after['landmarks'][8, 0] += .02
        result = classify_hand_motion(before, after, 2)
        self.assertTrue(result['changing'])
        self.assertTrue(result['finger_change'])
        self.assertFalse(result['hand_shift'])

    def test_whole_hand_shift_triggers_without_articulation(self):
        before = self.track()
        after = copy.deepcopy(before)
        after['landmarks'][:, 0] += .03
        result = classify_hand_motion(before, after, 2)
        self.assertTrue(result['changing'])
        self.assertTrue(result['hand_shift'])
        self.assertFalse(result['finger_change'])

    def test_invalid_joint_or_subthreshold_jitter_is_not_a_transition(self):
        before = self.track()
        for displacement, valid in ((.005, True), (.03, False)):
            after = copy.deepcopy(before)
            after['landmarks'][8, 0] += displacement
            after['joint_valid'][8] = valid
            self.assertFalse(classify_hand_motion(before, after, 2)['changing'])

    def test_current_visual_c7_to_c_change_is_not_mistaken_for_decay(self):
        for visual, expected in ((True, 'C'), (False, 'C7')):
            tracker = MusicTracker()
            for seq, now in ((1, 0.), (2, .1)):
                frame = observation(seq, (48, 52, 55, 58))
                frame['spectrum'] = {'rms': .1}
                tracker.update(frame, now)
            for seq, now in ((3, .2), (4, .29)):
                frame = observation(seq)
                frame.update(visual_transition=visual, spectrum={'rms': .03})
                result = tracker.update(frame, now)
            self.assertEqual(result['stable_chord']['label'], expected)

    def test_complex_high_position_sequence_uses_two_distinct_audio_frames(self):
        tracker = MusicTracker()
        now, seq = 0., 0
        for pitches in ((60, 64, 67, 70), (60, 64, 67, 74), (60, 65, 67), (60, 63, 66, 69)):
            expected = recognize_chord([dict(midi_note=p, confidence=.9) for p in pitches])['label']
            for _ in range(2):
                seq += 1
                frame = observation(seq, pitches)
                frame['visual_transition'] = True
                result = tracker.update(frame, now)
                now += .1
            self.assertEqual(result['stable_chord']['label'], expected)

    def test_motion_timestamp_and_source_must_match_current_frame(self):
        frame = dict(received_monotonic=10., sequence=8,
                     hand_motion=dict(sequence=8, changing=True, source='current_pl_joint_flow'))
        self.assertTrue(visual_transition(frame, 10.1))
        for changes in ({'received_monotonic': 9.}, {'received_monotonic': 11.}, {'sequence': 9},
                        {'hand_motion': dict(sequence=8, changing=True, source='pc_model')}):
            self.assertFalse(visual_transition(dict(frame, **changes), 10.1))

    def test_processing_timestamp_does_not_make_an_old_frame_fresh(self):
        from av_fusion import visual_age
        visual = dict(received_monotonic=10.,processing_completed_monotonic=10.4,
            sequence=8,hand_motion=dict(sequence=8,changing=True,source='current_pl_joint_flow'))
        self.assertAlmostEqual(visual_age(visual,10.45),.45)
        self.assertTrue(visual_transition(visual,10.45))
        for changes,now in (({'received_monotonic':9.},10.45),({},10.71),
                            ({'processing_completed_monotonic':10.5},10.45),
                            ({'processing_completed_monotonic':9.9},10.45),
                            ({'processing_completed_monotonic':float('nan')},10.45),
                            ({'processing_completed_monotonic':None},10.45)):
            with self.subTest(changes=changes,now=now):
                self.assertIsNone(visual_age(dict(visual,**changes),now))
                self.assertFalse(visual_transition(dict(visual,**changes),now))

    def test_joint_verdict_needs_actual_positions_and_current_chord_tones(self):
        notes = [dict(midi_note=p,confidence=.95) for p in (60,64,67,70)]
        result = dict(chord=recognize_chord(notes),voicing_midi_notes=[60,64,67,70],
                      aligned_positions=[])
        self.assertIsNone(attach_joint_decision(result)['joint_decision']['chord'])
        result['aligned_positions'] = [dict(finger='index',midi_note=60,audio_supported=True),
                                       dict(finger='middle',midi_note=64,audio_supported=True)]
        decision = attach_joint_decision(result)['joint_decision']
        self.assertEqual(decision['status'],'audio_visual_consistent')
        self.assertEqual(decision['chord']['label'],'C7')
        self.assertFalse(decision['complete_fingering_known'])
        result['voicing_midi_notes'] = [60,64,67,74]
        self.assertEqual(attach_joint_decision(result)['joint_decision']['status'],'awaiting_audio_transition')


if __name__ == '__main__':
    unittest.main()

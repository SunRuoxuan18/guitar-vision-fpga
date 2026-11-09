import copy
import tempfile
from pathlib import Path
import unittest
from av_fusion import fuse, publish_visual, read_visual, FretMappingTracker


class FusionTests(unittest.TestCase):
    def test_non_object_exchange_is_not_visual_state(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'visual.json'
            for raw in ('[]', 'null', '42', '"text"'):
                path.write_text(raw)
                self.assertIsNone(read_visual(path))

    def audio(self):
        return dict(status='recognized',stale=False,fft={'analysis_test_mode':0},
            chord={'status':'recognized','label':'C'},
            notes=[dict(midi_note=n,confidence=.9) for n in (48,52,55,60,64)])

    def visual(self):
        return dict(received_monotonic=10.,fret_count=8,positions=[
            dict(finger='ring',string_line=2,fret_interval=3,relative=True),
            dict(finger='middle',string_line=3,fret_interval=2,relative=True),
            dict(finger='index',string_line=5,fret_interval=1,relative=True)])

    def test_relative_positions_participate_without_fabricated_absolute_proof(self):
        result = fuse(self.audio(),self.visual(),10.1)
        self.assertTrue(result['mapping_hypotheses'])
        self.assertIn(result['status'],('mapping_candidate','mapping_ambiguous'))
        self.assertFalse(result['absolute_frets_known'])
        self.assertFalse(result['matches'])
        self.assertFalse(result['synchronized'])
        self.assertTrue(all(p['pressed'] is None for p in result['mapping_hypotheses'][0]['positions']))

    def test_stale_or_future_positions_never_support_chord(self):
        for now in (9.9,10.31):
            self.assertEqual(fuse(self.audio(),self.visual(),now)['reason'],'vision_stale')

    def test_silence_and_internal_test_cannot_fuse(self):
        for changes in ({'status':'silence'},{'fft':{'analysis_test_mode':2}},{'stale':True}):
            audio = dict(self.audio(),**changes)
            self.assertFalse(fuse(audio,self.visual(),10.1)['mapping_hypotheses'])

    def test_absolute_independent_positions_match(self):
        visual = self.visual()
        visual['positions'] = [dict(finger='index',midi_note=60,relative=False)]
        self.assertEqual(fuse(self.audio(),visual,10.1)['status'],'audio_visual_supported')

    def test_octave_ambiguity_cannot_anchor(self):
        audio = self.audio()
        for note in audio['notes']:
            note['octave_ambiguous'] = True
        self.assertFalse(fuse(audio,self.visual(),10.1)['mapping_hypotheses'])

    def test_atomic_exchange_keeps_current_and_clears_missing_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'visual.json'
            self.assertIsNone(read_visual(path))
            publish_visual(path,{'sequence':1,'visual_positions':{'positions':[{'finger':'index'}]}},10.)
            self.assertEqual(read_visual(path)['sequence'],1)
            publish_visual(path,{'sequence':2},10.1)
            self.assertEqual(read_visual(path)['positions'],[])

    def test_starting_audio_with_null_fft_does_not_break_snapshot(self):
        self.assertEqual(fuse(dict(status='starting',fft=None), self.visual(), 10.1)['reason'],
                         'live_audio_unavailable')

    def aligned(self):
        from audio_music import recognize_chord
        tracker = FretMappingTracker()
        visual = dict(self.visual(), stream='one',generation=1,lattice_epoch=1,sequence=1)
        audio = self.audio()
        audio.update(sequence=1,chord=recognize_chord(audio['notes']))
        self.assertNotEqual(tracker.update(audio,visual,10.1)['status'],'audio_visual_aligned')
        second = dict(audio,sequence=2,notes=[dict(midi_note=n,confidence=.9) for n in (45,52,57,60,64)])
        second['chord'] = recognize_chord(second['notes'])
        visual = dict(visual,sequence=2,received_monotonic=10.2,positions=[
            dict(finger='ring',string_line=4,fret_interval=2,relative=True),
            dict(finger='middle',string_line=3,fret_interval=2,relative=True),
            dict(finger='index',string_line=5,fret_interval=1,relative=True)])
        result = tracker.update(second,visual,10.25)
        return tracker,second,visual,result

    def test_two_harmonies_align_grid_without_claiming_independent_visual_proof(self):
        tracker,audio,visual,result = self.aligned()
        self.assertEqual(result['status'],'audio_visual_aligned')
        self.assertTrue(result['aligned_positions'])
        self.assertFalse(result['independent_visual_anchor'])
        self.assertFalse(result['absolute_frets_known'])
        self.assertFalse(result['pressed_strings_known'])
        self.assertFalse(result['sounding_strings_known'])

    def test_alignment_cannot_survive_changed_grid_stream_or_expired_evidence(self):
        for changes in ({'lattice_epoch':2},{'stream':'other'},{'fret_count':9}):
            tracker,audio,visual,result = self.aligned()
            changed = dict(visual,**changes)
            self.assertNotEqual(tracker.update(audio,changed,10.26)['status'],'audio_visual_aligned')
        tracker,audio,visual,result = self.aligned()
        fresh = dict(visual, received_monotonic=15.9)
        current = fuse(audio,fresh,16.)
        self.assertNotEqual(tracker.apply_anchor(current,fresh,16.)['status'],'audio_visual_aligned')
        self.assertEqual(tracker.update(audio,visual,16.)['reason'],'vision_stale')

    def test_different_ambiguous_names_do_not_block_measured_high_fret_alignment(self):
        tracker = FretMappingTracker()
        def update(sequence, pitches, cells, now):
            from audio_music import recognize_chord
            audio = dict(status='recognized',stale=False,sequence=sequence,
                notes=[dict(midi_note=n,confidence=.99) for n in pitches])
            audio['chord'] = recognize_chord(audio['notes'])
            visual = dict(received_monotonic=now,sequence=sequence,stream='one',generation=1,
                lattice_epoch=1,fret_count=8,positions=[
                    dict(finger=finger,string_line=line,fret_interval=cell,relative=True)
                    for finger,line,cell in cells])
            return tracker.update(audio,visual,now+.01)
        # In this cropped high-register window, first_fret=12. Two different
        # pitch collections constrain the alignment; neither names pressure.
        update(1,(60,64,67,70), [('index',2,3),('middle',3,2),('ring',4,3)],10.)
        result = update(2,(60,65,67,72), [('index',2,3),('middle',3,3),('ring',4,5)],10.2)
        self.assertEqual(result['status'],'audio_visual_aligned')
        self.assertEqual(result['hand_fret_span'],[15,17])
        self.assertEqual(result['joint_decision']['status'],'audio_visual_consistent')
        self.assertFalse(result['absolute_frets_known'])
        self.assertFalse(result['pressed_strings_known'])


if __name__ == '__main__':
    unittest.main()

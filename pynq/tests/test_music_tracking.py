"""Sequence-level checks: silence, transitions and modal ambiguity."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio_music import recognize_chord
from music_tracking import MusicTracker


def observation(sequence, midis=(48, 52, 55), status="recognized", confidence=0.9):
    notes = [{"midi_note": midi, "confidence": confidence, "relative_strength": 1.0}
             for midi in midis]
    return {"sequence": sequence, "status": status, "signal_active": status == "recognized",
            "notes": notes, "chord": recognize_chord(notes)}


class TrackerTests(unittest.TestCase):
    def test_visible_c_to_am_change_needs_one_new_pitch_class_and_two_audio_frames(self):
        tracker = MusicTracker()
        tracker.update(observation(1), 0.)
        self.assertEqual(tracker.update(observation(2), .1)['stable_chord']['label'], 'C')
        first = dict(observation(3, (45,48,52)), visual_transition=True)
        self.assertEqual(tracker.update(first, .2)['stable_chord']['label'], 'C')
        second = dict(observation(4, (45,48,52)), visual_transition=True)
        result = tracker.update(second, .3)
        self.assertEqual(result['stable_chord']['label'], 'Am')
        self.assertEqual(result['tracking']['required_observations'], 2)

    def test_current_visual_change_shortens_only_reliable_harmonic_switch(self):
        for visual,expected in ((True,'D'),(False,'C')):
            tracker = MusicTracker()
            tracker.update(observation(1),0.)
            tracker.update(observation(2),.1)
            for sequence,now in ((3,.2),(4,.29)):
                frame = observation(sequence,(50,54,57))
                frame['visual_transition'] = visual
                result = tracker.update(frame,now)
            self.assertEqual(result['stable_chord']['label'],expected)

    def test_visual_motion_does_not_name_a_single_audio_frame(self):
        tracker = MusicTracker()
        tracker.update(observation(1),0.)
        tracker.update(observation(2),.1)
        frame = observation(3,(50,54,57))
        frame['visual_transition'] = True
        self.assertEqual(tracker.update(frame,.2)['stable_chord']['label'],'C')

    def test_two_observations_and_transition(self):
        tracker = MusicTracker()
        first = tracker.update(observation(1), 0)
        self.assertEqual(first["raw_chord"]["label"], "C")
        self.assertEqual(first["tracking"]["chord_status"], "pending")
        self.assertEqual(tracker.update(observation(2), .1)["stable_chord"]["label"], "C")
        pending = tracker.update(observation(3, (45, 48, 52)), .2)
        self.assertEqual(pending["stable_chord"]["label"], "C")
        self.assertEqual(pending["tracking"]["chord_status"], "switching")
        self.assertEqual(tracker.update(observation(4, (45, 48, 52)), .3)["stable_chord"]["label"], "C")
        self.assertEqual(tracker.update(observation(5, (45, 48, 52)), .45)["stable_chord"]["label"], "Am")

    def test_single_frame_glitch_never_confirms(self):
        tracker = MusicTracker()
        for sequence, midis in enumerate(((48, 52, 55), (48, 52, 55), (45, 48, 52), (48, 52, 55))):
            result = tracker.update(observation(sequence, midis), sequence * .1)
            self.assertNotEqual(result["stable_chord"].get("label"), "Am")

    def test_silence_errors_and_inactive_unknown_clear_chord(self):
        for status, midis in (("silence", ()), ("unknown", ()), ("error", ())):
            tracker = MusicTracker()
            tracker.update(observation(1), 0)
            tracker.update(observation(2), .1)
            result = tracker.update(observation(3, midis, status), .2)
            self.assertIsNone(result["stable_chord"]["label"])

    def test_alternating_sustain_candidates_keep_confirmed_chord(self):
        tracker = MusicTracker()
        tracker.update(observation(1), 0)
        tracker.update(observation(2), .1)
        for sequence, midis in enumerate(((45, 48, 52), (48, 52, 55),
                                          (48, 55), (48, 52, 55)) * 3, 3):
            result = tracker.update(observation(sequence, midis), sequence * .1)
            self.assertEqual(result["stable_chord"]["label"], "C")

    def test_active_unknown_is_bounded_hold_and_never_refreshed_by_noise(self):
        tracker = MusicTracker()
        tracker.update(observation(1), 0)
        tracker.update(observation(2), .1)
        for sequence, now in ((3, .2), (4, .5)):
            frame = observation(sequence, (), "unknown")
            frame["signal_active"] = True
            result = tracker.update(frame, now)
            self.assertEqual(result["tracking"]["chord_status"], "holding")
            self.assertEqual(result["stable_chord"]["label"], "C")
        frame["sequence"] = 5
        self.assertIsNone(tracker.update(frame, 1)["stable_chord"]["label"])

    def test_quieter_missing_third_is_not_a_new_power_chord(self):
        tracker = MusicTracker()
        for sequence, now in ((1, 0), (2, .1)):
            frame = observation(sequence)
            frame["spectrum"] = {"rms": .1}
            tracker.update(frame, now)
        for sequence, now in ((3, .2), (4, .4), (5, .7), (6, 1.2)):
            frame = observation(sequence, (48, 55))
            frame["spectrum"] = {"rms": .04}
            result = tracker.update(frame, now)
            self.assertEqual(result["stable_chord"]["label"], "C")
            self.assertEqual(result["tracking"]["chord_status"], "holding")
        frame["sequence"] = 7
        self.assertIsNone(tracker.update(frame, 2.2)["stable_chord"]["label"])

    def test_restrummed_power_chord_and_inversion_can_replace_previous(self):
        for midis, label, times in (((48, 55), "C5", (.2, .35, .5)),
                                    ((52, 55, 60), "C/E", (.2, .4, .6, .8))):
            tracker = MusicTracker()
            for sequence, now in ((1, 0), (2, .1)):
                frame = observation(sequence)
                frame["spectrum"] = {"rms": .02}
                tracker.update(frame, now)
            for sequence, now in enumerate(times, 3):
                frame = observation(sequence, midis)
                frame["spectrum"] = {"rms": .1}
                result = tracker.update(frame, now)
            self.assertEqual(result["stable_chord"]["label"], label)

    def test_slow_decay_uses_envelope_peak_instead_of_last_frame(self):
        tracker = MusicTracker()
        for sequence, now, rms in ((1, 0, .1), (2, .1, .1),
                                   (3, .3, .08), (4, .5, .06), (5, .7, .04)):
            frame = observation(sequence)
            frame["spectrum"] = {"rms": rms}
            tracker.update(frame, now)
        for sequence, now in ((6, .8), (7, 1), (8, 1.2)):
            frame = observation(sequence, (48, 55))
            frame["spectrum"] = {"rms": .035}
            result = tracker.update(frame, now)
            self.assertEqual(result["stable_chord"]["label"], "C")
            self.assertEqual(result["tracking"]["chord_status"], "holding")

    def test_duplicate_mismatch_and_large_gap_cannot_vote_or_hold(self):
        tracker = MusicTracker()
        tracker.update(observation(1), 0)
        tracker.update(observation(2), .1)
        tracker.update(observation(3, (45, 48, 52)), .2)
        for now in (.3, .4, .5):
            result = tracker.update(observation(3, (45, 48, 52)), now)
            self.assertEqual(result["stable_chord"]["label"], "C")
            self.assertEqual(result["tracking"]["confirmations"], 1)
        result = tracker.update(observation(4, (45, 48, 52)), 3)
        self.assertIsNone(result["stable_chord"]["label"])
        self.assertEqual(result["tracking"]["confirmations"], 1)

    def test_new_quiet_attack_can_replace_a_decaying_subset(self):
        tracker = MusicTracker()
        for sequence, now in ((1, 0), (2, .1)):
            frame = observation(sequence)
            frame["spectrum"] = {"rms": .1}
            tracker.update(frame, now)
        for sequence, now, rms in ((3, .2, .03), (4, .3, .068),
                                   (5, .45, .06), (6, .6, .055)):
            frame = observation(sequence, (48, 55))
            frame["spectrum"] = {"rms": rms}
            result = tracker.update(frame, now)
        self.assertEqual(result["stable_chord"]["label"], "C5")

    def test_weak_repeated_candidate_cannot_confirm_or_switch(self):
        tracker = MusicTracker()
        for sequence in range(4):
            result = tracker.update(observation(sequence, confidence=.2), sequence * .1)
        self.assertIsNone(result["stable_chord"]["label"])

    def test_slow_polling_does_not_lower_switch_requirement_on_hold_expiry(self):
        tracker = MusicTracker()
        tracker.update(observation(1), 0)
        tracker.update(observation(2), .1)
        first = tracker.update(observation(3, (45, 48, 52)), 1)
        self.assertIsNone(first["stable_chord"]["label"])
        second = tracker.update(observation(4, (45, 48, 52)), 1.3)
        self.assertIsNone(second["stable_chord"]["label"])
        self.assertEqual(second["tracking"]["required_observations"], 3)
        self.assertEqual(tracker.update(observation(5, (45, 48, 52)), 1.6)["stable_chord"]["label"], "Am")

    def test_duplicate_polling_cannot_bridge_a_gap_between_fresh_frames(self):
        tracker = MusicTracker()
        tracker.update(observation(1), 0)
        tracker.update(observation(1), 1)
        tracker.update(observation(1), 1.9)
        result = tracker.update(observation(2), 2.1)
        self.assertIsNone(result["stable_chord"]["label"])
        self.assertEqual(result["tracking"]["confirmations"], 1)

    def test_repeated_sequence_and_wrap(self):
        tracker = MusicTracker()
        tracker.update(observation(0xffffffff), 0)
        repeated = tracker.update(observation(0xffffffff), .1)
        self.assertEqual(repeated["tracking"]["confirmations"], 1)
        wrapped = tracker.update(observation(0), .2)
        self.assertEqual(wrapped["stable_chord"]["label"], "C")
        self.assertEqual(wrapped["scale_evidence"]["event_count"], 1)
        self.assertIsNone(tracker.update(observation(0), 3)["stable_chord"]["label"])

    def test_context_clock_and_large_gap_reset_confirmation(self):
        for now, context in ((.2, "changed"), (-1, "live"), (3, "live")):
            tracker = MusicTracker()
            tracker.update(observation(1), 0, "live")
            result = tracker.update(observation(2), now, context)
            self.assertEqual(result["tracking"]["confirmations"], 1)
            self.assertIsNone(result["stable_chord"]["label"])

    def test_melody_preserves_all_relative_modes(self):
        tracker = MusicTracker()
        for sequence, midi in enumerate((60, 62, 64, 65, 67, 69, 71)):
            result = tracker.update(observation(sequence, (midi,)), float(sequence))
        scale = result["scale_evidence"]
        self.assertEqual(scale["status"], "ambiguous")
        self.assertEqual(len(scale["groups"]), 1)
        self.assertEqual(len(scale["candidates"]), 7)
        labels = {(item["tonic"], item["scale"]) for item in scale["candidates"]}
        self.assertIn(("C", "major"), labels)
        self.assertIn(("A", "natural_minor"), labels)
        self.assertIn(("D", "dorian"), labels)

    def test_octaves_held_notes_and_weak_outlier(self):
        tracker = MusicTracker()
        tracker.update(observation(1, (48, 60, 52, 55)), 0)
        for sequence in range(2, 8):
            result = tracker.update(observation(sequence, (48, 60, 52, 55)), sequence * .1)
        self.assertEqual(result["scale_evidence"]["event_count"], 1)
        weak = observation(8, (61,), confidence=.2)
        result = tracker.update(weak, 1)
        self.assertNotIn("C#", result["scale_evidence"]["observed_pitch_classes"])

    def test_history_expires_and_silence_does_not_add_events(self):
        tracker = MusicTracker()
        tracker.update(observation(1), 0)
        self.assertTrue(tracker.update(observation(2, (), "silence"), 1)["scale_evidence"]["candidates"])
        result = tracker.update(observation(3, (), "silence"), 13)
        self.assertEqual(result["scale_evidence"]["event_count"], 0)
        self.assertEqual(result["scale_evidence"]["status"], "insufficient")


if __name__ == "__main__":
    unittest.main()

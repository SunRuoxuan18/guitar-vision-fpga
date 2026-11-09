"""End-to-end sustain tracking checks using synthesized FFT power spectra."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_music import MusicRecognizer
from music_tracking import MusicTracker

from test_audio_music import synthesize


class SustainTrackingTests(unittest.TestCase):
    def setUp(self):
        self.recognizer = MusicRecognizer()

    def observe(self, tracker, sequence, now, notes, *, strengths, decay=1.1,
                noise=0.0003):
        raw = self.recognizer.analyze(
            synthesize(notes, strengths=strengths, decay=decay, noise=noise,
                       seed=1000 + sequence),
            sequence=sequence,
        )
        self.assertGreater(raw["spectrum"]["rms"], self.recognizer.config.minimum_rms, raw)
        return tracker.update(raw, now)

    def test_decaying_triad_holds_only_until_sustain_limit(self):
        tracker = MusicTracker()
        frames = []

        # The complete C triad confirms first; its unevenly decaying E falls below
        # recognition, causing the real recognizer to emit C5 rather than a test label.
        for sequence, now in ((1, 0.0), (2, 0.1)):
            frames.append(self.observe(tracker, sequence, now, (48, 52, 55),
                                       strengths=(1.0, 0.8, 0.8)))
        self.assertEqual(frames[-1]["raw_chord"]["label"], "C")
        self.assertEqual(frames[-1]["stable_chord"]["label"], "C")

        noisy = self.observe(tracker, 3, 0.2, (), strengths=(), noise=0.15)
        self.assertEqual(noisy["status"], "unknown")
        self.assertTrue(noisy["signal_active"])
        self.assertEqual(noisy["stable_chord"]["label"], "C")
        self.assertEqual(noisy["tracking"]["chord_status"], "holding")

        for sequence, now in ((4, 0.4), (5, 1.0), (6, 1.85)):
            frame = self.observe(tracker, sequence, now, (48, 52, 55),
                                 strengths=(1.0, 0.06, 0.55), decay=1.7)
            self.assertEqual(frame["raw_chord"]["label"], "C5")
            self.assertEqual(frame["stable_chord"]["label"], "C")
            self.assertEqual(frame["tracking"]["chord_status"], "holding")
            frames.append(frame)

        expired = self.observe(tracker, 7, 2.0, (48, 52, 55),
                               strengths=(1.0, 0.06, 0.55), decay=1.7)
        self.assertEqual(expired["raw_chord"]["label"], "C5")
        self.assertIsNone(expired["stable_chord"]["label"])
        self.assertNotEqual(expired["tracking"]["chord_status"], "holding")

    def test_real_chord_change_converges_and_silence_clears(self):
        tracker = MusicTracker()
        for sequence, now in ((1, 0.0), (2, 0.1)):
            result = self.observe(tracker, sequence, now, (48, 52, 55),
                                  strengths=(1.0, 0.8, 0.8))
        self.assertEqual(result["stable_chord"]["label"], "C")

        for sequence, now in ((3, 0.2), (4, 0.5), (5, 0.75)):
            result = self.observe(tracker, sequence, now, (45, 48, 52),
                                  strengths=(1.0, 0.75, 0.6), decay=0.8)
            self.assertEqual(result["raw_chord"]["label"], "Am")
            if sequence < 5:
                self.assertEqual(result["stable_chord"]["label"], "C")
                self.assertEqual(result["tracking"]["chord_status"], "switching")
        self.assertEqual(result["stable_chord"]["label"], "Am")
        self.assertEqual(result["tracking"]["chord_status"], "confirmed")

        silence = self.recognizer.analyze(synthesize([], noise=0.0, seed=2000), sequence=6)
        self.assertEqual(silence["status"], "silence")
        cleared = tracker.update(silence, 0.8)
        self.assertIsNone(cleared["stable_chord"]["label"])
        self.assertEqual(cleared["tracking"]["chord_status"], "none")


if __name__ == "__main__":
    unittest.main()

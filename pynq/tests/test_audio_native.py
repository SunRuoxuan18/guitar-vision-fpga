"""Run on ARM with FRETLENS_AUDIO_NATIVE=1 after compiling libaudio_fit.so."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from audio_music import MusicRecognizer
from test_audio_music import synthesize


@unittest.skipUnless(os.environ.get("FRETLENS_AUDIO_NATIVE") == "1", "ARM native library is opt-in")
class NativeEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accelerated = MusicRecognizer()
        with patch.dict(os.environ, {"FRETLENS_AUDIO_NATIVE": "0"}):
            cls.reference = MusicRecognizer()

    def test_chord_notes_and_confidence_match_python(self):
        cases = (([69], {"harmonics": 1}), ([60, 64, 67], {}), ([57, 60, 64], {}),
                 ([40, 47, 52, 56, 59, 64], {}), ([48, 52, 55, 59], {"noise": .002}),
                 ([45], {"fundamental": 0, "decay": .7}),
                 ([50, 57, 60, 65], {"cents": 17, "strengths": [1, .7, .5, .4]}),
                 ([], {"noise": .15}))
        for notes, options in cases:
            with self.subTest(notes=notes):
                power = synthesize(notes, **options)
                expected = self.reference.analyze(power, 1)
                actual = self.accelerated.analyze(power, 1)
                self.assertEqual(actual["status"], expected["status"])
                self.assertEqual(actual["chord"]["label"], expected["chord"]["label"])
                self.assertEqual([n["midi_note"] for n in actual["notes"]],
                                 [n["midi_note"] for n in expected["notes"]])
                self.assertAlmostEqual(actual["confidence"], expected["confidence"], places=3)

    def test_nnls_matches_python_for_overlapping_designs(self):
        rng = np.random.default_rng(340)
        for count in (1, 3, 6, 12):
            design = rng.random((count, 2048)).astype(np.float32)
            design /= np.linalg.norm(design, axis=1)[:, None]
            for dtype in (np.float32, np.float64):
                spectrum = rng.random(2048).astype(dtype)
                expected_weights, expected_residual = self.reference._fit(design, spectrum)
                weights, residual = self.accelerated._fit(design, spectrum)
                np.testing.assert_allclose(weights, expected_weights, rtol=1e-4, atol=2e-5)
                np.testing.assert_allclose(residual, expected_residual, rtol=1e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()

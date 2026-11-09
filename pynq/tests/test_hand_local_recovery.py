"""Local recovery must reject ambiguous/repeated and occluded observations."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hand_tracking import recover_local_joint


class LocalRecoveryTests(unittest.TestCase):
    def test_independently_moved_tip_recovers_without_relaxing_lk_gates(self):
        rng = np.random.RandomState(7)
        patch = rng.randint(0,256,(17,17)).astype(np.uint8)
        before = np.zeros((80,100),np.uint8)
        after = before.copy()
        before[32:49,42:59] = patch
        after[32:49,46:63] = patch
        result = recover_local_joint(before,after,np.array([50.,40.]),np.array([50.,40.]))
        np.testing.assert_allclose(result,[54,40],atol=.01)

    def test_occlusion_and_low_texture_are_rejected(self):
        rng = np.random.RandomState(8)
        before = rng.randint(0,256,(80,100)).astype(np.uint8)
        after = np.zeros_like(before)
        self.assertIsNone(recover_local_joint(before,after,np.array([50.,40.]),np.array([50.,40.])))
        self.assertIsNone(recover_local_joint(after,after,np.array([50.,40.]),np.array([50.,40.])))

    def test_repeated_pattern_is_not_unique_evidence(self):
        before = np.tile(np.array([0,255],np.uint8),(80,50))
        self.assertIsNone(recover_local_joint(before,before,np.array([50.,40.]),np.array([50.,40.])))


if __name__ == '__main__':
    unittest.main()

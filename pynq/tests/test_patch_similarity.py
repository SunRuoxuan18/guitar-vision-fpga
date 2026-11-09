import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patch_similarity import patch_correlations


class PatchTests(unittest.TestCase):
    def test_matches_independent_opencv_for_both_template_modes(self):
        rng = np.random.RandomState(627)
        for pixels in (11,15,16):
            a = rng.randint(0,256,(32,pixels,pixels)).astype(np.uint8)
            b = a.copy()
            b[1] = 255-a[1]
            b[2] = 80
            a[3] = 80
            b[4] = rng.randint(0,256,(pixels,pixels))
            expected = np.array([cv2.matchTemplate(x,y,cv2.TM_CCOEFF_NORMED)[0,0]
                                 for x,y in zip(a,b)])
            np.testing.assert_allclose(patch_correlations(a,b,flat_template=True),expected,atol=2e-5)
            expected[2] = 0
            np.testing.assert_allclose(patch_correlations(a,b,minimum_energy=100.),expected,atol=2e-5)

    def test_energy_threshold_and_noncontiguous_inputs(self):
        a = np.ones((3,11,22),np.uint8)
        a[:,0,0] = 2
        self.assertFalse(patch_correlations(a[:,:,::2],a[:,:,::2],minimum_energy=100.).any())
        np.testing.assert_allclose(patch_correlations(a[:,:,::2],a[:,:,::2]),1,atol=1e-6)

    def test_invalid_buffers_rejected_before_native_call(self):
        for shape in ((0,11,11),(241,11,11),(1,17,17)):
            with self.assertRaises(ValueError):
                patch_correlations(np.zeros(shape,np.uint8),np.zeros(shape,np.uint8))
        for minimum in (-1,np.nan,np.inf):
            with self.assertRaises(ValueError):
                patch_correlations(np.zeros((1,11,11),np.uint8),np.zeros((1,11,11),np.uint8),minimum)


if __name__ == '__main__':
    unittest.main()

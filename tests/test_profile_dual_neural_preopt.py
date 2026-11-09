"""Synthetic grouped MAC and missing-shape regressions."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from profile_dual_neural_preopt import ROOT, conv_profile, fixed_shape, helper, np, resources


class ProfileTests(unittest.TestCase):
    def test_grouped_and_depthwise_macs(self):
        for groups, outputs, expected in [(1, 8, 7200), (2, 8, 3600), (4, 4, 900)]:
            with self.subTest(groups=groups):
                node = helper.make_node("Conv", ["x", "w"], ["y"], name="test", group=groups)
                shapes = {"x": [1, 4, 7, 7], "y": [1, outputs, 5, 5]}
                row = conv_profile(node, shapes, {"w": np.zeros((outputs, 4 // groups, 3, 3), np.float32)})
                self.assertEqual(row["macs"], expected)
                self.assertEqual(row["depthwise"], groups == 4)
                self.assertEqual(row["production_conv_eligible"], groups != 2)

    def test_missing_or_symbolic_shape_rejected(self):
        for shapes in ({}, {"x": [1, None, 3, 3]}, {"x": [0, 3]}):
            with self.assertRaisesRegex(ValueError, "Missing fixed"):
                fixed_shape(shapes, "x")

    def test_spatial_cap_and_invalid_groups(self):
        node = helper.make_node("Conv", ["x", "w"], ["y"], name="large")
        shapes = {"x": [1, 3, 320, 320], "y": [1, 16, 320, 320]}
        row = conv_profile(node, shapes, {"w": np.zeros((16, 3, 1, 1), np.float32)})
        self.assertFalse(row["production_conv_eligible"])
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            conv_profile(node, shapes, {"w": np.zeros((16, 2, 1, 1), np.float32)})

    def test_conv_missing_output_shape_rejected(self):
        node = helper.make_node("Conv", ["x", "w"], ["y"])
        with self.assertRaisesRegex(ValueError, "Missing fixed positive shape: y"):
            conv_profile(node, {"x": [1, 3, 8, 8]}, {"w": np.zeros((4, 3, 1, 1), np.float32)})

    def test_report_bram_units_and_replacement_budget(self):
        result = resources(ROOT / "output/hand_first_fold_20261026/hand_neural_utilization.rpt")
        self.assertEqual(result["other_fixed"], {"lut": 26504, "dsp": 71, "bram36_equivalent": 76})
        self.assertEqual(result["additional_second_core_budget_keep_existing"]["dsp"], 8)
        self.assertIsNone(result["new_core_resources"])


if __name__ == "__main__":
    unittest.main()

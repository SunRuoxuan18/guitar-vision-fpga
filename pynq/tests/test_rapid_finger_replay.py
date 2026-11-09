"""Regression coverage for the reproducible simulated rapid-finger replay."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "diagnostics"))
from rapid_finger_replay import _motion_events, run_replay


class RapidFingerReplayTests(unittest.TestCase):
    def test_accumulated_motion_is_one_phase_until_the_kind_clears(self):
        outputs = [
            {"sequence": 1, "timestamp_ms": 33, "result": {"motion": {"finger_change": True}}},
            {"sequence": 2, "timestamp_ms": 66, "result": {"motion": {"finger_change": True, "hand_shift": True}}},
            {"sequence": 3, "timestamp_ms": 99, "result": {"motion": {"finger_change": False, "hand_shift": True}}},
            {"sequence": 4, "timestamp_ms": 132, "result": {"motion": {"finger_change": True, "hand_shift": False}}},
        ]

        self.assertEqual(_motion_events(outputs), [
            {"sequence": 1, "timestamp_ms": 33, "kind": "finger_change"},
            {"sequence": 2, "timestamp_ms": 66, "kind": "hand_shift"},
            {"sequence": 4, "timestamp_ms": 132, "kind": "finger_change"},
        ])

    def test_known_articulation_and_hand_shift_are_separate_and_timely(self):
        report = run_replay()

        self.assertTrue(report["passed"])
        self.assertTrue(report["simulated_tracking"])
        self.assertFalse(report["model_accuracy_claimed"])
        self.assertEqual(report["frame_timestamps_ms"], [0, 33, 66, 99, 132])
        self.assertEqual([(event["sequence"], event["kind"]) for event in report["detected_changes"]],
                         [(1, "finger_change"), (2, "hand_shift")])
        self.assertEqual(report["maximum_detection_lag_ms"], 0)
        self.assertEqual(report["missed_changes"], [])
        self.assertEqual(report["false_changes"], [])

    def test_occlusion_does_not_create_a_new_motion_phase_and_old_result_is_caught_up(self):
        report = run_replay()

        self.assertIn(report["occlusion"]["request_reason"],
                      ("insufficient_fingertips", "reacquire", "finger_motion"))
        self.assertFalse(report["occlusion"]["novel_motion_event"])
        self.assertEqual(report["false_changes"], [])
        self.assertEqual(report["outdated_neural_result"]["detected_sequence"], 1)
        self.assertEqual(report["outdated_neural_result"]["output_sequence"], 4)
        self.assertEqual(report["outdated_neural_result"]["tracked_sequence"], 4)
        self.assertEqual(report["outdated_neural_result"]["keypoint_source"], "optical_flow")


if __name__ == "__main__":
    unittest.main()

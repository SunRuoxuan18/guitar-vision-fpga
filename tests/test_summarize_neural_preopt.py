import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("summary", Path(__file__).resolve().parents[1] / "scripts/summarize_neural_preopt.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def test_resource_units_and_incomplete_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "util.rpt"
            path.write_text("| Slice LUTs* | 15150 | 0 |\n| DSPs | 82 | 0 |\n| Block RAM Tile | 39.5 | 0 |\n", encoding="utf-8")
            self.assertEqual(summary.ooc_resources(path), {"lut": 15150, "dsp": 82, "bram36_equivalent": 39.5})
            path.write_text("| Slice LUTs* | 15150 |\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                summary.ooc_resources(path)

    def test_one_core_and_two_core_budget_are_distinct(self):
        fixed = {"lut": 26504, "dsp": 71, "bram36_equivalent": 76}
        candidate = {"lut": 15150, "dsp": 82, "bram36_equivalent": 39}
        capacity = {"lut": 53200, "dsp": 220, "bram36_equivalent": 140}
        shared = summary.resource_budget(fixed, candidate, capacity)
        self.assertTrue(shared["arithmetic_within_capacity"])
        self.assertEqual(shared["remaining"]["dsp"], 67)
        duplicate = summary.resource_budget(fixed, candidate, capacity, copies=2)
        self.assertFalse(duplicate["arithmetic_within_capacity"])
        self.assertEqual(duplicate["remaining"]["dsp"], -15)
        with self.assertRaises(ValueError):
            summary.resource_budget(fixed, candidate, capacity, copies=0)


if __name__ == "__main__":
    unittest.main()

import csv
import unittest
from pathlib import Path


RESULTS = Path(__file__).resolve().parents[1] / "results"


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class PublishedResultTests(unittest.TestCase):
    def test_dolly_rows_match_submitted_tables(self):
        data = {
            (row["run"], row["checkpoint"]): row
            for row in rows(RESULTS / "2026-10-06/main_results.csv")
        }
        expected = {
            ("run_00", "T1"): 0.492375,
            ("run_00", "T2"): 0.58575,
            ("run_00", "T"): 0.590625,
            ("run_00", "T0"): 0.00125,
            ("run_01", "T1"): 0.506,
            ("run_01", "T2"): 0.5855,
            ("run_01", "T"): 0.602,
            ("run_01", "T0"): -0.00825,
            ("run_02", "T1"): 0.510875,
            ("run_02", "T2"): 0.601,
            ("run_02", "T"): 0.616,
            ("run_02", "T0"): -0.0035,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertAlmostEqual(float(data[key]["signed_gap"]), value)
                self.assertEqual(int(data[key]["audit_pairs"]), 8000)
        for run in ("run_00", "run_01", "run_02"):
            self.assertGreater(float(data[(run, "T")]["signed_gap"]), 0.0304)
            self.assertLess(float(data[(run, "T0")]["signed_gap"]), 0.0304)

    def test_ultrachat_400k_rows_match_submitted_table(self):
        selected = {
            (row["run_id"], row["stage"]): row
            for row in rows(RESULTS / "2026-10-06-distinct/curve.csv")
            if row["steps"] == "12500"
        }
        expected = {
            ("run_00", "target"): 0.312375,
            ("run_00", "control"): 0.00025,
            ("run_01", "target"): 0.315,
            ("run_01", "control"): -0.00775,
            ("run_02", "target"): 0.32275,
            ("run_02", "control"): -0.004125,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                row = selected[key]
                self.assertAlmostEqual(float(row["watermark_signed_gap"]), value)
                self.assertEqual(int(row["examples_seen"]), 400000)
                self.assertEqual(int(row["unique_examples_seen"]), 400000)
                self.assertEqual(int(row["audit_pairs"]), 8000)
        for run in ("run_00", "run_01", "run_02"):
            self.assertGreater(float(selected[(run, "target")]["watermark_signed_gap"]), 0.0304)
            self.assertLess(float(selected[(run, "control")]["watermark_signed_gap"]), 0.0304)


if __name__ == "__main__":
    unittest.main()

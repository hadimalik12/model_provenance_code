import math
import unittest

from experiments.watermark_inheritance.calibration_analysis import (
    clopper_pearson,
    evaluate_decisions,
    holm_rejections,
    paired_cutoff,
)


class CalibrationAnalysisTests(unittest.TestCase):
    def test_manuscript_cutoff(self):
        self.assertAlmostEqual(paired_cutoff(0.05, 8000), 0.030368073095415258)
        self.assertGreater(paired_cutoff(0.01, 8000), paired_cutoff(0.05, 8000))
        self.assertGreater(paired_cutoff(0.05, 250), paired_cutoff(0.05, 8000))

    def test_primary_decision_uses_absolute_gap_and_strict_cutoff(self):
        result = evaluate_decisions([False] * 10, [True] * 10, 0.05)
        self.assertEqual(result["signed_gap"], -1)
        self.assertTrue(result["primary_false_alarm"])
        cutoff = paired_cutoff(0.05, 100)
        positives = round(cutoff * 100)
        result = evaluate_decisions([True] * positives + [False] * (100 - positives), [False] * 100, 0.05)
        self.assertEqual(result["primary_false_alarm"], result["advantage"] > cutoff)

    def test_holm_stops_at_first_nonrejection(self):
        self.assertEqual(holm_rejections([0.001, 0.02, 0.03], 0.05), [True, True, True])
        self.assertEqual(holm_rejections([0.001, 0.03, 0.031], 0.05), [True, False, False])

    def test_exact_zero_event_upper_bound(self):
        interval = clopper_pearson(0, 15)
        self.assertEqual(interval["two_sided_lower"], 0)
        self.assertAlmostEqual(interval["one_sided_upper"], 1 - 0.05 ** (1 / 15))


if __name__ == "__main__":
    unittest.main()

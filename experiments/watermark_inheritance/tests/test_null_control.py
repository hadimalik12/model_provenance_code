import math
import unittest

from experiments.watermark_inheritance.generate import sampling_seed
from experiments.watermark_inheritance.null_control import double_prime_seed, null_evaluation


class NullControlTests(unittest.TestCase):
    def test_double_prime_seed_is_deterministic_and_fresh(self):
        config = {"seed": 1, "shard_seed": 7}
        value = double_prime_seed(config, "pair_00001")
        self.assertEqual(value, double_prime_seed(config, "pair_00001"))
        self.assertNotEqual(value, sampling_seed(7, "full", "pair_00001", 0))
        self.assertNotEqual(value, sampling_seed(7, "full", "pair_00001", 1))

    def test_equal_control_decisions_have_zero_advantage(self):
        original, double = [], []
        for index, value in enumerate([1.0, 0.0, 1.0, 0.0]):
            split = "calibration" if index == 0 else "audit"
            original.append({
                "pair_id": str(index), "audit_split": split,
                "s_prime": {"z": value},
            })
            double.append({"pair_id": str(index), "audit_split": split, "z": value})
        result = null_evaluation(original, double, threshold=0.5)
        self.assertEqual(result["advantage"], 0.0)
        self.assertEqual(result["p_shard"], 1.0)
        self.assertTrue(result["advantage_below_cutoff"])
        self.assertFalse(result["two_sided_null_rejected"])
        self.assertAlmostEqual(
            result["single_audit_advantage_cutoff"],
            math.sqrt(2 * math.log(2 / 0.05) / 3),
        )


if __name__ == "__main__":
    unittest.main()

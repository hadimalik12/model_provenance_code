import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.watermark_inheritance.common import load_json
from experiments.watermark_inheritance.configure import RUNS, configure


class ConfigureTests(unittest.TestCase):
    def test_creates_the_three_paper_runs_and_refuses_drift(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "experiments.watermark_inheritance.configure.repository_revision",
            return_value="revision",
        ):
            root = Path(tmp) / "artifacts"
            shared = Path(tmp) / "cache"
            paths = configure(root, shared)
            self.assertEqual(len(paths), 3)
            for path, (run_id, seed, key, wrong_key) in zip(paths, RUNS):
                config = load_json(path)
                self.assertEqual(config["run_id"], run_id)
                self.assertEqual(config["seed"], seed)
                self.assertEqual(config["watermark_key"], key)
                self.assertEqual(config["wrong_key"], wrong_key)
                self.assertEqual(config["pairs"], 10000)
                self.assertEqual(config["calibration_pairs"], 2000)
            changed = load_json(paths[0])
            changed["pairs"] = 1
            paths[0].write_text(json.dumps(changed))
            with self.assertRaisesRegex(RuntimeError, "Refusing"):
                configure(root, shared)


if __name__ == "__main__":
    unittest.main()

"""Regression checks for source integrity, overlap matching and the submitted DAG."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.table_08_controlled_provenance.shared.filter_fragments import fragments, matched_patterns, normalize
from experiments.table_08_controlled_provenance.shared.local_source import sha256_file, validate
from experiments.table_08_controlled_provenance.shared.prepare import choose


class FilterAndJobsTests(unittest.TestCase):
    def test_window_duplicates_are_removed_before_keyed_assignment(self):
        rows = [{"prompt": f"Describe function number {i} clearly",
                 "response": f"def function_{i}(value): return value + {i}"} for i in range(8)]
        def transform(row):
            row["text_hash"] = str(int(row["response"].split("+")[-1]) // 2)
            return row
        selected, control, counts = choose(rows, b"x" * 32,
            {"max_prompt_chars": 300, "max_response_chars": 1100}, 2, transform=transform)
        self.assertEqual(counts["duplicate_windows_removed"], 4)
        self.assertEqual(len({r["text_hash"] for r in selected + control}), 4)

    def test_overlapping_patterns_and_changed_code_are_detected(self):
        self.assertEqual(matched_patterns(["abc", "bc"], " A B C "), {0, 1})
        self.assertEqual(normalize("ＦＯＯ\n Bar"), "foobar")
        common = "abcdefghijklmnopqrstuvwxyz0123456789" * 5
        blocks = sorted(fragments("different_prefix" + common + "different_suffix"))
        self.assertTrue(matched_patterns(blocks, "another_prefix" + common + "another_suffix"))
        self.assertFalse(matched_patterns(blocks, "entirely unrelated text"))

    def test_changed_filtered_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.jsonl"
            data.write_text('{"prompt":"fixture"}\n')
            source = {"local_jsonl": str(data), "filter_manifest": str(root / "manifest.json"),
                      "local_sha256": sha256_file(data), "dataset": "fixture", "revision": "fixture",
                      "subset": "python", "split": "train", "n_per_arm": 2,
                      "max_prompt_chars": 300, "max_response_chars": 1100}
            manifest = {k: source[k] for k in ("dataset", "revision", "subset", "split", "local_sha256")}
            manifest.update(limits={"max_prompt_chars": 300, "max_response_chars": 1100},
                            retained_matches_under_this_rule=0, reference_documents=416000, retained_pairs=4,
                            eligible_sha256=source["local_sha256"], retained_detected_matches=0,
                            reference_scope="416000 documents (synthetic test metadata)")
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(validate(source)[0], data)
            data.write_text('{"prompt":"changed"}\n')
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate(source)

    def test_submission_dependencies_without_contacting_slurm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_sbatch = root / "sbatch"
            fake_sbatch.write_text(f"#!{sys.executable}\n" +
                "import json,os,sys\nfrom pathlib import Path\n"
                "p=Path(os.environ['TABLE8_TEST_LOG'])\n"
                "rows=p.read_text().splitlines() if p.exists() else []\n"
                "with p.open('a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                "print(1001+len(rows))\n")
            fake_sbatch.chmod(0o700)
            fake_python = root / "preflight"
            fake_python.write_text("#!/bin/sh\nexit 0\n")
            fake_python.chmod(0o700)
            log = root / "calls.jsonl"
            env = {**os.environ, "PATH": str(root) + ":" + os.environ["PATH"],
                   "TABLE8_TEST_LOG": str(log), "TABLE8_PYTHON": str(fake_python)}
            env.pop("TABLE8_CONFIG", None)
            subprocess.run(["bash", "experiments/table_08_controlled_provenance/shared/jobs/submit_all.sh"],
                           env=env, check=True, capture_output=True, text=True)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 23)
            jobs = {}
            for number, args in enumerate(calls, 1001):
                export = next(a.split("=", 1)[1] for a in args if a.startswith("--export="))
                values = dict(part.split("=", 1) for part in export.split(",") if "=" in part)
                key = tuple(values.get("TABLE8_" + field, "") for field in ("TASK", "FAMILY", "STAGE", "PROFILE"))
                dependency = next((a.split("=", 1)[1] for a in args if a.startswith("--dependency=")), "")
                self.assertIn("--kill-on-invalid-dep=yes", args)
                jobs[key] = (str(number), dependency)
            source = jobs[("source", "", "", "")][0]
            for family, profiles in {"1b": ["hh_sft"], "1_4b": ["helpful_sft", "tulu_sft"]}.items():
                parent, parent_dep = jobs[("train", family, "parent", "")]
                self.assertEqual(parent_dep, "afterok:" + source)
                self.assertEqual(jobs[("audit", family, "base", "")][1], "afterok:" + source)
                self.assertEqual(jobs[("audit", family, "parent", "")][1], "afterok:" + parent)
                for profile in profiles:
                    data, _ = jobs[("downstream", "", "", profile)]
                    target, dep = jobs[("train", family, "target", profile)]
                    self.assertEqual(dep, f"afterok:{parent}:{data}")
                    control, dep = jobs[("train", family, "control", profile)]
                    self.assertEqual(dep, "afterok:" + data)
                    self.assertEqual(jobs[("audit", family, "target", profile)][1], "afterok:" + target)
                    self.assertEqual(jobs[("audit", family, "control", profile)][1], "afterok:" + control)
            audit_ids = {value[0] for key, value in jobs.items() if key[0] == "audit"}
            final_dep = jobs[("compile", "", "", "")][1].split(":")
            self.assertEqual(final_dep[0], "afterok")
            self.assertEqual(set(final_dep[1:]), audit_ids)


if __name__ == "__main__":
    unittest.main()

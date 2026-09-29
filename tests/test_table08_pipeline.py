"""CPU end-to-end check of Table 8 with a randomly initialized tiny model."""
from __future__ import annotations

import json
import importlib
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import yaml

from experiments.table_08_controlled_provenance.shared.common import fingerprint, read_jsonl, write_json, write_jsonl
from experiments.table_08_controlled_provenance.shared.local_source import sha256_file


class TinyPipeline(unittest.TestCase):
    def test_parent_target_control_and_audit(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

        with tempfile.TemporaryDirectory(prefix="table08_") as temporary:
            root = Path(temporary)
            base = root / "tiny_base"
            base.mkdir()
            vocab = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2,
                                                         "Task": 3, "Python": 4, "def": 5,
                                                         "return": 6, "value": 7, "Assistant": 8,
                                                         "Human": 9}
            for i in range(12):
                vocab[str(i)] = len(vocab)
                vocab[f"add_number_{i}"] = len(vocab)
            raw_tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
            raw_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
                                                pad_token="[PAD]", eos_token="[EOS]",
                                                unk_token="[UNK]")
            tokenizer.save_pretrained(base)
            model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_positions=64,
                                                n_embd=32, n_layer=1, n_head=4,
                                                pad_token_id=0, eos_token_id=1))
            model.save_pretrained(base)
            cfg = {"artifact_root": str(root / "artifacts"),
                   "source": {"dataset": "fixture", "format": "codesearchnet", "subset": "python", "split": "train",
                              "revision": "fixture", "n_per_arm": 4,
                              "calibration_per_arm": 1, "test_per_arm": 1,
                              "max_prompt_chars": 300, "max_response_chars": 1100},
                   "training": {"max_length": 32, "batch_size": 1, "grad_accumulation": 1,
                                "learning_rate": 0.0001, "epochs": 1,
                                "optimizer": "adamw_torch", "dtype": "float32", "seed": 1},
                   "downstream": {"max_examples": 2, "validation_examples": 1, "epochs": 1},
                   "families": {"1b": str(base)}, "profiles": {"1b": ["hh_sft"]}}
            config_path = root / "config.yaml"
            source_rows = [{"func_documentation_string": f"Write operation number {i} to add integers",
                            "func_code_string": f"def add_number_{i}(value):\n    return value + {i} # fixture response"}
                           for i in range(12)]
            source_path = root / "source.jsonl"
            write_jsonl(source_path, source_rows)
            source_manifest = root / "source_manifest.json"
            cfg["source"].update(local_jsonl=str(source_path), filter_manifest=str(source_manifest),
                                 local_sha256=sha256_file(source_path))
            write_json(source_manifest, {"dataset": "fixture", "revision": "fixture", "subset": "python", "split": "train",
                                         "limits": {"max_prompt_chars": 300, "max_response_chars": 1100},
                                         "local_sha256": cfg["source"]["local_sha256"], "retained_pairs": 12,
                                         "reference_documents": 416000, "retained_matches_under_this_rule": 0,
                                         "eligible_sha256": cfg["source"]["local_sha256"],
                                         "retained_detected_matches": 0,
                                         "reference_scope": "416000 documents (synthetic test metadata)"})
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

            def run(module, *args):
                name = f"experiments.table_08_controlled_provenance.shared.{module}"
                argv = ["--config", str(config_path), *args]
                if os.environ.get("TABLE8_SMOKE_SUBPROCESSES") == "1":
                    subprocess.run([sys.executable, "-m", name, *argv], check=True,
                                   env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
                    return
                # Exercise the same CLI entry points and real scorer in one
                # process to avoid repeated slow imports on shared storage.
                entry = importlib.import_module(name)
                original_run = subprocess.run
                def invoke(command, **kwargs):
                    if len(command) > 1 and str(command[1]).endswith("score_causal_lm_logprobs.py"):
                        with patch.object(sys, "argv", command[1:]):
                            runpy.run_path(command[1], run_name="__main__")
                        return subprocess.CompletedProcess(command, 0)
                    return original_run(command, **kwargs)
                with patch.object(sys, "argv", [name, *argv]), patch("subprocess.run", side_effect=invoke):
                    entry.main()

            run("prepare")
            prepared = root / "artifacts" / "prepared"
            self.assertFalse({r["id"] for r in read_jsonl(prepared / "s_train.jsonl")} &
                             {r["id"] for r in read_jsonl(prepared / "s_prime.jsonl")})
            downstream = root / "artifacts" / "downstream" / "hh_sft"
            write_jsonl(downstream / "train.jsonl", [
                {"prompt": "Human: Say hi\nAssistant:", "response": "Hello"},
                {"prompt": "Human: Say bye\nAssistant:", "response": "Goodbye"}])
            write_json(downstream / "manifest.json", {"config_fingerprint": fingerprint(cfg)})
            run("train", "--family", "1b", "--stage", "parent", "--allow-cpu-smoke")
            for stage in ("target", "control"):
                run("train", "--family", "1b", "--stage", stage,
                    "--profile", "hh_sft", "--allow-cpu-smoke")
            for stage, extra in (("base", []), ("parent", []),
                                 ("target", ["--profile", "hh_sft"]),
                                 ("control", ["--profile", "hh_sft"])):
                run("audit", "--family", "1b", "--stage", stage,
                    "--batch-size", "1", *extra)
            run("compile")
            report = (root / "artifacts" / "table_06_report.md").read_text()
            for stage in ("base", "parent", "target", "control"):
                self.assertIn(f"| {stage} |", report)


if __name__ == "__main__":
    unittest.main()

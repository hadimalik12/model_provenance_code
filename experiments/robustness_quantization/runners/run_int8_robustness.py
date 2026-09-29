"""Run a Pythia parent/derived-target FP16-vs-INT8 provenance robustness audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
SCORE_KEYS = "min_k_20_logprob"


@dataclass(frozen=True)
class Representation:
    name: str
    model: str
    quantization: str
    role: str
    dtype: str


def representations(parent_model: str, target_model: str) -> tuple[Representation, ...]:
    """Return one parent and FP16/INT8 views of a verified-derived target."""
    return (
        Representation("parent_fp16", parent_model, "none", "parent", "float16"),
        Representation("target_fp16", target_model, "none", "target", "float16"),
        # LLM.int8 weights stay quantized; FP32 residual/non-linear modules
        # prevent non-finite logits observed with FP16 residuals on PACE GPUs.
        Representation("target_int8", target_model, "int8", "target", "float32"),
    )


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/robustness_quantization/int8_pythia1_4b_lamini_seed0",
    )
    parser.add_argument("--parent-model", default="EleutherAI/pythia-1.4b")
    parser.add_argument(
        "--target-model", default="herMaster/pythia1.4B-finetuned-on-lamini-docs"
    )
    parser.add_argument(
        "--target-slug",
        help="Target namespace below scores/ and results/; defaults to the target model ID.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-individual-report",
        action="store_true",
        help="Do not write a per-target report; use for a multi-target batch run.",
    )
    return parser.parse_args()


def run(command: list[str], expected: Path | None, resume: bool) -> None:
    if resume and expected is not None and expected.exists():
        print(f"[resume] exists: {expected}", flush=True)
        return
    print("\n+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    root = (REPO_ROOT / args.artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target_slug = args.target_slug or slug(args.target_model)
    config = REPO_ROOT / "experiments/robustness_quantization/configs/int8.yaml"
    # Keep experiment-level artifacts in one root, with target-specific
    # metadata in a separate namespace, following the Table 5 layout.
    config_text = config.read_text()
    config_text = config_text.replace("EleutherAI/pythia-1.4b", args.parent_model)
    config_text = config_text.replace(
        "herMaster/pythia1.4B-finetuned-on-lamini-docs", args.target_model
    )
    config_locks = root / "config_locks"
    config_locks.mkdir(exist_ok=True)
    (config_locks / f"{target_slug}.yaml").write_text(config_text)
    run_specs = root / "run_specs"
    run_specs.mkdir(exist_ok=True)
    (run_specs / f"{target_slug}.json").write_text(
        json.dumps(
            {
                "parent_model": args.parent_model,
                "target_model": args.target_model,
                "seed": args.seed,
                "primary_score": "min_k_20_logprob",
                "target_int8_residual_dtype": "float32",
            },
            indent=2,
        )
        + "\n"
    )

    main_data = root / "prepared" / "mimir_github"
    control_data = root / "prepared" / "nonmember_control"
    run(
        [
            PYTHON,
            "scripts/data/prepare_mimir_domain.py",
            "--config", "github",
            "--split", "ngram_13_0.2",
            "--num-train-member", "560",
            "--num-train-nonmember", "340",
            "--num-test-per-class", "400",
            "--max-words", "64",
            "--min-words", "8",
            "--seed", str(args.seed),
            "--allow-overlap",
            "--output-dir", str(main_data),
        ],
        main_data / "test.jsonl",
        args.resume,
    )
    run(
        [
            PYTHON,
            "scripts/data/prepare_mimir_domain_nonmember_control.py",
            "--config", "github",
            "--ngram-split", "ngram_13_0.2",
            "--num-train-per-class", "150",
            "--num-test-per-class", "150",
            "--max-words", "64",
            "--min-words", "8",
            "--seed", str(args.seed),
            "--output-dir", str(control_data),
        ],
        control_data / "test.jsonl",
        args.resume,
    )

    parent_results = root / "results" / "main" / target_slug / "parent_fp16" / "results.json"
    for representation in representations(args.parent_model, args.target_model):
        print(
            f"\n=== {representation.name}: {representation.model} "
            f"({representation.quantization}) ===",
            flush=True,
        )
        score_dir = root / "scores" / "main" / target_slug / representation.name
        result_dir = root / "results" / "main" / target_slug / representation.name
        run(
            [
                PYTHON,
                "scripts/scoring/score_causal_lm_logprobs.py",
                "--model", representation.model,
                "--train-file", str(main_data / "train.jsonl"),
                "--test-file", str(main_data / "test.jsonl"),
                "--output-dir", str(score_dir),
                "--min-k-pcts", "20",
                "--batch-size", str(args.batch_size),
                "--dtype", representation.dtype,
                "--quantization", representation.quantization,
                "--max-length", "512",
            ],
            score_dir / "test_scores.jsonl",
            args.resume,
        )
        parent_arg = (
            ["--parent-results", str(parent_results)]
            if representation.role == "target"
            else []
        )
        run(
            [
                PYTHON,
                "scripts/audit/run_threshold_audit.py",
                "--train-scores", str(score_dir / "train_scores.jsonl"),
                "--test-scores", str(score_dir / "test_scores.jsonl"),
                "--output-dir", str(result_dir),
                "--score-keys", SCORE_KEYS,
                "--primary-score", "min_k_20_logprob",
                "--model-label", representation.model,
                "--mimir-split", "ngram_13_0.2",
                "--num-test-per-class", "400",
                "--seed", str(args.seed),
                "--run-shuffled-control",
                "--no-copy-report-to-docs",
                *parent_arg,
            ],
            result_dir / "results.json",
            args.resume,
        )

        if representation.role == "target":
            control_scores = root / "scores" / "nonmember_control" / target_slug / representation.name
            control_results = root / "results" / "nonmember_control" / target_slug / representation.name
            run(
                [
                    PYTHON,
                    "scripts/scoring/score_causal_lm_logprobs.py",
                    "--model", representation.model,
                    "--train-file", str(control_data / "train.jsonl"),
                    "--test-file", str(control_data / "test.jsonl"),
                    "--output-dir", str(control_scores),
                    "--min-k-pcts", "20",
                    "--batch-size", str(args.batch_size),
                    "--dtype", representation.dtype,
                    "--quantization", representation.quantization,
                ],
                control_scores / "test_scores.jsonl",
                args.resume,
            )
            run(
                [
                    PYTHON,
                    "scripts/audit/run_threshold_audit.py",
                    "--train-scores", str(control_scores / "train_scores.jsonl"),
                    "--test-scores", str(control_scores / "test_scores.jsonl"),
                    "--output-dir", str(control_results),
                    "--score-keys", SCORE_KEYS,
                    "--primary-score", "min_k_20_logprob",
                    "--model-label", representation.model,
                    "--num-test-per-class", "150",
                    "--seed", str(args.seed),
                    "--no-copy-report-to-docs",
                ],
                control_results / "results.json",
                args.resume,
            )

    if not args.skip_individual_report:
        run(
            [
                PYTHON,
                "experiments/robustness_quantization/reports/compile.py",
                "--artifact-root", str(root),
                "--target-slug", target_slug,
            ],
            root / "reports" / "details" / f"{target_slug}.csv",
            resume=False,
        )


if __name__ == "__main__":
    main()

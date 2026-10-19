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
MIMIR_REVISION = "02500d3b7cece0cb7628e939ba9fc93fdb6362ae"
MODEL_REVISIONS = {
    "EleutherAI/pythia-1b": "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2",
    "Leogrin/eleuther-pythia1b-hh-sft": "b29293529d7fb430257df6ae8079afeba2f42f9b",
    "EleutherAI/pythia-1.4b": "fedc38a16eea3bd36a96b906d78d11d2ce18ed79",
    "herMaster/pythia1.4B-finetuned-on-lamini-docs": "a6c699d6832bd3852c13d9c728c8599666b04c22",
    "kykim0/pythia-1.4b-tulu-v2-mix": "4d21350c5197fcfc6cac0b2db277c3e9a158b303",
    "LinguaCustodia/fin-pythia-1.4b": "818675a1325994d01daaec677b9ac40dddcc916d",
    "lomahony/pythia-1.4b-helpful-dpo": "599b0f9ff8bdbccb4ccc8bf8a6a3cb7770c1b800",
    "lomahony/pythia-1.4b-helpful-sft": "da0e31a28430fe2b03b009ebbfdbee74010931c2",
}


def pinned_revision(model: str) -> str:
    try:
        return MODEL_REVISIONS[model]
    except KeyError as exc:
        raise KeyError(f"No immutable revision declared for {model}") from exc


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
        default="artifacts/pythia_int8_quantization/int8_pythia1_4b_lamini_seed0",
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
    config = REPO_ROOT / "experiments/pythia_int8_quantization/configs/int8.yaml"
    # Keep experiment-level artifacts in one root, with target-specific
    # metadata in a separate namespace.
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
                "parent_revision": pinned_revision(args.parent_model),
                "target_model": args.target_model,
                "target_revision": pinned_revision(args.target_model),
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
            "--dataset-revision", MIMIR_REVISION,
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
            "--dataset-revision", MIMIR_REVISION,
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
                "--model-revision", pinned_revision(representation.model),
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
                "--run-shuffled-control",
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
                    "--model-revision", pinned_revision(representation.model),
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
                ],
                control_results / "results.json",
                args.resume,
            )

    if not args.skip_individual_report:
        run(
            [
                PYTHON,
                "experiments/pythia_int8_quantization/reports/compile.py",
                "--artifact-root", str(root),
                "--target-slug", target_slug,
            ],
            root / "reports" / "details" / f"{target_slug}.csv",
            resume=False,
        )


if __name__ == "__main__":
    main()

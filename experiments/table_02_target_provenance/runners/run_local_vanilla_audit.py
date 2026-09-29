"""Run the vanilla GitHub shard audit locally, without Slurm.

This is the focused LLM rerun for the paper's Pythia-1B and Pythia-1.4B
known-lineage experiments.  It uses the paper's MIN-K 20% score plus a
pre-specified mean-log-probability secondary score.  For each seed it creates:

* a MIMIR GitHub member/nonmember calibration and held-out evaluation split;
* parent and verified-derived-target audits on that identical split;
* a shuffled-label calibration control for every main audit; and
* a nonmember-vs-nonmember control for every target audit.

Run on an allocated GPU node, for example:

    python experiments/table_02_target_provenance/runners/run_local_vanilla_audit.py \
      --seeds 0 --batch-size 4 --dtype float16

After checking that seed 0 works, launch all pre-specified splits:

    python experiments/table_02_target_provenance/runners/run_local_vanilla_audit.py \
      --seeds 0,1,2 --batch-size 4 --dtype float16 --resume

The companion compiler is:

    python experiments/table_02_target_provenance/reports/compile_local_vanilla_audit.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
SCORES = "mean_logprob,min_k_20_logprob"


@dataclass(frozen=True)
class Family:
    name: str
    parent: str
    targets: tuple[str, ...]


# Include only targets represented in the submitted paper's Pythia table and
# whose claimed parent family is explicit in their model identifier/model card.
FAMILIES = {
    "1b": Family(
        name="Pythia-1B",
        parent="EleutherAI/pythia-1b",
        targets=("Leogrin/eleuther-pythia1b-hh-sft",),
    ),
    "1.4b": Family(
        name="Pythia-1.4B",
        parent="EleutherAI/pythia-1.4b",
        targets=(
            "herMaster/pythia1.4B-finetuned-on-lamini-docs",
            "kykim0/pythia-1.4b-tulu-v2-mix",
            "LinguaCustodia/fin-pythia-1.4b",
            "lomahony/pythia-1.4b-helpful-dpo",
            "lomahony/pythia-1.4b-helpful-sft",
        ),
    ),
}


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local vanilla Pythia GitHub provenance audits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--families", default="1b,1.4b",
                        help="Comma-separated subset of: 1b,1.4b")
    parser.add_argument("--seeds", default="0",
                        help="Comma-separated fixed MIMIR split seeds")
    parser.add_argument("--artifact-root",
                        default="artifacts/table_02_target_provenance/local_vanilla")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""),
                        help="Optional Hugging Face token; defaults to $HF_TOKEN")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="float16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-words", type=int, default=64)
    parser.add_argument("--min-words", type=int, default=8)
    # MIMIR GitHub yields 963 usable member snippets after the required
    # 64-word preprocessing and deduplication.  560 + 400 therefore fits;
    # the paper's nominal 600 + 400 does not fit this validated source.
    parser.add_argument("--calibration-members", type=int, default=560)
    parser.add_argument("--calibration-nonmembers", type=int, default=340)
    parser.add_argument("--heldout-per-class", type=int, default=400)
    parser.add_argument("--resume", action="store_true",
                        help="Skip a step when its expected output already exists")
    parser.add_argument("--skip-nonmember-control", action="store_true",
                        help="Skip the nonmember-vs-nonmember target controls")
    return parser.parse_args()


def run(command: list[str], expected: Path | None, *, resume: bool) -> None:
    if resume and expected and expected.exists():
        print(f"[resume] exists: {expected}", flush=True)
        return
    print("\n+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def prepare_data(args: argparse.Namespace, seed: int, seed_root: Path) -> tuple[Path, Path]:
    main_data = seed_root / "prepared" / "mimir_github"
    control_data = seed_root / "prepared" / "mimir_github_nonmember_control"
    token_args = ["--token", args.hf_token] if args.hf_token else []

    run([
        PYTHON, "scripts/data/prepare_mimir_domain.py",
        "--config", "github", "--split", "ngram_13_0.2",
        "--num-train-member", str(args.calibration_members),
        "--num-train-nonmember", str(args.calibration_nonmembers),
        "--num-test-per-class", str(args.heldout_per_class),
        "--max-words", str(args.max_words), "--min-words", str(args.min_words),
        "--seed", str(seed), "--output-dir", str(main_data), *token_args,
    ], main_data / "test.jsonl", resume=args.resume)

    if not args.skip_nonmember_control:
        run([
            PYTHON, "scripts/data/prepare_mimir_domain_nonmember_control.py",
            "--config", "github", "--ngram-split", "ngram_13_0.2",
            "--num-train-per-class", str(args.calibration_nonmembers),
            "--num-test-per-class", str(args.heldout_per_class),
            "--max-words", str(args.max_words), "--min-words", str(args.min_words),
            "--seed", str(seed), "--output-dir", str(control_data), *token_args,
        ], control_data / "test.jsonl", resume=args.resume)
    return main_data, control_data


def score_and_audit(
    args: argparse.Namespace,
    *,
    seed: int,
    seed_root: Path,
    model: str,
    main_data: Path,
    control_data: Path,
    parent_results: Path | None,
    is_target: bool,
) -> Path:
    model_slug = slug(model)
    main_scores = seed_root / "scores" / "main" / model_slug
    main_results = seed_root / "results" / "main" / model_slug

    run([
        PYTHON, "scripts/scoring/score_causal_lm_logprobs.py",
        "--model", model,
        "--train-file", str(main_data / "train.jsonl"),
        "--test-file", str(main_data / "test.jsonl"),
        "--output-dir", str(main_scores),
        "--min-k-pcts", "20", "--batch-size", str(args.batch_size),
        "--dtype", args.dtype, "--max-length", str(args.max_length),
    ], main_scores / "test_scores.jsonl", resume=args.resume)

    parent_arg = ["--parent-results", str(parent_results)] if parent_results else []
    run([
        PYTHON, "scripts/audit/run_threshold_audit.py",
        "--train-scores", str(main_scores / "train_scores.jsonl"),
        "--test-scores", str(main_scores / "test_scores.jsonl"),
        "--output-dir", str(main_results),
        "--score-keys", SCORES, "--primary-score", "min_k_20_logprob",
        "--model-label", model, "--mimir-split", "ngram_13_0.2",
        "--num-train-per-class", str(args.calibration_members),
        "--num-test-per-class", str(args.heldout_per_class),
        "--max-words", str(args.max_words), "--min-words", str(args.min_words),
        "--seed", str(seed), "--batch-size", str(args.batch_size),
        "--data-dir", str(main_data), "--run-shuffled-control",
        "--no-copy-report-to-docs", *parent_arg,
    ], main_results / "results.json", resume=args.resume)

    if is_target and not args.skip_nonmember_control:
        control_scores = seed_root / "scores" / "nonmember_control" / model_slug
        control_results = seed_root / "results" / "nonmember_control" / model_slug
        run([
            PYTHON, "scripts/scoring/score_causal_lm_logprobs.py",
            "--model", model,
            "--train-file", str(control_data / "train.jsonl"),
            "--test-file", str(control_data / "test.jsonl"),
            "--output-dir", str(control_scores),
            "--min-k-pcts", "20", "--batch-size", str(args.batch_size),
            "--dtype", args.dtype, "--max-length", str(args.max_length),
        ], control_scores / "test_scores.jsonl", resume=args.resume)
        run([
            PYTHON, "scripts/audit/run_threshold_audit.py",
            "--train-scores", str(control_scores / "train_scores.jsonl"),
            "--test-scores", str(control_scores / "test_scores.jsonl"),
            "--output-dir", str(control_results),
            "--score-keys", SCORES, "--primary-score", "min_k_20_logprob",
            "--model-label", model, "--mimir-split", "ngram_13_0.2",
            "--num-test-per-class", str(args.heldout_per_class),
            "--max-words", str(args.max_words), "--min-words", str(args.min_words),
            "--seed", str(seed), "--batch-size", str(args.batch_size),
            "--data-dir", str(control_data), "--no-copy-report-to-docs",
        ], control_results / "results.json", resume=args.resume)
    return main_results / "results.json"


def main() -> None:
    args = parse_args()
    family_ids = [entry.strip() for entry in args.families.split(",") if entry.strip()]
    unknown = sorted(set(family_ids) - set(FAMILIES))
    if unknown:
        raise SystemExit(f"Unknown family id(s): {', '.join(unknown)}")
    seeds = [int(entry.strip()) for entry in args.seeds.split(",") if entry.strip()]
    artifact_root = (REPO_ROOT / args.artifact_root).resolve()

    print("Local vanilla Pythia GitHub audit")
    print(f"Families: {', '.join(family_ids)} | Seeds: {seeds}")
    print("Scores: MIN-K 20% (primary), mean log-probability (secondary)")
    print(f"Artifacts: {artifact_root}", flush=True)

    for seed in seeds:
        seed_root = artifact_root / f"seed_{seed}"
        main_data, control_data = prepare_data(args, seed, seed_root)
        for family_id in family_ids:
            family = FAMILIES[family_id]
            print(f"\n=== seed={seed}, family={family.name} ===", flush=True)
            parent_results = score_and_audit(
                args, seed=seed, seed_root=seed_root, model=family.parent,
                main_data=main_data, control_data=control_data,
                parent_results=None, is_target=False,
            )
            for target in family.targets:
                score_and_audit(
                    args, seed=seed, seed_root=seed_root, model=target,
                    main_data=main_data, control_data=control_data,
                    parent_results=parent_results, is_target=True,
                )

    print("\nDone. Compile with:")
    print(f"  {PYTHON} experiments/table_02_target_provenance/reports/compile_local_vanilla_audit.py --artifact-root {args.artifact_root}")


if __name__ == "__main__":
    main()

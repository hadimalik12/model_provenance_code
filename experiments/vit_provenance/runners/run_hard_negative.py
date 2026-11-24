"""Run the AISTATS ImageNet hard-negative on the submitted public-ViT split.

The target is Meta's pinned DeiT-Base ImageNet-1K checkpoint.  It shares the
ImageNet witness source but is not a fine-tune of the claimed Google ViT
parent.  Scoring and threshold calibration are delegated to the exact scripts
used by the submitted public-ViT audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
MODEL = "facebook/deit-base-patch16-224"
MODEL_REVISION = "fb2c78a54a5637dec350432794f7b93e31f910c9"
DATASET = "evanarlian/imagenet_1k_resized_256"
DATASET_REVISION = "8de107e08d1f61884d5a4ec2dea8740667264a60"
EXPECTED_MANIFEST = {
    "dataset_repo": DATASET,
    "dataset_revision": DATASET_REVISION,
    "train_shard_idxs": [0, 1, 2, 3, 4],
    "val_shard_idxs": [0, 1],
    "members_per_class": 50,
    "nonmembers_per_class": 50,
    "train_frac": 0.5,
    "n_classes_used": 77,
    "seed": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("artifacts/vit_provenance/public_derivatives_seed0/prepared/imagenet_shard"),
        help="Exact prepared split shared with the submitted public-ViT audit",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/vit_provenance/aistats27_hard_negative/deit_base_imagenet1k"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Require the exact shared split instead of preparing it when absent",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Score eight members and eight controls per phase from the exact split",
    )
    parser.add_argument("--smoke-per-label", type=int, default=8)
    return parser.parse_args()


def run(command: list[str], expected: Path | None, *, resume: bool) -> None:
    if resume and expected is not None and expected.is_file():
        print(f"[resume] exists: {expected}", flush=True)
        return
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def stratified_smoke(records: list[dict], per_label: int) -> list[dict]:
    """Keep a deterministic, balanced subset without changing source records."""
    if per_label < 1:
        raise ValueError("--smoke-per-label must be positive")
    selected = []
    for label in (1, 0):
        group = [record for record in records if int(record["label"]) == label]
        if len(group) < per_label:
            raise ValueError(f"Need {per_label} records for label {label}; found {len(group)}")
        selected.extend(group[:per_label])
    return selected


def validate_shared_data(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared ImageNet manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = [
        f"{key}: expected {expected!r}, found {manifest.get(key)!r}"
        for key, expected in EXPECTED_MANIFEST.items()
        if manifest.get(key) != expected
    ]
    for name in ("train.jsonl", "test.jsonl"):
        if not (data_dir / name).is_file():
            errors.append(f"missing {name}")
    if errors:
        raise ValueError("Prepared data do not match the submitted protocol:\n- " + "\n- ".join(errors))
    return manifest


def prepare_data(data_dir: Path, *, resume: bool, skip_prepare: bool) -> dict:
    manifest = data_dir / "manifest.json"
    if not manifest.is_file():
        if skip_prepare:
            raise FileNotFoundError(
                f"Exact prepared split not found at {data_dir}; refusing to substitute another split"
            )
        run(
            [
                PYTHON,
                "experiments/vit_provenance/runners/prepare_shards.py",
                "--dataset-revision", DATASET_REVISION,
                "--train-shard-idxs", "0,1,2,3,4",
                "--val-shard-idxs", "0,1",
                "--members-per-class", "50",
                "--nonmembers-per-class", "50",
                "--max-classes", "77",
                "--train-frac", "0.5",
                "--seed", "0",
                "--output-dir", str(data_dir),
            ],
            manifest,
            resume=resume,
        )
    return validate_shared_data(data_dir)


def smoke_data(data_dir: Path, root: Path, *, per_label: int, resume: bool) -> Path:
    output = root / "prepared_smoke"
    manifest_path = output / "manifest.json"
    if resume and manifest_path.is_file():
        return output
    for split in ("train", "test"):
        records = stratified_smoke(read_jsonl(data_dir / f"{split}.jsonl"), per_label)
        write_jsonl(output / f"{split}.jsonl", records)
    manifest_path.write_text(
        json.dumps(
            {
                "source_data_dir": str(data_dir),
                "selection": "first N records per label from each fixed submitted-protocol phase",
                "per_label_per_phase": per_label,
                "scientific_result": False,
                "purpose": "pipeline smoke test only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def primary_metrics(results_path: Path) -> dict:
    blob = json.loads(results_path.read_text(encoding="utf-8"))
    return next(
        result["test"]
        for result in blob["main_results"]
        if result["score_name"] == "clv_score"
    )


def write_summary(root: Path, results_path: Path, *, smoke: bool) -> dict:
    metrics = primary_metrics(results_path)
    n_pairs = int(metrics["n_pos"])
    advantage = float(metrics["shard_advantage"])
    audit_cutoff = math.sqrt(math.log(40.0) / n_pairs)
    decision = "reject" if advantage > audit_cutoff else "fail_to_reject"
    summary = {
        "status": "smoke_only" if smoke else "observed",
        "scientific_result": not smoke,
        "target_model": MODEL,
        "target_revision": MODEL_REVISION,
        "claimed_parent": "google/vit-base-patch16-224",
        "shared_training_data": "ImageNet-1K",
        "score": "clv_score",
        "n_evaluation_pairs": n_pairs,
        "signed_shard_advantage": advantage,
        "level": 0.05,
        "audit_cutoff": audit_cutoff,
        "decision": decision,
        "interpretation": (
            "A positive result indicates detection of a shared ImageNet data signal, not exact lineage. "
            "A negative result supports specificity for this control but does not prove lineage."
        ),
    }
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (report_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    lines = [
        "# Independent ImageNet-1K transformer hard-negative",
        "",
        f"**Status:** `{summary['status']}`  ",
        f"**Target:** `{MODEL}@{MODEL_REVISION}`  ",
        f"**Signed advantage:** {advantage:.4f}  ",
        f"**Level-0.05 cutoff:** {audit_cutoff:.4f}  ",
        f"**Decision:** `{decision}`",
        "",
        summary["interpretation"],
    ]
    (report_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.prepare_only and args.skip_prepare:
        raise SystemExit("--prepare-only and --skip-prepare are mutually exclusive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    data_dir = (REPO_ROOT / args.data_dir).resolve()
    artifact_root = (REPO_ROOT / args.artifact_root).resolve()
    prepare_data(data_dir, resume=args.resume, skip_prepare=args.skip_prepare)
    if args.prepare_only:
        print(f"Prepared and validated exact split: {data_dir}")
        return

    run_root = artifact_root / "smoke" if args.smoke else artifact_root / "production"
    scoring_data = (
        smoke_data(data_dir, run_root, per_label=args.smoke_per_label, resume=args.resume)
        if args.smoke
        else data_dir
    )
    run(
        [PYTHON, "scripts/setup/verify_hf_model_access.py", "--model", f"{MODEL}@{MODEL_REVISION}"],
        None,
        resume=False,
    )
    scores = run_root / "scores"
    results = run_root / "results"
    run(
        [
            PYTHON,
            "experiments/vit_provenance/scoring/score_clv.py",
            "--target-model", MODEL,
            "--target-revision", MODEL_REVISION,
            "--train-file", str(scoring_data / "train.jsonl"),
            "--test-file", str(scoring_data / "test.jsonl"),
            "--output-dir", str(scores),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--dtype", args.dtype,
        ],
        scores / "manifest.json",
        resume=args.resume,
    )
    run(
        [
            PYTHON,
            "scripts/audit/run_threshold_audit.py",
            "--train-scores", str(scores / "train_scores.jsonl"),
            "--test-scores", str(scores / "test_scores.jsonl"),
            "--output-dir", str(results),
            "--score-keys", "clv_score",
            "--primary-score", "clv_score",
            "--model-label", MODEL,
            "--run-shuffled-control",
        ],
        results / "results.json",
        resume=args.resume,
    )
    summary = write_summary(run_root, results / "results.json", smoke=args.smoke)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

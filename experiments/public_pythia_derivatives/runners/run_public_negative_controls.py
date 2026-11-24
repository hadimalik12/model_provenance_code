"""Run the two requested public-Pythia negative controls on a frozen split.

The runner deliberately lives outside ``run_audit.py`` so the submitted model
roster and immutable submitted results are not changed.  It reuses the exact
matched MIMIR GitHub construction/evaluation files from the AISTATS analysis.
Both scores are derived from one set of cached token log probabilities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON = sys.executable
DEFAULT_DATA = Path(
    "artifacts/public_pythia_derivatives/aistats27_matched/seed_0/prepared/"
    "mimir_github_matched_v3"
)
DEFAULT_ARTIFACT_ROOT = Path(
    "artifacts/public_pythia_derivatives/public_negative_controls/full"
)
PRIMARY_SCORE = "min_k_20_logprob"
SECONDARY_SCORE = "mean_logprob"


@dataclass(frozen=True)
class ControlModel:
    key: str
    model_id: str
    revision: str
    batch_size: int
    question: str


MODELS = {
    "gpt_neo": ControlModel(
        key="gpt_neo",
        model_id="EleutherAI/gpt-neo-1.3B",
        revision="dbe59a7f4a88d01d1ba9798d78dbe3fe038792c8",
        batch_size=4,
        question="Same Pile corpus, independently trained non-Pythia model",
    ),
    "tulu_llama": ControlModel(
        key="tulu_llama",
        model_id="allenai/tulu-2-7b",
        revision="3c6e328ae91fabdd0daf09de16887de9615c1f66",
        batch_size=1,
        question="Llama-family model trained on the Tulu V2 SFT mixture",
    ),
}

CACHED_PYTHIA_TULU = {
    "model_id": "kykim0/pythia-1.4b-tulu-v2-mix",
    "revision": "4d21350c5197fcfc6cac0b2db277c3e9a158b303",
    "score_dir": Path(
        "artifacts/public_pythia_derivatives/aistats27_matched/seed_0/scores/main/"
        "kykim0__pythia-1.4b-tulu-v2-mix"
    ),
    "result_file": Path(
        "artifacts/public_pythia_derivatives/aistats27_matched/seed_0/results/main/"
        "kykim0__pythia-1.4b-tulu-v2-mix/results.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", default="gpt_neo,tulu_llama",
        help="Comma-separated subset of gpt_neo,tulu_llama",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--max-examples", type=int,
        help="Smoke-test cap per split. Any capped run is marked non-scientific.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> tuple[int, dict[int, int]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    labels = {label: sum(row["label"] == label for row in rows) for label in (0, 1)}
    return len(rows), labels


def command_text(command: list[str]) -> str:
    return " ".join(command)


def run(command: list[str], expected: Path, *, resume: bool, dry_run: bool) -> None:
    if resume and expected.is_file():
        print(f"[resume] {expected}", flush=True)
        return
    print("+ " + command_text(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def score_and_audit(
    model: ControlModel,
    *,
    data_dir: Path,
    artifact_root: Path,
    dtype: str,
    max_length: int,
    max_examples: int | None,
    resume: bool,
    dry_run: bool,
) -> None:
    model_root = artifact_root / model.key
    score_dir = model_root / "scores"
    result_dir = model_root / "audit"
    cap = ["--max-examples", str(max_examples)] if max_examples else []
    run(
        [
            PYTHON,
            "scripts/scoring/score_causal_lm_logprobs.py",
            "--model", model.model_id,
            "--model-revision", model.revision,
            "--train-file", str(data_dir / "train.jsonl"),
            "--test-file", str(data_dir / "test.jsonl"),
            "--output-dir", str(score_dir),
            "--min-k-pcts", "20",
            "--batch-size", str(model.batch_size),
            "--dtype", dtype,
            "--max-length", str(max_length),
            *cap,
        ],
        score_dir / "test_scores.jsonl",
        resume=resume,
        dry_run=dry_run,
    )
    run(
        [
            PYTHON,
            "scripts/audit/run_threshold_audit.py",
            "--train-scores", str(score_dir / "train_scores.jsonl"),
            "--test-scores", str(score_dir / "test_scores.jsonl"),
            "--output-dir", str(result_dir),
            "--score-keys", f"{PRIMARY_SCORE},{SECONDARY_SCORE}",
            "--primary-score", PRIMARY_SCORE,
            "--model-label", model.model_id,
            "--run-shuffled-control",
        ],
        result_dir / "results.json",
        resume=resume,
        dry_run=dry_run,
    )


def compile_summary(artifact_root: Path, max_examples: int | None) -> None:
    rows: list[dict] = []
    sources = [
        (
            "pythia_tulu_cached",
            CACHED_PYTHIA_TULU["model_id"],
            REPO_ROOT / CACHED_PYTHIA_TULU["result_file"],
            REPO_ROOT / CACHED_PYTHIA_TULU["score_dir"] / "manifest.json",
            "Pythia family, Tulu V2 SFT mixture",
        ),
        *[
            (
                model.key,
                model.model_id,
                artifact_root / model.key / "audit" / "results.json",
                artifact_root / model.key / "scores" / "manifest.json",
                model.question,
            )
            for model in MODELS.values()
        ],
    ]
    for key, model_id, result_path, manifest_path, question in sources:
        if not result_path.is_file():
            continue
        result_blob = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.is_file() else {}
        )
        for score in (PRIMARY_SCORE, SECONDARY_SCORE):
            item = next(
                row for row in result_blob["main_results"] if row["score_name"] == score
            )
            metrics = item["test"]
            cutoff = math.sqrt(math.log(40) / metrics["n_pos"])
            cost = manifest.get("cost", {})
            rows.append(
                {
                    "control": key,
                    "model": model_id,
                    "question": question,
                    "score": score,
                    "evaluation_examples": metrics["n"],
                    "signed_advantage": metrics["shard_advantage"],
                    "auc": metrics["auc"],
                    "cutoff_0.05": cutoff,
                    "decision": (
                        "detects_witness"
                        if metrics["shard_advantage"] > cutoff
                        else "no_witness_detection"
                    ),
                    "model_forward_passes": cost.get("model_forward_passes"),
                    "scored_tokens": cost.get("scored_tokens"),
                    "wall_seconds": cost.get("total_wall_seconds"),
                }
            )
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_result": max_examples is None,
        "smoke_cap_per_split": max_examples,
        "primary_score": PRIMARY_SCORE,
        "secondary_score": SECONDARY_SCORE,
        "score_direction": "higher is more member-like",
        "decision_rule": "signed held-out advantage > sqrt(log(40) / held-out members)",
        "rows": rows,
        "interpretation": (
            "A positive decision detects Pile witness information and does not establish "
            "exact parentage or the route of data access."
        ),
    }
    (artifact_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    selected = [entry.strip() for entry in args.models.split(",") if entry.strip()]
    unknown = sorted(set(selected) - set(MODELS))
    if unknown:
        raise SystemExit(f"Unknown model key(s): {', '.join(unknown)}")
    data_dir = (REPO_ROOT / args.data_dir).resolve()
    artifact_root = (REPO_ROOT / args.artifact_root).resolve()
    required = [data_dir / "train.jsonl", data_dir / "test.jsonl"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Missing frozen input(s): " + ", ".join(missing))
    split_info = {}
    for path in required:
        n_rows, labels = count_jsonl(path)
        split_info[path.name] = {
            "rows": n_rows,
            "labels": labels,
            "sha256": sha256(path),
        }
    if not args.max_examples:
        expected = {
            "train.jsonl": {"rows": 680, "labels": {0: 340, 1: 340}},
            "test.jsonl": {"rows": 800, "labels": {0: 400, 1: 400}},
        }
        for name, values in expected.items():
            if split_info[name]["rows"] != values["rows"] or split_info[name]["labels"] != values["labels"]:
                raise SystemExit(f"Frozen split changed unexpectedly: {name}: {split_info[name]}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_result": args.max_examples is None,
        "selected_models": selected,
        "models": {key: MODELS[key].__dict__ for key in selected},
        "cached_pythia_tulu": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in CACHED_PYTHIA_TULU.items()
        },
        "data_dir": str(data_dir),
        "splits": split_info,
        "max_examples_per_split": args.max_examples,
        "max_length": args.max_length,
        "dtype": args.dtype,
        "primary_score": PRIMARY_SCORE,
        "secondary_score": SECONDARY_SCORE,
    }
    (artifact_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8"
    )
    for key in selected:
        score_and_audit(
            MODELS[key],
            data_dir=data_dir,
            artifact_root=artifact_root,
            dtype=args.dtype,
            max_length=args.max_length,
            max_examples=args.max_examples,
            resume=args.resume,
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        compile_summary(artifact_root, args.max_examples)
        print(f"Summary: {artifact_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

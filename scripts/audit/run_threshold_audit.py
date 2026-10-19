"""Calibrate threshold distinguishers and evaluate them on a held-out split.

The construction and evaluation score files are always explicit. The command
writes ``results.json`` plus a compact ``audit_report.md`` under ``--output-dir``.
All scores must follow the repository convention: higher is more member-like.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shard_audit.auditing.distinguishers import (
    evaluate_at_threshold,
    run_distinguisher,
    safe_auc,
    tpr_at_fpr,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"Score file is empty: {path}")
    LOGGER.info("Loaded %d records from %s", len(records), path)
    return records


def rounded(values: dict, digits: int = 4) -> dict:
    return {
        key: round(value, digits) if isinstance(value, float) else value
        for key, value in values.items()
    }


def summary_table(results: list[dict], split: str) -> str:
    lines = [
        "| Score | Accuracy | Bal. accuracy | AUC | TPR@1% FPR | Shard adv. | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result[split]
        auc = "N/A" if metrics.get("auc") is None else f'{metrics["auc"]:.3f}'
        tpr = (
            "N/A"
            if metrics.get("tpr_at_1_fpr") is None
            else f'{metrics["tpr_at_1_fpr"]:.3f}'
        )
        lines.append(
            f'| {result["score_name"]} | {metrics["accuracy"]:.3f} '
            f'| {metrics["balanced_accuracy"]:.3f} | {auc} | {tpr} '
            f'| {metrics["shard_advantage"]:.3f} '
            f'| {result["calibrated_threshold"]:.4f} |'
        )
    return "\n".join(lines)


def write_report(
    results: list[dict],
    *,
    primary_score: str,
    model_label: str,
    train_records: list[dict],
    test_records: list[dict],
    criterion: str,
    output_dir: Path,
) -> Path:
    primary = next(
        (result for result in results if result["score_name"] == primary_score),
        results[0],
    )
    n_train_pos = sum(record["label"] == 1 for record in train_records)
    n_test_pos = sum(record["label"] == 1 for record in test_records)
    lines = [
        "# Held-out threshold audit",
        "",
        f'**Date:** {datetime.now(timezone.utc).date().isoformat()}  ',
        f'**Model:** `{model_label}`  ',
        f'**Primary score:** `{primary["score_name"]}`  ',
        f'**Calibration criterion:** `{criterion}`',
        "",
        "The threshold is selected only on the construction split and then held fixed for evaluation.",
        "",
        "| Split | Total | Members | Controls |",
        "|---|---:|---:|---:|",
        f"| Construction | {len(train_records)} | {n_train_pos} | {len(train_records) - n_train_pos} |",
        f"| Evaluation | {len(test_records)} | {n_test_pos} | {len(test_records) - n_test_pos} |",
        "",
        "## Construction split",
        "",
        summary_table(results, "train"),
        "",
        "## Held-out evaluation",
        "",
        summary_table(results, "test"),
        "",
        "All score directions are `higher = more member-like`. Requested control diagnostics are stored in `results.json`.",
    ]
    destination = output_dir / "audit_report.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train-scores", type=Path, required=True)
    parser.add_argument("--test-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-keys",
        default="mean_logprob,min_k_5_logprob,min_k_10_logprob,min_k_20_logprob,min_k_40_logprob",
    )
    parser.add_argument("--primary-score", default="min_k_20_logprob")
    parser.add_argument(
        "--criterion",
        default="balanced_accuracy",
        choices=("accuracy", "balanced_accuracy"),
        help="Metric optimized while sweeping construction-split thresholds",
    )
    parser.add_argument("--n-thresholds", type=int, default=2000)
    parser.add_argument("--model-label", default="unspecified")
    parser.add_argument(
        "--parent-results",
        type=Path,
        help="Parent results.json used for the threshold-transfer diagnostic",
    )
    parser.add_argument(
        "--run-shuffled-control",
        action="store_true",
        help="Break score/label association on both splits as a leakage check",
    )
    parser.add_argument("--shuffled-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_keys = [key.strip() for key in args.score_keys.split(",") if key.strip()]
    train_records = load_jsonl(args.train_scores)
    test_records = load_jsonl(args.test_scores)
    train_labels = [record["label"] for record in train_records]
    test_labels = [record["label"] for record in test_records]

    results = []
    for key in score_keys:
        if key not in train_records[0] or key not in test_records[0]:
            LOGGER.warning("Score key %r is absent from one or both splits; skipping", key)
            continue
        result = run_distinguisher(
            train_labels,
            [record[key] for record in train_records],
            test_labels,
            [record[key] for record in test_records],
            score_name=key,
            criterion=args.criterion,
            n_thresholds=args.n_thresholds,
        )
        results.append(result)
    if not results:
        raise SystemExit("None of the requested score keys exists in both input files")

    shuffled_control = None
    if args.run_shuffled_control:
        rng = random.Random(args.shuffled_seed)
        shuffled_train = list(train_labels)
        shuffled_test = list(test_labels)
        rng.shuffle(shuffled_train)
        rng.shuffle(shuffled_test)
        shuffled_control = []
        for result in results:
            key = result["score_name"]
            control = run_distinguisher(
                shuffled_train,
                [record[key] for record in train_records],
                shuffled_test,
                [record[key] for record in test_records],
                score_name=key,
                criterion=args.criterion,
                n_thresholds=args.n_thresholds,
            )
            shuffled_control.append(
                {
                    "score_name": key,
                    "test_advantage": control["test"]["shard_advantage"],
                    "test_accuracy": control["test"]["accuracy"],
                    "test_auc": control["test"]["auc"],
                }
            )

    parent_transfer = None
    if args.parent_results:
        if not args.parent_results.is_file():
            raise FileNotFoundError(f"Parent results do not exist: {args.parent_results}")
        parent_blob = json.loads(args.parent_results.read_text(encoding="utf-8"))
        parent_results = (
            parent_blob.get("main_results", [])
            if isinstance(parent_blob, dict)
            else parent_blob
        )
        parent_by_score = {result["score_name"]: result for result in parent_results}
        parent_transfer = []
        for result in results:
            key = result["score_name"]
            if key not in parent_by_score:
                continue
            threshold = parent_by_score[key]["calibrated_threshold"]
            test_scores = [record[key] for record in test_records]
            metrics = evaluate_at_threshold(test_labels, test_scores, threshold)
            auc = safe_auc(test_labels, test_scores)
            tpr = tpr_at_fpr(test_labels, test_scores, 0.01)
            parent_transfer.append(
                {
                    "score_name": key,
                    "parent_threshold": threshold,
                    "test_accuracy": round(metrics["accuracy"], 4),
                    "test_balanced_accuracy": round(metrics["balanced_accuracy"], 4),
                    "test_advantage": round(metrics["shard_advantage"], 4),
                    "test_auc": round(auc, 4) if auc is not None else None,
                    "test_tpr_at_1_fpr": round(tpr, 4) if tpr is not None else None,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable = [
        {
            "score_name": result["score_name"],
            "calibration_criterion": result["calibration_criterion"],
            "calibrated_threshold": result["calibrated_threshold"],
            "train": rounded(result["train"]),
            "test": rounded(result["test"]),
        }
        for result in results
    ]
    results_path = args.output_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "main_results": serializable,
                "shuffled_label_control": shuffled_control,
                "parent_threshold_transfer": parent_transfer,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = write_report(
        results,
        primary_score=args.primary_score,
        model_label=args.model_label,
        train_records=train_records,
        test_records=test_records,
        criterion=args.criterion,
        output_dir=args.output_dir,
    )
    LOGGER.info("Results: %s", results_path)
    LOGGER.info("Report: %s", report_path)


if __name__ == "__main__":
    main()

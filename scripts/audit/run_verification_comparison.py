"""Run end-to-end verification comparing un-augmented vs 4-view augmented LLM audit scores."""

import argparse
import json
import logging
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.auditing.distinguishers import run_distinguisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-scores", required=True)
    parser.add_argument("--test-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    train_recs = _load_jsonl(args.train_scores)
    test_recs = _load_jsonl(args.test_scores)

    train_labels = [r["label"] for r in train_recs]
    test_labels = [r["label"] for r in test_recs]

    scores_to_test = ["min_k_20_logprob", "aug_min_k_20_logprob_mean"]
    results = {}

    for score_key in scores_to_test:
        train_scores = [r[score_key] for r in train_recs]
        test_scores = [r[score_key] for r in test_recs]

        res = run_distinguisher(
            train_labels, train_scores,
            test_labels, test_scores,
            score_name=score_key,
            criterion="balanced_accuracy",
            n_thresholds=1000,
        )
        results[score_key] = res

    # Write summary report
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "verification_comparison.md")
    
    orig = results["min_k_20_logprob"]["test"]
    aug = results["aug_min_k_20_logprob_mean"]["test"]

    report_lines = [
        "# End-to-End Audit Verification: Un-augmented vs 4-View Augmented",
        "",
        "| Metric | Original (Un-augmented `min_k_20`) | 4-View Augmented (`aug_min_k_20_logprob_mean`) | Delta |",
        "|---|---:|---:|---:|",
        f"| **Test Accuracy** | {orig['accuracy']:.4f} | {aug['accuracy']:.4f} | {aug['accuracy'] - orig['accuracy']:+.4f} |",
        f"| **Bal. Accuracy** | {orig['balanced_accuracy']:.4f} | {aug['balanced_accuracy']:.4f} | {aug['balanced_accuracy'] - orig['balanced_accuracy']:+.4f} |",
        f"| **Shard Advantage** | {orig['shard_advantage']:.4f} | {aug['shard_advantage']:.4f} | {aug['shard_advantage'] - orig['shard_advantage']:+.4f} |",
        f"| **Test AUC** | {orig['auc'] if orig['auc'] else 0.5:.4f} | {aug['auc'] if aug['auc'] else 0.5:.4f} | {(aug['auc'] or 0.5) - (orig['auc'] or 0.5):+.4f} |",
        "",
        "## Detailed Results",
        "```json",
        json.dumps(results, indent=2),
        "```"
    ]

    with open(summary_path, "w") as f:
        f.write("\n".join(report_lines))

    print("\n" + "\n".join(report_lines[:8]))
    logger.info("Verification report saved to %s", summary_path)


if __name__ == "__main__":
    main()

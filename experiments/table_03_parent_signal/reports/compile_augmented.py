"""Compile the augmented Pythia parent-signal table on MIMIR GitHub.

This is not paper Table 4.  Paper Table 4 is the WikiMIA score-selection table.
This script reports an augmented parent-model sanity check that is closest to an
augmented version of paper Table 3.

Output:
    outputs/replicated_table_4.txt
    outputs/replicated_table_4.csv
    outputs/augmented_parent_signal_github.txt
    outputs/augmented_parent_signal_github.csv

Usage:
    python experiments/table_03_parent_signal/reports/compile.py
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.experiments.experiment_registry import (
    AUG_MIN_K_20,
    PYTHIA_PARENTS,
    VANILLA_MIN_K_20,
    gamma,
    slug,
)
from src.shard_audit.reporting.reporting_tables import (
    load_json,
    score_test_metrics,
    verdict,
    write_csv,
    write_text,
)


SCORE_KEYS = [
    ("Min-K% (k=20)", VANILLA_MIN_K_20),
    ("Aug Min-K% (k=20)", AUG_MIN_K_20),
]

ARTIFACT_ROOT = "artifacts/table_03_parent_signal/augmented_seed0"
RUNS_DIR = f"{ARTIFACT_ROOT}/results"
M_EVAL = 400  # held-out test size per class
GAMMA_005 = gamma(0.05, M_EVAL)


def main():
    lines = []
    csv_rows = []

    header = (
        f"{'Model':<15} {'Score':<18} {'Ctrl Acc':>9} {'Δ_ctrl':>8} "
        f"{'Main Acc':>9} {'Δ_main':>8} {'γ_0.05':>8} {'Verdict'}"
    )
    lines.append(header)
    lines.append("-" * 105)

    for parent in PYTHIA_PARENTS:
        clean = slug(parent.hf_id)

        # Main results
        main_result_file = os.path.join(
            RUNS_DIR, f"github_{clean}", "results.json"
        )

        # Nonmember control results
        ctrl_result_file = os.path.join(
            RUNS_DIR, f"nonmember_control_{clean}", "results.json"
        )

        main_data = load_json(main_result_file)
        ctrl_data = load_json(ctrl_result_file)

        for score_label, score_key in SCORE_KEYS:
            # Default control values
            acc_ctrl = 0.50
            adv_ctrl = 0.00

            # Extract control metrics
            if ctrl_data:
                ctrl_metrics = score_test_metrics(ctrl_data, score_key)
                acc_ctrl = ctrl_metrics.get("accuracy", 0.50)
                adv_ctrl = ctrl_metrics.get("shard_advantage", 0.00)

            # Extract main metrics
            if main_data:
                main_metrics = score_test_metrics(main_data, score_key)
                acc_main = main_metrics.get("accuracy", 0.50)
                adv_main = main_metrics.get("shard_advantage", 0.00)

                lines.append(
                    f"{parent.display_name:<15} {score_label:<18} "
                    f"{acc_ctrl:>8.1%} {adv_ctrl:>+8.3f} "
                    f"{acc_main:>8.1%} {adv_main:>+8.3f} "
                    f"{GAMMA_005:>8.3f} {verdict(adv_main, GAMMA_005)}"
                )

                csv_rows.append({
                    "model": parent.display_name,
                    "score": score_label,
                    "ctrl_acc": round(acc_ctrl, 3),
                    "delta_ctrl": round(adv_ctrl, 3),
                    "main_acc": round(acc_main, 3),
                    "delta_main": round(adv_main, 3),
                    "gamma_005": round(GAMMA_005, 3),
                    "verdict": verdict(adv_main, GAMMA_005),
                })
            else:
                lines.append(
                    f"{parent.display_name:<15} {score_label:<18}"
                    f"{'-- results not found --':>50}"
                )

        # Separator between models
        lines.append("")

    lines.append("-" * 105)
    output_str = "\n".join(lines)
    print(output_str)

    report_path = f"{ARTIFACT_ROOT}/reports/parent_signal.txt"
    write_text(report_path, output_str + "\n")
    print(f"\nSaved to {report_path}")

    if csv_rows:
        csv_path = f"{ARTIFACT_ROOT}/reports/parent_signal.csv"
        write_csv(csv_path, csv_rows)
        print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()

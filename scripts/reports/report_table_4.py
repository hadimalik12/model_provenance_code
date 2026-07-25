"""Compile Table 4: Score Selection on GitHub Data (Pythia 1B–12B, Min-K% k=5/10/20/40).

Reads per-model experiment results from the standard runs directory and produces
a formatted table matching the paper's Table 4 format with Ctrl/Main accuracy,
Δ_ctrl, Δ_main, γ₀.₀₅, and verdict.

Output:
    outputs/replicated_table_4.txt
    outputs/replicated_table_4.csv

Usage:
    python scripts/reports/report_table_4.py
"""

import csv
import json
import math
import os
import sys


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

TABLE_4_MODELS = [
    ("Pythia-1B",   "EleutherAI/pythia-1b"),
    ("Pythia-1.4B", "EleutherAI/pythia-1.4b"),
    ("Pythia-6.9B", "EleutherAI/pythia-6.9b"),
    ("Pythia-12B",  "EleutherAI/pythia-12b"),
]

SCORE_KEYS = [
    ("Min-K% (k=20)", "min_k_20_logprob"),
    ("Aug Min-K% (k=20)", "aug_min_k_20_logprob_mean"),
]

RUNS_DIR = "outputs/runs"
M_EVAL = 400  # held-out test size per class
GAMMA_005 = math.sqrt(math.log(2 / 0.05) / M_EVAL)


def _extract_score_test_metrics(data: dict, score_name: str) -> dict:
    """Extract the test metrics dict for a given score from results.json."""
    if not isinstance(data, dict):
        return {}
    main_results = data.get("main_results", [])
    if isinstance(main_results, list):
        for entry in main_results:
            if isinstance(entry, dict) and entry.get("score_name") == score_name:
                return entry.get("test", {})
    return {}


def main():
    lines = []
    csv_rows = []

    header = (
        f"{'Model':<15} {'Score':<18} {'Ctrl Acc':>9} {'Δ_ctrl':>8} "
        f"{'Main Acc':>9} {'Δ_main':>8} {'γ_0.05':>8} {'Verdict'}"
    )
    lines.append(header)
    lines.append("-" * 105)

    for model_name, model_hf in TABLE_4_MODELS:
        clean = model_hf.replace("/", "__")

        # Main results
        main_result_file = os.path.join(
            RUNS_DIR, f"table4_github_{clean}", "results.json"
        )

        # Nonmember control results
        ctrl_result_file = os.path.join(
            RUNS_DIR, f"table4_github_ctrl_{clean}", "results.json"
        )

        main_data = None
        ctrl_data = None

        if os.path.exists(main_result_file):
            with open(main_result_file) as f:
                main_data = json.load(f)

        if os.path.exists(ctrl_result_file):
            with open(ctrl_result_file) as f:
                ctrl_data = json.load(f)

        for score_label, score_key in SCORE_KEYS:
            # Default control values
            acc_ctrl = 0.50
            adv_ctrl = 0.00

            # Extract control metrics
            if ctrl_data:
                ctrl_metrics = _extract_score_test_metrics(ctrl_data, score_key)
                acc_ctrl = ctrl_metrics.get("accuracy", 0.50)
                adv_ctrl = ctrl_metrics.get("shard_advantage", 0.00)

            # Extract main metrics
            if main_data:
                main_metrics = _extract_score_test_metrics(main_data, score_key)
                acc_main = main_metrics.get("accuracy", 0.50)
                adv_main = main_metrics.get("shard_advantage", 0.00)

                verdict = ("Reject H0 / Yes" if adv_main > GAMMA_005
                           else "Fail to reject / No")

                lines.append(
                    f"{model_name:<15} {score_label:<18} "
                    f"{acc_ctrl:>8.1%} {adv_ctrl:>+8.3f} "
                    f"{acc_main:>8.1%} {adv_main:>+8.3f} "
                    f"{GAMMA_005:>8.3f} {verdict}"
                )

                csv_rows.append({
                    "model": model_name,
                    "score": score_label,
                    "ctrl_acc": round(acc_ctrl, 3),
                    "delta_ctrl": round(adv_ctrl, 3),
                    "main_acc": round(acc_main, 3),
                    "delta_main": round(adv_main, 3),
                    "gamma_005": round(GAMMA_005, 3),
                    "verdict": verdict,
                })
            else:
                lines.append(
                    f"{model_name:<15} {score_label:<18}"
                    f"{'-- results not found --':>50}"
                )

        # Separator between models
        lines.append("")

    lines.append("-" * 105)
    output_str = "\n".join(lines)
    print(output_str)

    # Write text table
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/replicated_table_4.txt", "w") as f:
        f.write(output_str + "\n")
    print("\nSaved to outputs/replicated_table_4.txt")

    # Write CSV
    if csv_rows:
        csv_path = "outputs/replicated_table_4.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(csv_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()

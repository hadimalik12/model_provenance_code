"""Compile paper Table 2 from vanilla MIMIR GitHub target runs."""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.experiments.experiment_registry import PYTHIA_TARGETS, VANILLA_MIN_K_20, gamma, slug
from src.shard_audit.reporting.reporting_tables import (
    load_json,
    score_test_metrics,
    shuffled_control_metrics,
    verdict,
    write_text,
)


M_EVAL = 400
GAMMA_0_05 = gamma(0.05, M_EVAL)
ARTIFACT_ROOT = "artifacts/table_02_target_provenance/vanilla_seed0"
RUNS_DIR = f"{ARTIFACT_ROOT}/results"
OUTPUT_PATH = f"{ARTIFACT_ROOT}/reports/table_2.txt"


def main():
    lines = []
    lines.append(f"{'Parent model':<15} {'Target model':<32} {'Ctrl Acc':>9} {'Δ_ctrl':>8} {'Main Acc':>9} {'Δ_main':>8} {'γ_0.05':>8} {'Verdict'}")
    lines.append("-" * 105)

    for target in PYTHIA_TARGETS:
        clean_target = slug(target.hf_id)
        result_file = os.path.join(RUNS_DIR, f"mimir_github_{clean_target}", "results.json")
        data = load_json(result_file)

        acc_ctrl = 0.50
        adv_ctrl = 0.00

        nm_result_file = os.path.join(RUNS_DIR, f"mimir_github_nonmember_control_{clean_target}", "results.json")
        nm_metrics = score_test_metrics(load_json(nm_result_file), VANILLA_MIN_K_20)
        if nm_metrics:
            acc_ctrl = nm_metrics.get("accuracy", acc_ctrl)
            adv_ctrl = nm_metrics.get("shard_advantage", adv_ctrl)
        else:
            shuffled = shuffled_control_metrics(data, VANILLA_MIN_K_20)
            acc_ctrl = shuffled.get("accuracy", acc_ctrl)
            adv_ctrl = shuffled.get("shard_advantage", adv_ctrl)

        if data:
            result = score_test_metrics(data, VANILLA_MIN_K_20)
            acc_main = result.get("accuracy", 0.5)
            adv_main = result.get("shard_advantage", 0.0)

            lines.append(f"{target.parent_name:<15} {target.display_name:<32} {acc_ctrl:>8.1%} {adv_ctrl:>+8.3f} {acc_main:>8.1%} {adv_main:>+8.3f} {GAMMA_0_05:>8.3f} {verdict(adv_main, GAMMA_0_05)}")
        else:
            lines.append(f"{target.parent_name:<15} {target.display_name:<32} {'-- results not found --':>45}")

    lines.append("-" * 105)
    output_str = "\n".join(lines)
    print(output_str)

    write_text(OUTPUT_PATH, output_str)
    print(f"\nSaved compiled table to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""MIA experiment: LinguaCustodia/fin-pythia-1.4b (Main + Control)

Target : LinguaCustodia/fin-pythia-1.4b
Parent : EleutherAI/pythia-1.4b
Dataset: MIMIR GitHub (iamgroot42/mimir, config=github, split=ngram_13_0.2)
Primary: min_k_20_logprob

Run from the repo root:
    python scripts/experiments/run_target_linguacustodia_fin_pythia_1_4b.py
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
TARGET_MODEL  = "LinguaCustodia/fin-pythia-1.4b"
TARGET_SLUG   = "linguacustodia_fin_pythia_1_4b"
PARENT_MODEL  = "EleutherAI/pythia-1.4b"
PRIMARY_SCORE = "min_k_20_logprob"

# Data paths
TRAIN_FILE   = "data/processed/mimir_github/train.jsonl"
TEST_FILE    = "data/processed/mimir_github/test.jsonl"
CTRL_DIR     = "data/processed/mimir_github_nonmember_control_seed0"
CTRL_TRAIN   = f"{CTRL_DIR}/train.jsonl"
CTRL_TEST    = f"{CTRL_DIR}/test.jsonl"

# Output paths (Main)
MAIN_SCORES_DIR = f"data/scores/mimir_github_{TARGET_SLUG}"
MAIN_RUNS_DIR   = f"outputs/runs/mimir_github_{TARGET_SLUG}"
PARENT_RESULTS  = "outputs/runs/mimir_github_pythia_1_4b_mink/results.json"

# Output paths (Control)
CTRL_SCORES_DIR = f"data/scores/mimir_github_nonmember_control_seed0_{TARGET_SLUG}"
CTRL_RUNS_DIR   = f"outputs/runs/nonmember_control_seed0_{TARGET_SLUG}"

REPORT_DIR = f"outputs/reports/target_membership_advantage_{TARGET_SLUG}"


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    from huggingface_hub import snapshot_download
    print(f"\nEnsuring model is downloaded: {TARGET_MODEL}")
    snapshot_download(TARGET_MODEL)

    os.makedirs(os.path.join(REPO_ROOT, "outputs", "logs"), exist_ok=True)
    py = sys.executable

    # Check parent results
    if not os.path.isfile(os.path.join(REPO_ROOT, PARENT_RESULTS)):
        print(f"ERROR: parent results not found: {PARENT_RESULTS}")
        print("Run scripts/experiments/run_parent_pythia_1_4b.py first.")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Experiment 1: Main (Member vs Nonmember)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"MAIN EXPERIMENT: {TARGET_MODEL}")
    print("=" * 60)
    
    run([
        py, "scripts/scoring/extract_logprob_scores.py",
        "--model",      TARGET_MODEL,
        "--train-file", TRAIN_FILE,
        "--test-file",  TEST_FILE,
        "--output-dir", MAIN_SCORES_DIR,
        "--min-k-pcts", "5,10,20,40",
        "--batch-size", "4",
        "--tokenizer",  PARENT_MODEL,
    ])

    run([
        py, "scripts/experiments/run_mia_experiment.py",
        "--train-scores",  f"{MAIN_SCORES_DIR}/train_scores.jsonl",
        "--test-scores",   f"{MAIN_SCORES_DIR}/test_scores.jsonl",
        "--output-dir",    MAIN_RUNS_DIR,
        "--primary-score", PRIMARY_SCORE,
        "--model-label",   TARGET_MODEL,
        "--run-shuffled-control",
    ])

    # ------------------------------------------------------------------ #
    # Experiment 2: Control (Nonmember vs Nonmember)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"CONTROL EXPERIMENT: {TARGET_MODEL} (Nonmember vs Nonmember)")
    print("=" * 60)

    run([
        py, "scripts/scoring/extract_logprob_scores.py",
        "--model",      TARGET_MODEL,
        "--train-file", CTRL_TRAIN,
        "--test-file",  CTRL_TEST,
        "--output-dir", CTRL_SCORES_DIR,
        "--min-k-pcts", "5,10,20,40",
        "--batch-size", "4",
        "--tokenizer",  PARENT_MODEL,
    ])

    run([
        py, "scripts/experiments/run_mia_experiment.py",
        "--train-scores",  f"{CTRL_SCORES_DIR}/train_scores.jsonl",
        "--test-scores",   f"{CTRL_SCORES_DIR}/test_scores.jsonl",
        "--output-dir",    CTRL_RUNS_DIR,
        "--primary-score", PRIMARY_SCORE,
        "--model-label",   f"{TARGET_MODEL}-control",
        "--run-shuffled-control",
    ])

    # ------------------------------------------------------------------ #
    # Step 3: Comparison Report
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON REPORT")
    print("=" * 60)
    run([
        py, "scripts/reports/compare_parent_target_advantage.py",
        "--parent-results", PARENT_RESULTS,
        "--target-results", f"{MAIN_RUNS_DIR}/results.json",
        "--parent-model",   PARENT_MODEL,
        "--target-model",   TARGET_MODEL,
        "--output-dir",     REPORT_DIR,
    ])

    print("\n" + "=" * 60)
    print("ALL TARGET EXPERIMENTS DONE")
    print(f"  Main Results:    {MAIN_RUNS_DIR}/results.json")
    print(f"  Control Results: {CTRL_RUNS_DIR}/results.json")
    print(f"  Report:          {REPORT_DIR}/summary.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

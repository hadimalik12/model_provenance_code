"""MIA experiment: EleutherAI/pythia-6.9b (Parent Baseline + Control)

Model  : EleutherAI/pythia-6.9b
Dataset: MIMIR GitHub (iamgroot42/mimir, config=github, split=ngram_13_0.2)
Primary: min_k_20_logprob

Experiments
-----------
1. Main Experiment: Member vs. Nonmember
2. Control Experiment: Nonmember vs. Nonmember (null control)

Run from the repo root:
    python scripts/experiments/run_parent_pythia_6_9b.py
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
MODEL         = "EleutherAI/pythia-6.9b"
SLUG          = "pythia_6_9b_mink"
PRIMARY_SCORE = "min_k_20_logprob"

# Data paths
TRAIN_FILE   = "data/processed/mimir_github/train.jsonl"
TEST_FILE    = "data/processed/mimir_github/test.jsonl"
CTRL_DIR     = "data/processed/mimir_github_nonmember_control_seed0"
CTRL_TRAIN   = f"{CTRL_DIR}/train.jsonl"
CTRL_TEST    = f"{CTRL_DIR}/test.jsonl"

# Output paths (Main)
MAIN_SCORES_DIR = f"data/scores/mimir_github_{SLUG}"
MAIN_RUNS_DIR   = f"outputs/runs/mimir_github_{SLUG}"

# Output paths (Control)
CTRL_SCORES_DIR = f"data/scores/mimir_github_nonmember_control_seed0_{SLUG}"
CTRL_RUNS_DIR   = f"outputs/runs/nonmember_control_seed0_{SLUG}"


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    from huggingface_hub import snapshot_download
    print(f"\nEnsuring model is downloaded: {MODEL}")
    snapshot_download(MODEL)

    os.makedirs(os.path.join(REPO_ROOT, "outputs", "logs"), exist_ok=True)
    py = sys.executable

    # ------------------------------------------------------------------ #
    # Step 0 — Data Preparation
    # ------------------------------------------------------------------ #
    if not os.path.isfile(os.path.join(REPO_ROOT, TRAIN_FILE)):
        print("\nPreparing main dataset...")
        run([py, "scripts/data/prepare_mimir_github.py", 
             "--num-train-per-class", "500", "--num-test-per-class", "200"])
    
    if not os.path.isfile(os.path.join(REPO_ROOT, CTRL_TRAIN)):
        print("\nPreparing nonmember control dataset...")
        run([py, "scripts/data/prepare_mimir_github_nonmember_control.py",
             "--num-train-per-class", "170", "--num-test-per-class", "200", "--seed", "0"])

    # ------------------------------------------------------------------ #
    # Experiment 1: Main (Member vs Nonmember)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"MAIN EXPERIMENT: {MODEL}")
    print("=" * 60)
    
    # Step 1.1: Scoring
    run([
        py, "scripts/scoring/extract_logprob_scores.py",
        "--model",      MODEL,
        "--train-file", TRAIN_FILE,
        "--test-file",  TEST_FILE,
        "--output-dir", MAIN_SCORES_DIR,
        "--min-k-pcts", "5,10,20,40",
        "--batch-size", "2",
    ])

    # Step 1.2: MIA
    run([
        py, "scripts/experiments/run_mia_experiment.py",
        "--train-scores",  f"{MAIN_SCORES_DIR}/train_scores.jsonl",
        "--test-scores",   f"{MAIN_SCORES_DIR}/test_scores.jsonl",
        "--output-dir",    MAIN_RUNS_DIR,
        "--primary-score", PRIMARY_SCORE,
        "--model-label",   MODEL,
        "--run-shuffled-control",
    ])

    # ------------------------------------------------------------------ #
    # Experiment 2: Control (Nonmember vs Nonmember)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print(f"CONTROL EXPERIMENT: {MODEL} (Nonmember vs Nonmember)")
    print("=" * 60)

    # Step 2.1: Scoring
    run([
        py, "scripts/scoring/extract_logprob_scores.py",
        "--model",      MODEL,
        "--train-file", CTRL_TRAIN,
        "--test-file",  CTRL_TEST,
        "--output-dir", CTRL_SCORES_DIR,
        "--min-k-pcts", "5,10,20,40",
        "--batch-size", "2",
    ])

    # Step 2.2: MIA
    run([
        py, "scripts/experiments/run_mia_experiment.py",
        "--train-scores",  f"{CTRL_SCORES_DIR}/train_scores.jsonl",
        "--test-scores",   f"{CTRL_SCORES_DIR}/test_scores.jsonl",
        "--output-dir",    CTRL_RUNS_DIR,
        "--primary-score", PRIMARY_SCORE,
        "--model-label",   f"{MODEL}-control",
        "--run-shuffled-control",
    ])

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS DONE")
    print(f"  Main Results:    {MAIN_RUNS_DIR}/results.json")
    print(f"  Control Results: {CTRL_RUNS_DIR}/results.json")
    print("=" * 60)


if __name__ == "__main__":
    main()

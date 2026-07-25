#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --account=gts-vzikas3
#SBATCH -o replicate_table_2_12b_%j.log
#SBATCH -e replicate_table_2_12b_%j.log

echo "=========================================================="
echo " Starting Slurm Batch Job for Table 2 (12B Models)"
echo " Date: $(date)"
echo " Hostname: $(hostname)"
echo "=========================================================="

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"

echo "=== Installing environment on compute node local /tmp ==="
bash scripts/local_scripts/install_pace.sh

echo "=== Loading modules & activating environment ==="
module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/model_provenance_venv
unset -f python || true
export HF_HOME="$REPO_ROOT/data/.hf_cache"
export HF_HUB_CACHE="$REPO_ROOT/data/.hf_cache"

DATA_DIR="data/processed/mimir_github_12b"

# Step 1: Prepare MIMIR GitHub dataset (fresh 64-word split)
echo "=== Step 1: Preparing MIMIR GitHub Dataset (64 words) ==="
python scripts/data/prepare_mimir_github.py \
  --num-train-member 560 \
  --num-train-nonmember 340 \
  --num-test-per-class 400 \
  --max-words 64 --min-words 8 \
  --seed 0 \
  --token "${HF_TOKEN:-}" \
  --output-dir "$DATA_DIR"

if [ ! -f "$DATA_DIR/train.jsonl" ]; then
  echo "ERROR: Dataset preparation failed. $DATA_DIR/train.jsonl was not created."
  echo "Please make sure your HuggingFace token is exported: export HF_TOKEN='hf_...'"
  exit 1
fi

NM_DATA_DIR="data/processed/mimir_github_nonmember_control_12b"
echo "=== Step 1b: Preparing Nonmember Control Dataset ==="
python scripts/data/prepare_mimir_github_nonmember_control.py \
  --num-train-per-class 450 \
  --num-test-per-class 400 \
  --seed 0 \
  --output-dir "$NM_DATA_DIR"

# Step 2: List of 12B Parent and Target Models
TARGET_MODELS=(
  "EleutherAI/pythia-12b"
  "lomahony/eleuther-pythia12b-hh-dpo"
  "lomahony/eleuther-pythia12b-hh-sft"
)

echo "=== Step 2 & 3: Augmented Log-Prob Scoring & MIA Experiment (12B Models) ==="

for TARGET in "${TARGET_MODELS[@]}"; do
  CLEAN_TARGET=$(echo "$TARGET" | sed 's/\//__/g')
  SCORE_DIR="data/scores/mimir_github_${CLEAN_TARGET}"
  RUN_DIR="outputs/runs/mimir_github_${CLEAN_TARGET}"
  NM_SCORE_DIR="data/scores/mimir_github_nonmember_control_${CLEAN_TARGET}"
  NM_RUN_DIR="outputs/runs/mimir_github_nonmember_control_${CLEAN_TARGET}"

  echo "----------------------------------------------------------"
  echo "Scoring & Auditing Target Model: $TARGET"
  echo "----------------------------------------------------------"

  # 1. 4-View Augmented Scoring (Main Dataset)
  python scripts/scoring/extract_logprob_scores.py \
    --model "$TARGET" \
    --train-file "$DATA_DIR/train.jsonl" \
    --test-file "$DATA_DIR/test.jsonl" \
    --output-dir "$SCORE_DIR" \
    --min-k-pcts 20 \
    --n-aug 4 \
    --dtype float16

  # 1b. 4-View Augmented Scoring (Nonmember Control Dataset)
  python scripts/scoring/extract_logprob_scores.py \
    --model "$TARGET" \
    --train-file "$NM_DATA_DIR/train.jsonl" \
    --test-file "$NM_DATA_DIR/test.jsonl" \
    --output-dir "$NM_SCORE_DIR" \
    --min-k-pcts 20 \
    --n-aug 4 \
    --dtype float16

  # Determine parent model and check if parent results are available
  PARENT_MODEL="EleutherAI/pythia-12b"
  CLEAN_PARENT=$(echo "$PARENT_MODEL" | sed 's/\//__/g')
  PARENT_RESULTS="outputs/runs/mimir_github_${CLEAN_PARENT}/results.json"

  PARENT_ARG=""
  if [ -f "$PARENT_RESULTS" ] && [ "$TARGET" != "$PARENT_MODEL" ]; then
    PARENT_ARG="--parent-results $PARENT_RESULTS"
  fi

  # 2. Threshold Calibration & Advantage Evaluation (Main Dataset)
  if [ -f "$SCORE_DIR/train_scores.jsonl" ] && [ -f "$SCORE_DIR/test_scores.jsonl" ]; then
    python scripts/experiments/run_mia_experiment.py \
      --train-scores "$SCORE_DIR/train_scores.jsonl" \
      --test-scores "$SCORE_DIR/test_scores.jsonl" \
      --output-dir "$RUN_DIR" \
      --score-keys min_k_20_logprob \
      --primary-score min_k_20_logprob \
      --run-shuffled-control \
      $PARENT_ARG
  else
    echo "Warning: Score files for $TARGET not found in $SCORE_DIR. Skipping experiment."
  fi

  # 2b. Threshold Calibration & Advantage Evaluation (Nonmember Control Dataset)
  if [ -f "$NM_SCORE_DIR/train_scores.jsonl" ] && [ -f "$NM_SCORE_DIR/test_scores.jsonl" ]; then
    python scripts/experiments/run_mia_experiment.py \
      --train-scores "$NM_SCORE_DIR/train_scores.jsonl" \
      --test-scores "$NM_SCORE_DIR/test_scores.jsonl" \
      --output-dir "$NM_RUN_DIR" \
      --score-keys min_k_20_logprob \
      --primary-score min_k_20_logprob
  fi
done

# Step 4: Compile Final Table 2
echo "=== Step 4: Compiling Replicated Table 2 (Augmented) ==="
python scripts/reports/report_table_2.py

# Step 5: Clean up processed dataset directory
echo "=== Step 5: Cleaning up processed dataset directory ($DATA_DIR) ==="
rm -rf "$DATA_DIR"

echo "=========================================================="
echo " Table 2 (12B Models) Job Completed Successfully!"
echo "=========================================================="

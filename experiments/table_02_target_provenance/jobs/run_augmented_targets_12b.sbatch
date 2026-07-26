#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --account=gts-vzikas3
#SBATCH --job-name=table02_targets_12b
#SBATCH --output=experiments/table_02_target_provenance/logs/%x_%j.out
#SBATCH --error=experiments/table_02_target_provenance/logs/%x_%j.err

echo "=========================================================="
echo " Starting Slurm Batch Job for Table 2 (12B Models)"
echo " Date: $(date)"
echo " Hostname: $(hostname)"
echo "=========================================================="

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/scripts/cluster/install_pace.sh" ]; then
  echo "ERROR: missing $REPO_ROOT/scripts/cluster/install_pace.sh" >&2
  exit 1
fi
if [ -f "$REPO_ROOT/.env" ]; then
  source "$REPO_ROOT/.env"
fi

echo "=== Installing environment on compute node local /tmp ==="
for ATTEMPT in 1 2 3; do
  bash scripts/cluster/install_pace.sh && break
  if [ "$ATTEMPT" -eq 3 ]; then
    echo "ERROR: environment setup failed after $ATTEMPT attempts" >&2
    exit 1
  fi
  echo "Environment setup failed; retrying ($((ATTEMPT + 1))/3)..." >&2
  sleep 10
done

echo "=== Loading modules & activating environment ==="
module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/model_provenance_venv
unset -f python || true
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$REPO_ROOT/.cache/huggingface"

ARTIFACT_ROOT="artifacts/table_02_target_provenance/augmented_seed0"
DATA_DIR="$ARTIFACT_ROOT/prepared/mimir_github_12b"

# Step 1: Prepare MIMIR GitHub dataset (fresh 64-word split)
echo "=== Step 1: Preparing MIMIR GitHub Dataset (64 words) ==="
python scripts/data/prepare_mimir_domain.py \
  --split ngram_13_0.2 \
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

NM_DATA_DIR="$ARTIFACT_ROOT/prepared/nonmember_control_12b"
echo "=== Step 1b: Preparing Nonmember Control Dataset ==="
python scripts/data/prepare_mimir_domain_nonmember_control.py \
  --ngram-split ngram_13_0.2 \
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

for TARGET in "${TARGET_MODELS[@]}"; do
  python scripts/setup/verify_hf_model_access.py --model "$TARGET"
done

echo "=== Step 2 & 3: Augmented Log-Prob Scoring & MIA Experiment (12B Models) ==="

for TARGET in "${TARGET_MODELS[@]}"; do
  CLEAN_TARGET=$(echo "$TARGET" | sed 's/\//__/g')
  SCORE_DIR="$ARTIFACT_ROOT/scores/mimir_github_aug_${CLEAN_TARGET}"
  RUN_DIR="$ARTIFACT_ROOT/results/mimir_github_aug_${CLEAN_TARGET}"
  NM_SCORE_DIR="$ARTIFACT_ROOT/scores/mimir_github_nonmember_control_aug_${CLEAN_TARGET}"
  NM_RUN_DIR="$ARTIFACT_ROOT/results/mimir_github_nonmember_control_aug_${CLEAN_TARGET}"

  echo "----------------------------------------------------------"
  echo "Scoring & Auditing Target Model: $TARGET"
  echo "----------------------------------------------------------"

  # 1. 4-View Augmented Scoring (Main Dataset)
  python scripts/scoring/score_causal_lm_code_augmented.py \
    --model "$TARGET" \
    --train-file "$DATA_DIR/train.jsonl" \
    --test-file "$DATA_DIR/test.jsonl" \
    --output-dir "$SCORE_DIR" \
    --min-k-pcts 20 \
    --n-aug 4 \
    --dtype float16

  # 1b. 4-View Augmented Scoring (Nonmember Control Dataset)
  python scripts/scoring/score_causal_lm_code_augmented.py \
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
  PARENT_RESULTS="$ARTIFACT_ROOT/results/mimir_github_aug_${CLEAN_PARENT}/results.json"

  PARENT_ARG=""
  if [ -f "$PARENT_RESULTS" ] && [ "$TARGET" != "$PARENT_MODEL" ]; then
    PARENT_ARG="--parent-results $PARENT_RESULTS"
  fi

  # 2. Threshold Calibration & Advantage Evaluation (Main Dataset)
  if [ -f "$SCORE_DIR/train_scores.jsonl" ] && [ -f "$SCORE_DIR/test_scores.jsonl" ]; then
    python scripts/audit/run_threshold_audit.py \
      --train-scores "$SCORE_DIR/train_scores.jsonl" \
      --test-scores "$SCORE_DIR/test_scores.jsonl" \
      --output-dir "$RUN_DIR" \
      --score-keys aug_min_k_20_logprob_mean \
      --primary-score aug_min_k_20_logprob_mean \
      --run-shuffled-control \
      $PARENT_ARG
  else
    echo "Warning: Score files for $TARGET not found in $SCORE_DIR. Skipping experiment."
  fi

  # 2b. Threshold Calibration & Advantage Evaluation (Nonmember Control Dataset)
  if [ -f "$NM_SCORE_DIR/train_scores.jsonl" ] && [ -f "$NM_SCORE_DIR/test_scores.jsonl" ]; then
    python scripts/audit/run_threshold_audit.py \
      --train-scores "$NM_SCORE_DIR/train_scores.jsonl" \
      --test-scores "$NM_SCORE_DIR/test_scores.jsonl" \
      --output-dir "$NM_RUN_DIR" \
      --score-keys aug_min_k_20_logprob_mean \
      --primary-score aug_min_k_20_logprob_mean
  fi
done

# Step 4: Compile Final Table 2
echo "=== Step 4: Compiling Replicated Table 2 (Augmented) ==="
python experiments/table_02_target_provenance/reports/compile_augmented.py

# Step 5: Clean up processed dataset directory
echo "=========================================================="
echo " Table 2 (12B Models) Job Completed Successfully!"
echo "=========================================================="

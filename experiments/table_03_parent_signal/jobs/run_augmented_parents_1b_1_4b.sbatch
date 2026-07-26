#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --account=gts-vzikas3
#SBATCH --job-name=table03_parents_1b_1_4b
#SBATCH --output=experiments/table_03_parent_signal/logs/%x_%j.out
#SBATCH --error=experiments/table_03_parent_signal/logs/%x_%j.err

echo "=========================================================="
echo " Table 4: Score Selection (Pythia-1B & 1.4B) on GitHub"
echo " Date: $(date)"
echo " Hostname: $(hostname)"
echo "=========================================================="

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
  source "$REPO_ROOT/.env"
fi

echo "=== Installing environment on compute node local /tmp ==="
MAX_RETRIES=3
for attempt in $(seq 1 $MAX_RETRIES); do
  echo "Install attempt $attempt of $MAX_RETRIES"
  bash scripts/cluster/install_pace.sh && break
  echo "Install attempt $attempt failed, retrying in 10s..."
  sleep 10
done

echo "=== Loading modules & activating environment ==="
module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/model_provenance_venv
unset -f python || true

# Verify environment is working
python -c "import torch; print(f'torch OK: {torch.__version__}')" || { echo 'ERROR: Environment not working'; exit 1; }
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$REPO_ROOT/.cache/huggingface"

# ================================================================== #
# Step 1: Prepare MIMIR GitHub dataset
# ================================================================== #
ARTIFACT_ROOT="artifacts/table_03_parent_signal/augmented_seed0"
DATA_DIR="$ARTIFACT_ROOT/prepared/github_1b_1.4b"
NM_DATA_DIR="$ARTIFACT_ROOT/prepared/nonmember_control_1b_1.4b"

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

echo "=== Step 1b: Preparing Nonmember Control Dataset ==="
python scripts/data/prepare_mimir_domain_nonmember_control.py \
  --ngram-split ngram_13_0.2 \
  --num-train-per-class 450 \
  --num-test-per-class 400 \
  --seed 0 \
  --output-dir "$NM_DATA_DIR"

# ================================================================== #
# Step 2: Score & run experiments
# ================================================================== #
PARENT_MODELS=(
  "EleutherAI/pythia-1b"
  "EleutherAI/pythia-1.4b"
)

for MODEL in "${PARENT_MODELS[@]}"; do
  python scripts/setup/verify_hf_model_access.py --model "$MODEL"
done

for MODEL in "${PARENT_MODELS[@]}"; do
  CLEAN_MODEL=$(echo "$MODEL" | sed 's/\//__/g')
  SCORE_DIR="$ARTIFACT_ROOT/scores/github_${CLEAN_MODEL}"
  RUN_DIR="$ARTIFACT_ROOT/results/github_${CLEAN_MODEL}"
  NM_SCORE_DIR="$ARTIFACT_ROOT/scores/nonmember_control_${CLEAN_MODEL}"
  NM_RUN_DIR="$ARTIFACT_ROOT/results/nonmember_control_${CLEAN_MODEL}"

  echo "----------------------------------------------------------"
  echo "Scoring Model: $MODEL"
  echo "----------------------------------------------------------"

  # Score on main dataset with augmentation
  python scripts/scoring/score_causal_lm_domain_augmented.py \
    --model "$MODEL" \
    --train-file "$DATA_DIR/train.jsonl" \
    --test-file "$DATA_DIR/test.jsonl" \
    --output-dir "$SCORE_DIR" \
    --domain github \
    --n-aug 4 \
    --min-k-pcts 20 \
    --batch-size 4 \
    --dtype float16

  # Score on nonmember control dataset with augmentation
  if [ -f "$NM_DATA_DIR/train.jsonl" ]; then
    python scripts/scoring/score_causal_lm_domain_augmented.py \
      --model "$MODEL" \
      --train-file "$NM_DATA_DIR/train.jsonl" \
      --test-file "$NM_DATA_DIR/test.jsonl" \
      --output-dir "$NM_SCORE_DIR" \
      --domain github \
      --n-aug 4 \
      --min-k-pcts 20 \
      --batch-size 4 \
      --dtype float16
  fi

  # Run MIA experiment on main dataset
  if [ -f "$SCORE_DIR/train_scores.jsonl" ] && [ -f "$SCORE_DIR/test_scores.jsonl" ]; then
    python scripts/audit/run_threshold_audit.py \
      --train-scores "$SCORE_DIR/train_scores.jsonl" \
      --test-scores "$SCORE_DIR/test_scores.jsonl" \
      --output-dir "$RUN_DIR" \
      --score-keys min_k_20_logprob,aug_min_k_20_logprob_mean \
      --primary-score min_k_20_logprob \
      --run-shuffled-control
  fi

  # Run MIA experiment on nonmember control dataset
  if [ -f "$NM_SCORE_DIR/train_scores.jsonl" ] && [ -f "$NM_SCORE_DIR/test_scores.jsonl" ]; then
    python scripts/audit/run_threshold_audit.py \
      --train-scores "$NM_SCORE_DIR/train_scores.jsonl" \
      --test-scores "$NM_SCORE_DIR/test_scores.jsonl" \
      --output-dir "$NM_RUN_DIR" \
      --score-keys min_k_20_logprob,aug_min_k_20_logprob_mean \
      --primary-score min_k_20_logprob
  fi
done

echo "=== Compiling Table 4 (partial) ==="
python experiments/table_03_parent_signal/reports/compile.py

echo "=== Cleaning up processed data ==="
rm -rf "$DATA_DIR" "$NM_DATA_DIR"

echo "=========================================================="
echo " Table 4 (1B & 1.4B) Job Completed Successfully!"
echo "=========================================================="

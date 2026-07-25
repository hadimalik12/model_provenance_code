#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --account=gts-vzikas3
#SBATCH -o replicate_table_5_aug_%j.log
#SBATCH -e replicate_table_5_aug_%j.log

echo "=========================================================="
echo " Table 5: Cross-Domain Signal with Augmentation"
echo " Date: $(date)"
echo " Hostname: $(hostname)"
echo "=========================================================="

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
  source "$REPO_ROOT/.env"
fi

echo "=== Installing environment on compute node local /tmp ==="
MAX_RETRIES=3
for attempt in $(seq 1 $MAX_RETRIES); do
  echo "Install attempt $attempt of $MAX_RETRIES"
  bash scripts/local_scripts/install_pace.sh && break
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
export HF_HOME="$REPO_ROOT/data/.hf_cache"
export HF_HUB_CACHE="$REPO_ROOT/data/.hf_cache"

# HuggingFace config names (these map to the filenames on HF)
DOMAINS_HF=("arxiv" "dm_mathematics" "github" "pile_cc" "wikipedia_(en)")
# Clean display/filesystem names
DOMAINS_CLEAN=("arxiv" "dm_mathematics" "github" "pile_cc" "wikipedia_en")

PARENT_MODEL="EleutherAI/pythia-1.4b"
TARGET_MODEL="lomahony/pythia-1.4b-helpful-sft"
PARENT_SLUG="parent_pythia_1_4b"
TARGET_SLUG="target_lomahony_pythia_1_4b_helpful_sft"

PROCESSED_ROOT="data/processed/mimir_domains"
VANILLA_SCORE_ROOT="data/scores/mimir_domains"
AUG_SCORE_ROOT="data/scores/mimir_domains_aug"
VANILLA_RUN_ROOT="outputs/runs/mimir_domains"
AUG_RUN_ROOT="outputs/runs/mimir_domains_aug"

N_TRAIN=200
N_TEST=200

# ------------------------------------------------------------------ #
# Step 1: Prepare all domains
# ------------------------------------------------------------------ #
echo ""
echo "=== Step 1: Preparing all MIMIR domains ==="

for i in "${!DOMAINS_HF[@]}"; do
  HF_DOMAIN="${DOMAINS_HF[$i]}"
  CLEAN_DOMAIN="${DOMAINS_CLEAN[$i]}"
  DATA_DIR="$PROCESSED_ROOT/$CLEAN_DOMAIN"
  echo "Preparing domain: $HF_DOMAIN -> $DATA_DIR"
  python scripts/data/prepare_mimir_github.py \
    --config "$HF_DOMAIN" \
    --split none \
    --num-train-per-class $N_TRAIN \
    --num-test-per-class $N_TEST \
    --seed 0 \
    --allow-overlap \
    --token "${HF_TOKEN:-}" \
    --output-dir "$DATA_DIR"
done

# ------------------------------------------------------------------ #
# Step 2: Score & Run Experiments (Vanilla & Augmented)
# ------------------------------------------------------------------ #
echo ""
echo "=== Step 2: Scoring & Evaluating ==="

MODELS=("$PARENT_MODEL" "$TARGET_MODEL")
SLUGS=("$PARENT_SLUG" "$TARGET_SLUG")

for i in "${!DOMAINS_CLEAN[@]}"; do
  HF_DOMAIN="${DOMAINS_HF[$i]}"
  CLEAN_DOMAIN="${DOMAINS_CLEAN[$i]}"
  echo "----------------------------------------------------------"
  echo "Domain: $CLEAN_DOMAIN"
  echo "----------------------------------------------------------"
  DATA_DIR="$PROCESSED_ROOT/$CLEAN_DOMAIN"

  if [ ! -f "$DATA_DIR/train.jsonl" ]; then
    echo "  WARNING: Data not found at $DATA_DIR/train.jsonl, skipping."
    continue
  fi

  for j in "${!MODELS[@]}"; do
    MODEL="${MODELS[$j]}"
    SLUG="${SLUGS[$j]}"
    echo "  Model: $MODEL ($SLUG)"

    # Vanilla Scoring
    VANILLA_SCORE_DIR="$VANILLA_SCORE_ROOT/$CLEAN_DOMAIN/$SLUG"
    if [ ! -f "$VANILLA_SCORE_DIR/test_scores.jsonl" ]; then
      echo "    [Vanilla] Scoring..."
      python scripts/scoring/extract_logprob_scores.py \
        --model "$MODEL" \
        --train-file "$DATA_DIR/train.jsonl" \
        --test-file "$DATA_DIR/test.jsonl" \
        --output-dir "$VANILLA_SCORE_DIR" \
        --min-k-pcts 20 \
        --batch-size 4 \
        --dtype float16
    else
      echo "    [Vanilla] Scores already exist."
    fi

    # Vanilla Experiment
    VANILLA_RUN_DIR="$VANILLA_RUN_ROOT/$CLEAN_DOMAIN/$SLUG"
    if [ -f "$VANILLA_SCORE_DIR/train_scores.jsonl" ] && [ -f "$VANILLA_SCORE_DIR/test_scores.jsonl" ]; then
      echo "    [Vanilla] Running experiment..."
      python scripts/experiments/run_mia_experiment.py \
        --train-scores "$VANILLA_SCORE_DIR/train_scores.jsonl" \
        --test-scores "$VANILLA_SCORE_DIR/test_scores.jsonl" \
        --output-dir "$VANILLA_RUN_DIR" \
        --score-keys min_k_20_logprob \
        --primary-score min_k_20_logprob
    fi

    # Augmented Scoring
    AUG_SCORE_DIR="$AUG_SCORE_ROOT/$CLEAN_DOMAIN/$SLUG"
    if [ ! -f "$AUG_SCORE_DIR/test_scores.jsonl" ]; then
      echo "    [Augmented] Scoring..."
      python scripts/scoring/extract_logprob_scores_domain_aug.py \
        --model "$MODEL" \
        --train-file "$DATA_DIR/train.jsonl" \
        --test-file "$DATA_DIR/test.jsonl" \
        --output-dir "$AUG_SCORE_DIR" \
        --domain "$CLEAN_DOMAIN" \
        --min-k-pcts 20 \
        --n-aug 4 \
        --dtype float16
    else
      echo "    [Augmented] Scores already exist."
    fi

    # Augmented Experiment
    AUG_RUN_DIR="$AUG_RUN_ROOT/$CLEAN_DOMAIN/$SLUG"
    if [ -f "$AUG_SCORE_DIR/train_scores.jsonl" ] && [ -f "$AUG_SCORE_DIR/test_scores.jsonl" ]; then
      echo "    [Augmented] Running experiment..."
      python scripts/experiments/run_mia_experiment.py \
        --train-scores "$AUG_SCORE_DIR/train_scores.jsonl" \
        --test-scores "$AUG_SCORE_DIR/test_scores.jsonl" \
        --output-dir "$AUG_RUN_DIR" \
        --score-keys aug_min_k_20_logprob_mean \
        --primary-score aug_min_k_20_logprob_mean
    fi
  done
done

# ------------------------------------------------------------------ #
# Step 3: Compile Table 5
# ------------------------------------------------------------------ #
echo ""
echo "=== Step 3: Compiling Table 5 ==="
python scripts/reports/report_table_5_aug.py

echo "=========================================================="
echo " Table 5 Job Completed Successfully!"
echo "=========================================================="

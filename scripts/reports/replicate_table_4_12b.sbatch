#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 8:00:00
#SBATCH --account=gts-vzikas3
#SBATCH -o replicate_table_4_12b_%j.log
#SBATCH -e replicate_table_4_12b_%j.log

echo "=========================================================="
echo " Table 4: Score Selection (Pythia-12B) on GitHub"
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

MODEL="EleutherAI/pythia-12b"
DATA_DIR="data/processed/table4_github_12b"
NM_DATA_DIR="data/processed/table4_github_ctrl_12b"
CLEAN_MODEL=$(echo "$MODEL" | sed 's/\//__/g')
SCORE_DIR="data/scores/table4_github_${CLEAN_MODEL}"
RUN_DIR="outputs/runs/table4_github_${CLEAN_MODEL}"
NM_SCORE_DIR="data/scores/table4_github_ctrl_${CLEAN_MODEL}"
NM_RUN_DIR="outputs/runs/table4_github_ctrl_${CLEAN_MODEL}"

# Step 1: Prepare data
echo "=== Step 1: Preparing MIMIR GitHub Dataset ==="
python scripts/data/prepare_mimir_github.py \
  --num-train-member 560 \
  --num-train-nonmember 340 \
  --num-test-per-class 400 \
  --max-words 64 --min-words 8 \
  --seed 0 \
  --token "${HF_TOKEN:-}" \
  --output-dir "$DATA_DIR"

python scripts/data/prepare_mimir_github_nonmember_control.py \
  --num-train-per-class 450 \
  --num-test-per-class 400 \
  --seed 0 \
  --output-dir "$NM_DATA_DIR"

# Step 2: Score with augmentation
echo "=== Step 2: Scoring Pythia-12B (float16, mem=128G) ==="
python scripts/scoring/extract_logprob_scores_domain_aug.py \
  --model "$MODEL" \
  --train-file "$DATA_DIR/train.jsonl" \
  --test-file "$DATA_DIR/test.jsonl" \
  --output-dir "$SCORE_DIR" \
  --domain github \
  --n-aug 4 \
  --min-k-pcts 20 \
  --batch-size 1 \
  --dtype float16

if [ -f "$NM_DATA_DIR/train.jsonl" ]; then
  python scripts/scoring/extract_logprob_scores_domain_aug.py \
    --model "$MODEL" \
    --train-file "$NM_DATA_DIR/train.jsonl" \
    --test-file "$NM_DATA_DIR/test.jsonl" \
    --output-dir "$NM_SCORE_DIR" \
    --domain github \
    --n-aug 4 \
    --min-k-pcts 20 \
    --batch-size 1 \
    --dtype float16
fi

# Step 3: Run experiments
echo "=== Step 3: Running MIA Experiments ==="
if [ -f "$SCORE_DIR/train_scores.jsonl" ] && [ -f "$SCORE_DIR/test_scores.jsonl" ]; then
  python scripts/experiments/run_mia_experiment.py \
    --train-scores "$SCORE_DIR/train_scores.jsonl" \
    --test-scores "$SCORE_DIR/test_scores.jsonl" \
    --output-dir "$RUN_DIR" \
    --score-keys min_k_20_logprob,aug_min_k_20_logprob_mean \
    --primary-score min_k_20_logprob \
    --run-shuffled-control
fi

if [ -f "$NM_SCORE_DIR/train_scores.jsonl" ] && [ -f "$NM_SCORE_DIR/test_scores.jsonl" ]; then
  python scripts/experiments/run_mia_experiment.py \
    --train-scores "$NM_SCORE_DIR/train_scores.jsonl" \
    --test-scores "$NM_SCORE_DIR/test_scores.jsonl" \
    --output-dir "$NM_RUN_DIR" \
    --score-keys min_k_20_logprob,aug_min_k_20_logprob_mean \
    --primary-score min_k_20_logprob
fi

# Step 4: Compile
echo "=== Step 4: Compiling Table 4 ==="
python scripts/reports/report_table_4.py

rm -rf "$DATA_DIR" "$NM_DATA_DIR"

echo "=========================================================="
echo " Table 4 (12B) Job Completed Successfully!"
echo "=========================================================="

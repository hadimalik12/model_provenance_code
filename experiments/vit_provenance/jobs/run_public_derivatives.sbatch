#!/bin/bash
# Submitted Section 5.2.2 / Appendix Table 8 public-ViT audit.
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --job-name=vit_public_audit
#SBATCH --output=experiments/vit_provenance/logs/%x_%j.out
#SBATCH --error=experiments/vit_provenance/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

bash scripts/cluster/install_pace.sh
module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv_pace}"
conda activate "$ENV_DIR"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"

ROOT="artifacts/vit_provenance/public_derivatives_seed0"
DATA="$ROOT/prepared/imagenet_shard"
SCORER="experiments/vit_provenance/scoring/score_clv.py"

if [ ! -f "$DATA/manifest.json" ]; then
  python experiments/vit_provenance/runners/prepare_shards.py \
    --train-shard-idxs 0,1,2,3,4 \
    --val-shard-idxs 0,1 \
    --members-per-class 50 \
    --nonmembers-per-class 50 \
    --max-classes 77 \
    --train-frac 0.5 \
    --seed 0 \
    --output-dir "$DATA"
fi

MODELS=(
  "parent|google/vit-base-patch16-224"
  "pets|ISxOdin/vit-base-oxford-iiit-pets"
  "cifar10|MatanBT/vit-base-patch16-224-cifar10"
  "cifar100|MatanBT/vit-base-patch16-224-cifar100"
  "eurosat|Abhiram4/vit-base-patch16-224-finetuned-eurosat"
  "celeba|MatanBT/vit-base-patch16-224-celeba-smiling"
  "har|rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
)

for ENTRY in "${MODELS[@]}"; do
  NAME="${ENTRY%%|*}"
  MODEL="${ENTRY#*|}"
  python scripts/setup/verify_hf_model_access.py --model "$MODEL"
  SCORE_DIR="$ROOT/scores/$NAME"
  RESULT_DIR="$ROOT/results/$NAME"
  if [ ! -f "$SCORE_DIR/manifest.json" ]; then
    python "$SCORER" \
      --target-model "$MODEL" \
      --train-file "$DATA/train.jsonl" \
      --test-file "$DATA/test.jsonl" \
      --output-dir "$SCORE_DIR" \
      --batch-size "${VIT_BATCH_SIZE:-16}"
  fi
  python scripts/audit/run_threshold_audit.py \
    --train-scores "$SCORE_DIR/train_scores.jsonl" \
    --test-scores "$SCORE_DIR/test_scores.jsonl" \
    --output-dir "$RESULT_DIR" \
    --score-keys clv_score \
    --primary-score clv_score \
    --model-label "$MODEL" \
    --run-shuffled-control
done

RANDOM_SCORES="$ROOT/scores/random_init"
if [ ! -f "$RANDOM_SCORES/manifest.json" ]; then
  python "$SCORER" \
    --target-model google/vit-base-patch16-224 \
    --random-init --random-seed 0 \
    --train-file "$DATA/train.jsonl" \
    --test-file "$DATA/test.jsonl" \
    --output-dir "$RANDOM_SCORES" \
    --batch-size "${VIT_BATCH_SIZE:-16}"
fi
python scripts/audit/run_threshold_audit.py \
  --train-scores "$RANDOM_SCORES/train_scores.jsonl" \
  --test-scores "$RANDOM_SCORES/test_scores.jsonl" \
  --output-dir "$ROOT/results/random_init" \
  --score-keys clv_score \
  --primary-score clv_score \
  --model-label google/vit-base-patch16-224::random_init_seed0 \
  --run-shuffled-control

python experiments/vit_provenance/reports/compile_public_derivatives.py

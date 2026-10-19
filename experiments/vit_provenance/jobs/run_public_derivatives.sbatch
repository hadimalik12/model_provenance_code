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

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
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
  "parent|google/vit-base-patch16-224|3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"
  "pets|ISxOdin/vit-base-oxford-iiit-pets|6fcb4ce433d4b43a19bd9b4d2d906b264685d45"
  "cifar10|MatanBT/vit-base-patch16-224-cifar10|7e0c5d0d3d7c449c3f25f1351abd50733c0a8cb9"
  "cifar100|MatanBT/vit-base-patch16-224-cifar100|853d63daa844810c636fd3a1216619e7a3042dbe"
  "eurosat|Abhiram4/vit-base-patch16-224-finetuned-eurosat|da2af5ab8f4c91b47c34dc30f03a8b5ba704352f"
  "celeba|MatanBT/vit-base-patch16-224-celeba-smiling|688ae3b5225e4def61cc9500994e107141cefdb2"
  "har|rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224|c16647aa649284692d1deacc52146cd631cba80a"
)

for ENTRY in "${MODELS[@]}"; do
  NAME="${ENTRY%%|*}"
  MODEL_AND_REVISION="${ENTRY#*|}"
  MODEL="${MODEL_AND_REVISION%%|*}"
  REVISION="${MODEL_AND_REVISION#*|}"
  python scripts/setup/verify_hf_model_access.py --model "$MODEL@$REVISION"
  SCORE_DIR="$ROOT/scores/$NAME"
  RESULT_DIR="$ROOT/results/$NAME"
  if [ ! -f "$SCORE_DIR/manifest.json" ]; then
    python "$SCORER" \
      --target-model "$MODEL" \
      --target-revision "$REVISION" \
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
    --target-revision 3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3 \
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

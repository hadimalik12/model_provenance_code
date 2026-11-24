#!/bin/bash
# Requires the exact prepared ImageNet split used by the public ViT audit.
#SBATCH --job-name=imagenet-split-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00
#SBATCH --output=experiments/vit_provenance/logs/%x-%j.out
#SBATCH --error=experiments/vit_provenance/logs/%x-%j.err

set -euo pipefail
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
module load pytorch/25 anaconda3
eval "$(conda shell.bash hook)"
ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv_pace}"
conda activate "$ENV_DIR"
PYTHON="$ENV_DIR/bin/python"
test -x "$PYTHON"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
PYTHON="$PYTHON" bash scripts/cluster/ensure_split_diagnostic_deps.sh

EXTRA=()
OUTPUT_DIR="${SPLIT_DIAG_OUTPUT_DIR:-artifacts/split_diagnostics/aistats27_q3/imagenet_production}"
if [ "${SPLIT_DIAG_SMOKE:-0}" = "1" ]; then
  EXTRA+=(--max-classes 4 --max-per-class 4 --permutations 20)
  OUTPUT_DIR="${SPLIT_DIAG_OUTPUT_DIR:-artifacts/split_diagnostics/aistats27_q3/imagenet_smoke}"
fi
"$PYTHON" experiments/vit_provenance/runners/diagnose_imagenet_splits.py \
  --device cuda --output-dir "$OUTPUT_DIR" --resume "${EXTRA[@]}"

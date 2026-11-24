#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-rtx6000
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH --job-name=public-neg-controls
#SBATCH --output=experiments/public_pythia_derivatives/logs/%x-%j.out
#SBATCH --error=experiments/public_pythia_derivatives/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi

module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv_pace}"
test -x "$ENV_DIR/bin/python"
conda activate "$ENV_DIR"

export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
export TOKENIZERS_PARALLELISM=false

ARTIFACT_ROOT="${PUBLIC_CONTROL_ARTIFACT_ROOT:-artifacts/public_pythia_derivatives/public_negative_controls/full}"
EXTRA_ARGS=()
if [ -n "${PUBLIC_CONTROL_MAX_EXAMPLES:-}" ]; then
  EXTRA_ARGS+=(--max-examples "$PUBLIC_CONTROL_MAX_EXAMPLES")
fi

python experiments/public_pythia_derivatives/runners/run_public_negative_controls.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --dtype float16 \
  --max-length 512 \
  --resume \
  "${EXTRA_ARGS[@]}"

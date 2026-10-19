#!/bin/bash
# Prepare the shared MIMIR split once before parallel model jobs start.
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH --job-name=pythia_audit_prepare
#SBATCH --output=experiments/public_pythia_derivatives/logs/%x_%j.out
#SBATCH --error=experiments/public_pythia_derivatives/logs/%x_%j.err

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

python experiments/public_pythia_derivatives/runners/run_audit.py \
  --seeds "${PYTHIA_SEEDS:-0}" \
  --prepare-only \
  --resume

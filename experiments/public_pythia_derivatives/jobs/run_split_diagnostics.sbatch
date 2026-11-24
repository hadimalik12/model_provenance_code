#!/bin/bash
# Frozen-encoder MIMIR diagnostics only. No Pythia model is loaded.
#SBATCH --job-name=mimir-split-diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --output=experiments/public_pythia_derivatives/logs/%x-%j.out
#SBATCH --error=experiments/public_pythia_derivatives/logs/%x-%j.err

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
if [ "${SPLIT_DIAG_SMOKE:-0}" = "1" ]; then
  EXTRA+=(--max-per-class 32 --permutations 20)
fi
"$PYTHON" experiments/public_pythia_derivatives/runners/diagnose_mimir_splits.py \
  --device cuda --resume "${EXTRA[@]}"

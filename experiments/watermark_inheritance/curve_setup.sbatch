#!/usr/bin/env bash
#SBATCH --job-name=wm_curve_data
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/watermark_curve/setup/logs/%x-%j.out
#SBATCH --error=artifacts/watermark_curve/setup/logs/%x-%j.err
set -euo pipefail
cd -P "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
REPO_ROOT=$(pwd -P)
module load pytorch/25 anaconda3 2>/dev/null || true
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/watermark/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
PYTHON_BIN=$(readlink -f "${WM_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
test -x "$PYTHON_BIN"
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
"$PYTHON_BIN" -m experiments.watermark_inheritance.curve_prepare \
  --workers "${SLURM_CPUS_PER_TASK:-4}" "$@"

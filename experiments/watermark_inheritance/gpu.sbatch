#!/usr/bin/env bash
#SBATCH --job-name=wm_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/watermark_inheritance/setup/logs/%x-%j.out
#SBATCH --error=artifacts/watermark_inheritance/setup/logs/%x-%j.err
set -euo pipefail
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
module load pytorch/25 anaconda3 2>/dev/null || true
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WM_ARTIFACT_ROOT="${WM_ARTIFACT_ROOT:-artifacts/watermark_inheritance}"
export WM_SHARED_DIR="${WM_SHARED_DIR:-.cache/watermark}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/$WM_SHARED_DIR/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHON_BIN=$(readlink -f "${WM_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
test -x "$PYTHON_BIN"
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv
"$PYTHON_BIN" -m experiments.watermark_inheritance.pipeline --phase "${1:?pilot or main}" --gpus "${2:?2 or 4}"

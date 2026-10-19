#!/usr/bin/env bash
#SBATCH --job-name=wm_curve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=160G
#SBATCH --time=26:00:00
#SBATCH --output=artifacts/watermark_curve/setup/logs/%x-%j.out
#SBATCH --error=artifacts/watermark_curve/setup/logs/%x-%j.err
set -euo pipefail
cd -P "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
REPO_ROOT=$(pwd -P)
module load pytorch/25 anaconda3 2>/dev/null || true
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
export PYTHONPATH="$REPO_ROOT"
curve_root="${1:-$PWD/artifacts/watermark_curve}"
python_bin=$(readlink -f "${WM_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
test -x "$python_bin"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
cd "$curve_root/setup/source"
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv
gpu_count=$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')
"$python_bin" -m experiments.watermark_inheritance.curve_pipeline \
  --root "$curve_root" --gpus "$gpu_count"

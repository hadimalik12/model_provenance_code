#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=01:00:00
set -euo pipefail
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
module load pytorch/25 anaconda3 2>/dev/null || true
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
"${1:?python}" -m experiments.watermark_inheritance.score --run-dir "${3:?run}" \
    --checkpoint-name "${4:?name}" --checkpoint-path "${5:?checkpoint}" --cpu-only
"$1" -m experiments.watermark_inheritance.curve_report --root "${2:?root}"

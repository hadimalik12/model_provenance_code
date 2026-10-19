#!/usr/bin/env bash
#SBATCH --job-name=wm_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --array=0-17%6
#SBATCH --output=artifacts/watermark_inheritance/setup/logs/%x-%A_%a.out
#SBATCH --error=artifacts/watermark_inheritance/setup/logs/%x-%A_%a.err
set -euo pipefail
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
module load pytorch/25 anaconda3 2>/dev/null || true
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WM_ARTIFACT_ROOT="${WM_ARTIFACT_ROOT:-artifacts/watermark_inheritance}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
PYTHON_BIN=$(readlink -f "${WM_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
test -x "$PYTHON_BIN"
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
checkpoints=(B P T1 T2 T T0)
task=${SLURM_ARRAY_TASK_ID:?}
run_index=$(( task / 6 ))
checkpoint_index=$(( task % 6 ))
run_name=$(printf 'run_%02d' "$run_index")
"$PYTHON_BIN" -m experiments.watermark_inheritance.score \
    --run-dir "$WM_ARTIFACT_ROOT/$run_name" \
    --checkpoint-name "${checkpoints[$checkpoint_index]}" --cpu-only

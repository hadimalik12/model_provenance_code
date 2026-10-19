#!/usr/bin/env bash
#SBATCH --job-name=wm_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/watermark_inheritance/setup/logs/%x-%j.out
#SBATCH --error=artifacts/watermark_inheritance/setup/logs/%x-%j.err
set -euo pipefail
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
module load pytorch/25 anaconda3 2>/dev/null || true
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
export PYTHONPATH="$REPO_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export WM_ARTIFACT_ROOT="${WM_ARTIFACT_ROOT:-artifacts/watermark_inheritance}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
PYTHON_BIN=$(readlink -f "${WM_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
test -x "$PYTHON_BIN"
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
for run in run_00 run_01 run_02; do
    "$PYTHON_BIN" -m experiments.watermark_inheritance.report \
        --run-dir "$WM_ARTIFACT_ROOT/$run"
done
"$PYTHON_BIN" -m experiments.watermark_inheritance.report \
    --run-dirs "$WM_ARTIFACT_ROOT/run_00" "$WM_ARTIFACT_ROOT/run_01" "$WM_ARTIFACT_ROOT/run_02" \
    --output-dir "$WM_ARTIFACT_ROOT/reports"

#!/usr/bin/env bash
#SBATCH --job-name=wm_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/watermark_inheritance/setup/logs/%x-%j.out
#SBATCH --error=artifacts/watermark_inheritance/setup/logs/%x-%j.err
set -euo pipefail
cd -P "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
REPO_ROOT=$(pwd -P)
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
PYTHON_BIN=$(readlink -f "${WM_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
test -x "$PYTHON_BIN"
mkdir -p "$WM_ARTIFACT_ROOT/setup/logs" "$WM_SHARED_DIR" \
  "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
"$PYTHON_BIN" -m experiments.watermark_inheritance.configure \
  --root "$WM_ARTIFACT_ROOT" --shared-dir "$WM_SHARED_DIR"
UPSTREAM="$WM_SHARED_DIR/radioactive-watermark"
UPSTREAM_COMMIT=19c6dc90e7a0dddb1a241b73e7fae708b68f024e
if [ ! -d "$UPSTREAM/.git" ]; then
  git clone https://github.com/facebookresearch/radioactive-watermark.git "$UPSTREAM"
fi
git -C "$UPSTREAM" fetch --tags origin
git -C "$UPSTREAM" checkout --detach "$UPSTREAM_COMMIT"
git -C "$UPSTREAM" diff --quiet
git -C "$UPSTREAM" diff --cached --quiet
"$PYTHON_BIN" -m unittest discover \
  -s experiments/watermark_inheritance/tests -v
"$PYTHON_BIN" -m experiments.watermark_inheritance.prepare \
  --run-dir "$WM_ARTIFACT_ROOT/run_00" --stage download --shared-dir "$WM_SHARED_DIR"
for run in run_00 run_01 run_02; do
    "$PYTHON_BIN" -m experiments.watermark_inheritance.prepare \
      --run-dir "$WM_ARTIFACT_ROOT/$run" --stage build --shared-dir "$WM_SHARED_DIR"
done
"$PYTHON_BIN" -m pip freeze > "$WM_ARTIFACT_ROOT/setup/packages.txt"

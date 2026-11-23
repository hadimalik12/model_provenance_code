#!/bin/bash
#SBATCH --job-name=wm-fp-uprep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}

"$PYTHON" -m experiments.watermark_inheritance.curve_prepare \
  --root "$ROOT/ultrachat_controls" \
  --original "$ROOT/ultrachat_shards" \
  --workers 16 --corpus full --count 400000 --epochs 1

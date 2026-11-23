#!/bin/bash
#SBATCH --job-name=wm-null-ultra
#SBATCH --array=0-2%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00

set -euo pipefail

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}
INDEX=${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}
RUN=$(printf '%s/ultrachat_controls/run_%02d' "$ROOT" "$INDEX")

"$PYTHON" -m experiments.watermark_inheritance.null_control generate \
  --run-dir "$RUN" --device cuda:0
"$PYTHON" -m experiments.watermark_inheritance.null_control score \
  --run-dir "$RUN" --checkpoint-name UC12500 \
  --checkpoint-path "$RUN/checkpoints/control_step12500" --device cuda:0

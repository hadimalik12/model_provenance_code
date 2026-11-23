#!/bin/bash
#SBATCH --job-name=wm-null-dolly
#SBATCH --array=0-11%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00

set -euo pipefail

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}
INDEX=${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}
RUN=$(printf '%s/dolly/unit_%02d' "$ROOT" "$INDEX")

"$PYTHON" -m experiments.watermark_inheritance.null_control generate \
  --run-dir "$RUN" --device cuda:0
"$PYTHON" -m experiments.watermark_inheritance.null_control score \
  --run-dir "$RUN" --checkpoint-name C3 \
  --checkpoint-path "$RUN/checkpoints/control_epoch3" --device cuda:0

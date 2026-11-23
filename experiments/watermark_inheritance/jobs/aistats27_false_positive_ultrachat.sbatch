#!/bin/bash
#SBATCH --job-name=wm-fp-ultra
#SBATCH --array=0-2%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=16:00:00

set -euo pipefail

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}
INDEX=${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}
RUN=$(printf '%s/ultrachat_controls/run_%02d' "$ROOT" "$INDEX")

"$PYTHON" -m experiments.watermark_inheritance.curve_train --run-dir "$RUN" --stage control --device cuda:0
for STEP in 512 1024 2048 3125 6250 12500; do
  "$PYTHON" -m experiments.watermark_inheritance.score --run-dir "$RUN" \
    --checkpoint-name "UC$STEP" --checkpoint-path "$RUN/checkpoints/control_step$STEP" --device cuda:0
done

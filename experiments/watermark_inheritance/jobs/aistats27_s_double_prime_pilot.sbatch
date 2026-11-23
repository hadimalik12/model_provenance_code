#!/bin/bash
#SBATCH --job-name=wm-null-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00

set -euo pipefail

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}
RUN="$ROOT/pilot"

"$PYTHON" -m experiments.watermark_inheritance.null_control generate \
  --run-dir "$RUN" --device cuda:0
"$PYTHON" -m experiments.watermark_inheritance.null_control score \
  --run-dir "$RUN" --checkpoint-name CP \
  --original-score-name pilot_CP_128 \
  --checkpoint-path "$RUN/checkpoints/pilot_control_step3" --device cuda:0

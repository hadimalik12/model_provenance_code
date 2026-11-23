#!/bin/bash
#SBATCH --job-name=wm-fp-ushard
#SBATCH --array=0-2%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00

set -euo pipefail

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}
INDEX=${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}
RUN=$(printf '%s/ultrachat_shards/run_%02d' "$ROOT" "$INDEX")

"$PYTHON" -m experiments.watermark_inheritance.prepare --run-dir "$RUN" --stage build
"$PYTHON" -m experiments.watermark_inheritance.generate --run-dir "$RUN" --phase full --device cuda:0
"$PYTHON" -m experiments.watermark_inheritance.generate --run-dir "$RUN" --phase full --world-size 1 --merge

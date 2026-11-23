#!/bin/bash
#SBATCH --job-name=wm-fp-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:00:00

set -euo pipefail

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}
RUN="$ROOT/pilot"

"$PYTHON" -m experiments.watermark_inheritance.reviewer_configure --root "$ROOT"
"$PYTHON" -m experiments.watermark_inheritance.prepare --run-dir "$RUN" --stage build
"$PYTHON" -m experiments.watermark_inheritance.generate --run-dir "$RUN" --phase full --device cuda:0
"$PYTHON" -m experiments.watermark_inheritance.generate --run-dir "$RUN" --phase full --world-size 1 --merge
"$PYTHON" -m experiments.watermark_inheritance.train --run-dir "$RUN" --stage control --device cuda:0 --max-steps 3
"$PYTHON" -m experiments.watermark_inheritance.score --run-dir "$RUN" --checkpoint-name CP \
  --checkpoint-path "$RUN/checkpoints/pilot_control_step3" --limit 128 --device cuda:0

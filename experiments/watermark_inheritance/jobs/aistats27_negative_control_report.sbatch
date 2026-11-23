#!/bin/bash
#SBATCH --job-name=wm-neg-report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}

"$PYTHON" -m experiments.watermark_inheritance.null_control report \
  --root "$ROOT" --output-dir "$ROOT/reports"

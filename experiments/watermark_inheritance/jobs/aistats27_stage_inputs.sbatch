#!/bin/bash
#SBATCH --job-name=wm-stage-inputs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}
RUN="$ROOT/pilot"

# Download and verify the pinned model and datasets once on CPU. All GPU jobs
# consume these immutable shared inputs through their generated configs.
"$PYTHON" -m experiments.watermark_inheritance.reviewer_configure --root "$ROOT"
"$PYTHON" -m experiments.watermark_inheritance.prepare --run-dir "$RUN" --stage download

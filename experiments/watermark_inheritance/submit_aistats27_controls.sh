#!/bin/bash
set -euo pipefail

PYTHON=${WM_PYTHON:-.venv_pace/bin/python}
ROOT=${WM_REVIEWER_ROOT:-artifacts/watermark_inheritance/aistats27_reviewer}
ACCOUNT=${WM_SLURM_ACCOUNT:-gts-vzikas3}
GPU_PARTITION=${WM_GPU_PARTITION:-gpu-h200}
CPU_PARTITION=${WM_CPU_PARTITION:-cpu-small}
"$PYTHON" -m experiments.watermark_inheritance.reviewer_configure --root "$ROOT"

STAGE=$(sbatch --parsable --account="$ACCOUNT" --partition="$CPU_PARTITION" \
  experiments/watermark_inheritance/jobs/aistats27_stage_inputs.sbatch)
PILOT=$(sbatch --parsable --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --dependency="afterok:$STAGE" \
  experiments/watermark_inheritance/jobs/aistats27_false_positive_pilot.sbatch)
DOLLY=$(sbatch --parsable --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --dependency="afterok:$PILOT" \
  experiments/watermark_inheritance/jobs/aistats27_false_positive_dolly.sbatch)
SHARDS=$(sbatch --parsable --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --dependency="afterok:$PILOT" \
  experiments/watermark_inheritance/jobs/aistats27_ultrachat_shards.sbatch)
PREP=$(sbatch --parsable --account="$ACCOUNT" --partition="$CPU_PARTITION" \
  --dependency="afterok:$SHARDS" \
  experiments/watermark_inheritance/jobs/aistats27_ultrachat_prepare.sbatch)
ULTRA=$(sbatch --parsable --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --dependency="afterok:$PREP" \
  experiments/watermark_inheritance/jobs/aistats27_false_positive_ultrachat.sbatch)
NULL_DOLLY=$(sbatch --parsable --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --dependency="afterok:$DOLLY" \
  experiments/watermark_inheritance/jobs/aistats27_s_double_prime_dolly.sbatch)
NULL_ULTRA=$(sbatch --parsable --account="$ACCOUNT" --partition="$GPU_PARTITION" \
  --dependency="afterok:$ULTRA" \
  experiments/watermark_inheritance/jobs/aistats27_s_double_prime_ultrachat.sbatch)
REPORT=$(sbatch --parsable --account="$ACCOUNT" --partition="$CPU_PARTITION" \
  --dependency="afterok:$NULL_DOLLY:$NULL_ULTRA" \
  experiments/watermark_inheritance/jobs/aistats27_negative_control_report.sbatch)
printf 'Shared inputs: %s\nPilot: %s\nDolly controls: %s\nUltraChat shards: %s\nUltraChat prepare: %s\nUltraChat controls: %s\nDolly S-double-prime: %s\nUltraChat S-double-prime: %s\nNegative-control report: %s\n' \
  "$STAGE" "$PILOT" "$DOLLY" "$SHARDS" "$PREP" "$ULTRA" "$NULL_DOLLY" "$NULL_ULTRA" "$REPORT"

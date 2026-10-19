#!/usr/bin/env bash
# Submit the Appendix D.1 UltraChat single-pass trajectory.
set -euo pipefail

SCRIPT_DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -P -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

WM_PYTHON=$(readlink -f "${WM_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
export WM_PYTHON
export WM_ARTIFACT_ROOT="${WM_ARTIFACT_ROOT:-artifacts/watermark_inheritance}"
export WM_CURVE_ROOT="${WM_CURVE_ROOT:-artifacts/watermark_curve_distinct}"
test -x "$WM_PYTHON"
test -f "$WM_ARTIFACT_ROOT/run_00/prepared/pairs.jsonl"
mkdir -p "$WM_CURVE_ROOT/setup/logs"

options=(--parsable --kill-on-invalid-dep=yes
  --output="$WM_CURVE_ROOT/setup/logs/%x-%j.out"
  --error="$WM_CURVE_ROOT/setup/logs/%x-%j.err"
  --export="ALL,WM_PYTHON=$WM_PYTHON,WM_ARTIFACT_ROOT=$WM_ARTIFACT_ROOT,WM_CURVE_ROOT=$WM_CURVE_ROOT")
if [ -n "${WM_ACCOUNT:-}" ]; then options+=(--account="$WM_ACCOUNT"); fi
if [ -n "${WM_CPU_PARTITION:-}" ]; then options+=(--partition="$WM_CPU_PARTITION"); fi
data_job=$(sbatch "${options[@]}" "$SCRIPT_DIR/curve_setup.sbatch" \
  --root "$WM_CURVE_ROOT" --original "$WM_ARTIFACT_ROOT" \
  --corpus full --count 600000 --epochs 1)
data_job=${data_job%%;*}
gpu_job=$("$WM_PYTHON" -m experiments.watermark_inheritance.curve_submit \
  --root "$WM_CURVE_ROOT" --data-job "$data_job" \
  --walltime "${WM_CURVE_TIME:-48:00:00}")
printf 'UltraChat trajectory submitted: data=%s gpu=%s\n' "$data_job" "$gpu_job"

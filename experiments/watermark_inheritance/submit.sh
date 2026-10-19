#!/usr/bin/env bash
# Submit the completed Dolly watermark experiment as one dependency chain.
set -euo pipefail

SCRIPT_DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -P -- "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

export WM_ARTIFACT_ROOT="${WM_ARTIFACT_ROOT:-artifacts/watermark_inheritance}"
export WM_SHARED_DIR="${WM_SHARED_DIR:-.cache/watermark}"
WM_PYTHON=$(readlink -f "${WM_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
export WM_PYTHON
test -x "$WM_PYTHON"
mkdir -p "$WM_ARTIFACT_ROOT/setup/logs"

common=(--parsable --kill-on-invalid-dep=yes
  --output="$WM_ARTIFACT_ROOT/setup/logs/%x-%j.out"
  --error="$WM_ARTIFACT_ROOT/setup/logs/%x-%j.err"
  --export="ALL,WM_ARTIFACT_ROOT=$WM_ARTIFACT_ROOT,WM_SHARED_DIR=$WM_SHARED_DIR,WM_PYTHON=$WM_PYTHON")
cpu=()
gpu=()
if [ -n "${WM_ACCOUNT:-}" ]; then cpu+=(--account="$WM_ACCOUNT"); fi
if [ -n "${WM_GPU_ACCOUNT:-${WM_ACCOUNT:-}}" ]; then gpu+=(--account="${WM_GPU_ACCOUNT:-$WM_ACCOUNT}"); fi
if [ -n "${WM_CPU_PARTITION:-}" ]; then cpu+=(--partition="$WM_CPU_PARTITION"); fi
if [ -n "${WM_GPU_PARTITION:-}" ]; then gpu+=(--partition="$WM_GPU_PARTITION"); fi
gpu_kind="${WM_GPU_TYPE-h200}"
if [ -n "$gpu_kind" ]; then
  pilot_gres="gpu:$gpu_kind:2"
  main_gres="gpu:$gpu_kind:4"
else
  pilot_gres="gpu:2"
  main_gres="gpu:4"
fi

setup=$(sbatch "${common[@]}" "${cpu[@]}" "$SCRIPT_DIR/setup.sbatch")
setup=${setup%%;*}
pilot=$(sbatch "${common[@]}" "${gpu[@]}" --dependency="afterok:$setup" \
  --gres="$pilot_gres" "$SCRIPT_DIR/gpu.sbatch" pilot 2)
pilot=${pilot%%;*}
main=$(sbatch "${common[@]}" "${gpu[@]}" --dependency="afterok:$pilot" \
  --job-name=wm_main --gres="$main_gres" --time="${WM_MAIN_TIME:-08:00:00}" \
  "$SCRIPT_DIR/gpu.sbatch" main 4)
main=${main%%;*}
score=$(sbatch "${common[@]}" "${cpu[@]}" --dependency="afterok:$main" \
  "$SCRIPT_DIR/cpu_score.sbatch")
score=${score%%;*}
report=$(sbatch "${common[@]}" "${cpu[@]}" --dependency="afterok:$score" \
  "$SCRIPT_DIR/report.sbatch")
report=${report%%;*}

printf 'Watermark experiment submitted: setup=%s pilot=%s main=%s score=%s report=%s\n' \
  "$setup" "$pilot" "$main" "$score" "$report"

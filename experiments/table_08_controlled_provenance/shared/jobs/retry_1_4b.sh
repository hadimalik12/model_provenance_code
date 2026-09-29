#!/bin/bash
# Retry only the failed 1.4B branch, reusing completed preparation and 1B work.
set -euo pipefail
SCRIPT_DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -P -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
CONFIG="${TABLE8_CONFIG:-experiments/table_08_controlled_provenance/keyed_rank/configs/config.yaml}"
GPU=experiments/table_08_controlled_provenance/shared/jobs/gpu.sbatch
CPU=experiments/table_08_controlled_provenance/shared/jobs/prepare.sbatch
PYTHON_BIN=$(readlink -f "${TABLE8_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
ROOT=artifacts/table_08_controlled_provenance/keyed_rank/csn_filtered_seed0

# Preserve failed partial outputs for inspection instead of deleting them.
STAMP=$(date +%Y%m%d_%H%M%S)
for path in "$ROOT/models/1_4b/parent" "$ROOT/models/1_4b/helpful_sft/control" "$ROOT/models/1_4b/tulu_sft/control"; do
  if [ -d "$path" ] && [ ! -f "$path/complete.json" ]; then mv "$path" "${path}.failed_${STAMP}"; fi
done

submit() {
  local dep="$1" task="$2" stage="$3" profile="$4"
  local args=(--parsable --kill-on-invalid-dep=yes --job-name="t8_${task}_1_4b_${profile}_${stage}" --export="ALL,TABLE8_CONFIG=$CONFIG,TABLE8_PYTHON=$PYTHON_BIN,TABLE8_TASK=$task,TABLE8_FAMILY=1_4b,TABLE8_STAGE=$stage,TABLE8_PROFILE=$profile")
  [ -z "$dep" ] || args+=(--dependency="afterok:$dep")
  local id; id=$(sbatch "${args[@]}" "$GPU"); echo "${id%%;*}"
}

PARENT=$(submit "" train parent "")
LAST=$(submit "$PARENT" audit parent "")
for PROFILE in helpful_sft tulu_sft; do
  TARGET=$(submit "$LAST" train target "$PROFILE")
  TARGET_AUDIT=$(submit "$TARGET" audit target "$PROFILE")
  CONTROL=$(submit "$TARGET_AUDIT" train control "$PROFILE")
  CONTROL_AUDIT=$(submit "$CONTROL" audit control "$PROFILE")
  LAST="$CONTROL_AUDIT"
  echo "$PROFILE target=$TARGET control=$CONTROL audits=$TARGET_AUDIT,$CONTROL_AUDIT"
done
REPORT=$(sbatch --parsable --kill-on-invalid-dep=yes --job-name=t8_final_report_retry --dependency="afterok:$LAST" \
  --export="ALL,TABLE8_CONFIG=$CONFIG,TABLE8_PYTHON=$PYTHON_BIN,TABLE8_TASK=compile" "$CPU")
echo "parent=$PARENT report=${REPORT%%;*}"

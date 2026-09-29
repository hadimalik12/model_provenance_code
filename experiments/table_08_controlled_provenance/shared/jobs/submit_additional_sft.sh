#!/bin/bash
# Submit the additional 1.4B LaMini and finance SFT interventions.
set -euo pipefail
SCRIPT_DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -P -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
CONFIG=experiments/table_08_controlled_provenance/keyed_rank/configs/additional_sft.yaml
CPU=experiments/table_08_controlled_provenance/shared/jobs/prepare.sbatch
GPU=experiments/table_08_controlled_provenance/shared/jobs/gpu.sbatch
PYTHON_BIN=$(readlink -f "${TABLE8_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
test -x "$PYTHON_BIN"
test -f "$CONFIG"
ARTIFACT_ROOT=artifacts/table_08_controlled_provenance/keyed_rank/additional_sft_seed0
if [ -e "$ARTIFACT_ROOT" ]; then
  echo "Refusing to reuse existing artifact root: $ARTIFACT_ROOT" >&2
  exit 1
fi
env -u PYTHONHOME -u PYTHONPATH -u LD_PRELOAD -u LD_LIBRARY_PATH \
  /usr/bin/python3 -m experiments.table_08_controlled_provenance.shared.local_source --config "$CONFIG"
mkdir -p experiments/table_08_controlled_provenance/logs

submit() {
  local dependency="$1" script="$2" task="$3" stage="$4" profile="$5"
  local args=(--parsable --kill-on-invalid-dep=yes --job-name="t8_${task}_1_4b_${profile}_${stage}" --export="ALL,TABLE8_CONFIG=$CONFIG,TABLE8_PYTHON=$PYTHON_BIN,TABLE8_TASK=$task,TABLE8_FAMILY=1_4b,TABLE8_STAGE=$stage,TABLE8_PROFILE=$profile")
  if [ -n "$dependency" ]; then args+=(--dependency="afterok:$dependency"); fi
  local job_id
  job_id=$(sbatch "${args[@]}" "$script")
  echo "${job_id%%;*}"
}

SOURCE_ID=$(submit "" "$CPU" source "" "")
PARENT_ID=$(submit "$SOURCE_ID" "$GPU" train parent "")
BASE_AUDIT_ID=$(submit "$SOURCE_ID" "$GPU" audit base "")
PARENT_AUDIT_ID=$(submit "$PARENT_ID" "$GPU" audit parent "")
REPORT_DEP=("$BASE_AUDIT_ID" "$PARENT_AUDIT_ID")
echo "source=$SOURCE_ID parent=$PARENT_ID base_audit=$BASE_AUDIT_ID parent_audit=$PARENT_AUDIT_ID"

for PROFILE in lamini_sft finance_sft; do
  DATA_ID=$(submit "$SOURCE_ID" "$CPU" downstream "" "$PROFILE")
  TARGET_ID=$(submit "$PARENT_ID:$DATA_ID" "$GPU" train target "$PROFILE")
  CONTROL_ID=$(submit "$DATA_ID" "$GPU" train control "$PROFILE")
  TARGET_AUDIT_ID=$(submit "$TARGET_ID" "$GPU" audit target "$PROFILE")
  CONTROL_AUDIT_ID=$(submit "$CONTROL_ID" "$GPU" audit control "$PROFILE")
  REPORT_DEP+=("$TARGET_AUDIT_ID" "$CONTROL_AUDIT_ID")
  echo "$PROFILE data=$DATA_ID target=$TARGET_ID control=$CONTROL_ID audits=$TARGET_AUDIT_ID,$CONTROL_AUDIT_ID"
done

IFS=:
REPORT_IDS="${REPORT_DEP[*]}"
REPORT_ID=$(sbatch --parsable --kill-on-invalid-dep=yes --job-name=t8_additional_sft_report --dependency="afterok:$REPORT_IDS" \
  --export="ALL,TABLE8_CONFIG=$CONFIG,TABLE8_PYTHON=$PYTHON_BIN,TABLE8_TASK=compile" "$CPU")
echo "report=${REPORT_ID%%;*}"

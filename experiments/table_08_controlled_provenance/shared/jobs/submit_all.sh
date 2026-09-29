#!/bin/bash
# Submit all configured Table 8 jobs with Slurm dependencies.
set -euo pipefail
SCRIPT_DIR=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -P -- "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"
CONFIG="${TABLE8_CONFIG:-experiments/table_08_controlled_provenance/global_shuffle/configs/config.yaml}"
CPU=experiments/table_08_controlled_provenance/shared/jobs/prepare.sbatch
GPU=experiments/table_08_controlled_provenance/shared/jobs/gpu.sbatch
PYTHON_BIN=$(readlink -f "${TABLE8_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
test -x "$PYTHON_BIN"
test -f "$CONFIG"
# The login-node XALT preload can prevent relocated Conda Python from even
# importing encodings. Use the system Python for this stdlib/PyYAML preflight;
# compute jobs continue to use the full experiment environment below.
env -u PYTHONHOME -u PYTHONPATH -u LD_PRELOAD -u LD_LIBRARY_PATH \
  /usr/bin/python3 -m experiments.table_08_controlled_provenance.shared.local_source --config "$CONFIG" --submission
mkdir -p experiments/table_08_controlled_provenance/logs

submit() {
  local dependency="$1" script="$2" task="$3" family="$4" stage="$5" profile="$6"
  local args=(--parsable --kill-on-invalid-dep=yes --job-name="t8_${task}_${family}_${profile}_${stage}" --export="ALL,TABLE8_CONFIG=$CONFIG,TABLE8_PYTHON=$PYTHON_BIN,TABLE8_TASK=$task,TABLE8_FAMILY=$family,TABLE8_STAGE=$stage,TABLE8_PROFILE=$profile")
  if [ -n "$dependency" ]; then args+=(--dependency="afterok:$dependency"); fi
  local job_id
  job_id=$(sbatch "${args[@]}" "$script")
  echo "${job_id%%;*}"
}

SOURCE_ID=$(submit "" "$CPU" source "" "" "")
echo "source $SOURCE_ID"
REPORT_DEP=()
for FAMILY in 1b 1_4b; do
  PARENT_ID=$(submit "$SOURCE_ID" "$GPU" train "$FAMILY" parent "")
  echo "$FAMILY parent $PARENT_ID"
  BASE_AUDIT_ID=$(submit "$SOURCE_ID" "$GPU" audit "$FAMILY" base "")
  PARENT_AUDIT_ID=$(submit "$PARENT_ID" "$GPU" audit "$FAMILY" parent "")
  REPORT_DEP+=("$BASE_AUDIT_ID" "$PARENT_AUDIT_ID")
  if [ "$FAMILY" = 1b ]; then PROFILES=(hh_sft); else PROFILES=(helpful_sft tulu_sft); fi
  for PROFILE in "${PROFILES[@]}"; do
    DATA_ID=$(submit "$SOURCE_ID" "$CPU" downstream "" "" "$PROFILE")
    TARGET_ID=$(submit "$PARENT_ID:$DATA_ID" "$GPU" train "$FAMILY" target "$PROFILE")
    CONTROL_ID=$(submit "$DATA_ID" "$GPU" train "$FAMILY" control "$PROFILE")
    TARGET_AUDIT_ID=$(submit "$TARGET_ID" "$GPU" audit "$FAMILY" target "$PROFILE")
    CONTROL_AUDIT_ID=$(submit "$CONTROL_ID" "$GPU" audit "$FAMILY" control "$PROFILE")
    REPORT_DEP+=("$TARGET_AUDIT_ID" "$CONTROL_AUDIT_ID")
    echo "$FAMILY $PROFILE data=$DATA_ID target=$TARGET_ID control=$CONTROL_ID audits=$TARGET_AUDIT_ID,$CONTROL_AUDIT_ID"
  done
done
IFS=:
REPORT_IDS="${REPORT_DEP[*]}"
REPORT_ID=$(sbatch --parsable --kill-on-invalid-dep=yes --job-name=t8_final_report --dependency="afterok:$REPORT_IDS" \
  --export="ALL,TABLE8_CONFIG=$CONFIG,TABLE8_PYTHON=$PYTHON_BIN,TABLE8_TASK=compile" "$CPU")
echo "report $REPORT_ID"

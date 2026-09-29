#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100,gpu-a100,gpu-l40s,gpu-h100,gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --constraint=V100-32GB|A100|L40S|H100|H200
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH --account=gts-vzikas3
#SBATCH --job-name=table08_gpu
#SBATCH --output=experiments/table_08_controlled_provenance/shared/logs/%x_%j.out
#SBATCH --error=experiments/table_08_controlled_provenance/shared/logs/%x_%j.err
set -euo pipefail
cd -P "${SLURM_SUBMIT_DIR:?Submit from repository root}"
REPO_ROOT=$(pwd -P)
module load pytorch/25 anaconda3 || true
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PYTHON_BIN=$(readlink -f "${TABLE8_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
test -x "$PYTHON_BIN"
CONFIG="${TABLE8_CONFIG:-experiments/table_08_controlled_provenance/global_shuffle/configs/config.yaml}"
case "${TABLE8_TASK:?}" in
  train)
    ARGS=(--config "$CONFIG" --family "${TABLE8_FAMILY:?}" --stage "${TABLE8_STAGE:?}")
    if [ -n "${TABLE8_PROFILE:-}" ]; then ARGS+=(--profile "$TABLE8_PROFILE"); fi
    "$PYTHON_BIN" -m experiments.table_08_controlled_provenance.shared.train "${ARGS[@]}"
    ;;
  audit)
    ARGS=(--config "$CONFIG" --family "${TABLE8_FAMILY:?}" --stage "${TABLE8_STAGE:?}")
    if [ -n "${TABLE8_PROFILE:-}" ]; then ARGS+=(--profile "$TABLE8_PROFILE"); fi
    "$PYTHON_BIN" -m experiments.table_08_controlled_provenance.shared.audit "${ARGS[@]}"
    ;;
  *) echo "Unknown GPU task: $TABLE8_TASK" >&2; exit 2 ;;
esac

#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH --account=gts-vzikas3
#SBATCH --job-name=table08_prepare
#SBATCH --output=experiments/table_08_controlled_provenance/shared/logs/%x_%j.out
#SBATCH --error=experiments/table_08_controlled_provenance/shared/logs/%x_%j.err
set -euo pipefail
cd -P "${SLURM_SUBMIT_DIR:?Submit from repository root}"
REPO_ROOT=$(pwd -P)
module load pytorch/25 anaconda3 || true
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
export PYTHONPATH="$REPO_ROOT"
PYTHON_BIN=$(readlink -f "${TABLE8_PYTHON:-$REPO_ROOT/.venv_pace/bin/python}")
test -x "$PYTHON_BIN"
CONFIG="${TABLE8_CONFIG:-experiments/table_08_controlled_provenance/global_shuffle/configs/config.yaml}"
case "${TABLE8_TASK:?}" in
  source) "$PYTHON_BIN" -m experiments.table_08_controlled_provenance.shared.prepare --config "$CONFIG" ;;
  downstream) "$PYTHON_BIN" -m experiments.table_08_controlled_provenance.shared.downstream --config "$CONFIG" --profile "${TABLE8_PROFILE:?}" ;;
  compile) "$PYTHON_BIN" -m experiments.table_08_controlled_provenance.shared.compile --config "$CONFIG" ;;
  *) echo "Unknown CPU task: $TABLE8_TASK" >&2; exit 2 ;;
esac

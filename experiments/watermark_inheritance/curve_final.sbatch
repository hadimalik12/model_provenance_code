#!/usr/bin/env bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
module load pytorch/25 anaconda3 2>/dev/null || true
unset PYTHONHOME PYTHONPATH LD_PRELOAD LD_LIBRARY_PATH
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/.cache/triton}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
"${1:?python}" -m experiments.watermark_inheritance.curve_report --root "${2:?root}"
"$1" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); r=json.loads((p/"reports/curve.json").read_text()); assert r["completed_nonbaseline_points"] == r["planned_nonbaseline_points"]; print("All",r["planned_nonbaseline_points"],"interval audits complete")' "$2"

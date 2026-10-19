#!/bin/bash
#SBATCH -N 1
#SBATCH --mem=8G
#SBATCH -t 0:30:00
#SBATCH --job-name=pythia_audit_report
#SBATCH --output=experiments/public_pythia_derivatives/logs/%x_%j.out
#SBATCH --error=experiments/public_pythia_derivatives/logs/%x_%j.err

set -euo pipefail
REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"
PYTHON="${PYTHIA_PYTHON:-${ENV_DIR:-$REPO_ROOT/.venv_pace}/bin/python}"
test -x "$PYTHON"
"$PYTHON" experiments/public_pythia_derivatives/reports/compile_results.py

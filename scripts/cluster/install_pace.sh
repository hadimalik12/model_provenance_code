#!/bin/bash
set -euo pipefail

module purge
module load pytorch/25
module load anaconda3

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found after loading anaconda3 module" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv_pace}"

if [ -z "${HF_HOME:-}" ]; then
    export HF_HOME="$REPO_ROOT/.cache/huggingface"
fi
mkdir -p "$HF_HOME"

if [ -d "$ENV_DIR" ]; then
    echo "Conda env already exists at $ENV_DIR"
else
    echo "Creating conda env at $ENV_DIR"
    conda create --prefix "$ENV_DIR" python=3.11 -y
fi

conda activate "$ENV_DIR"

PIP="$ENV_DIR/bin/pip"
hash -r

"$PIP" install pip==26.1.2 setuptools==83.0.0 wheel==0.47.0
PYTHONNOUSERSITE=1 "$PIP" install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
PYTHONNOUSERSITE=1 "$PIP" install -r "$REPO_ROOT/reproducibility/requirements-lock.txt"
if [ -f "$REPO_ROOT/pyproject.toml" ] || [ -f "$REPO_ROOT/setup.py" ]; then
    PYTHONNOUSERSITE=1 "$PIP" install -e "$REPO_ROOT"
fi

mkdir -p "$REPO_ROOT/artifacts"
mkdir -p "$REPO_ROOT/.cache/huggingface"

cat <<EOF
Installation complete.
Activate with:
  conda activate $ENV_DIR

Experiment entrypoints are under:
  $REPO_ROOT/experiments/

For example:
  bash $REPO_ROOT/experiments/public_pythia_derivatives/submit.sh
EOF

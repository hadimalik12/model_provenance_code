#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --account=gts-vzikas3
#SBATCH --job-name=quantization_int8
#SBATCH --output=experiments/robustness_quantization/logs/%x_%j.out
#SBATCH --error=experiments/robustness_quantization/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
  source "$REPO_ROOT/.env"
fi

bash scripts/cluster/install_pace.sh
module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/model_provenance_venv
unset -f python || true

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$REPO_ROOT/.cache/huggingface"

mkdir -p experiments/robustness_quantization/logs

python scripts/setup/verify_hf_model_access.py \
  --model EleutherAI/pythia-1.4b \
  --model herMaster/pythia1.4B-finetuned-on-lamini-docs

python experiments/robustness_quantization/runners/run_int8_robustness.py --resume

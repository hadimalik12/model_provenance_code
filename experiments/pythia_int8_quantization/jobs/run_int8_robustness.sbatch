#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --job-name=quantization_int8
#SBATCH --output=experiments/pythia_int8_quantization/logs/%x_%j.out
#SBATCH --error=experiments/pythia_int8_quantization/logs/%x_%j.err

set -euo pipefail

REPO_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.env" ]; then
  source "$REPO_ROOT/.env"
fi

bash scripts/cluster/install_pace.sh
module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv_pace}"
conda activate "$ENV_DIR"
unset -f python || true

export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$REPO_ROOT/.cache/huggingface"

mkdir -p experiments/pythia_int8_quantization/logs

python scripts/setup/verify_hf_model_access.py \
  --model EleutherAI/pythia-1.4b@fedc38a16eea3bd36a96b906d78d11d2ce18ed79 \
  --model herMaster/pythia1.4B-finetuned-on-lamini-docs@a6c699d6832bd3852c13d9c728c8599666b04c22

python experiments/pythia_int8_quantization/runners/run_int8_robustness.py --resume

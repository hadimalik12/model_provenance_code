#!/bin/bash
# One-GPU preflight for the full INT8 experiment. This runs the same setup and
# scoring entrypoint on two records per split, without touching resumable output.
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH --job-name=quant_int8_smoke
#SBATCH --output=experiments/pythia_int8_quantization/logs/%x_%j.out
#SBATCH --error=experiments/pythia_int8_quantization/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
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

DATA_ROOT="$REPO_ROOT/artifacts/pythia_int8_quantization/int8_pythia1_4b_lamini_seed0/prepared/mimir_github"
test -f "$DATA_ROOT/train.jsonl"
test -f "$DATA_ROOT/test.jsonl"

python scripts/scoring/score_causal_lm_logprobs.py \
  --model herMaster/pythia1.4B-finetuned-on-lamini-docs \
  --train-file "$DATA_ROOT/train.jsonl" \
  --test-file "$DATA_ROOT/test.jsonl" \
  --output-dir "$REPO_ROOT/artifacts/pythia_int8_quantization/int8_pythia1_4b_lamini_seed0/smoke_int8_fp32_residual" \
  --min-k-pcts 20 \
  --batch-size 1 \
  --dtype float32 \
  --quantization int8 \
  --max-length 512 \
  --max-examples 2

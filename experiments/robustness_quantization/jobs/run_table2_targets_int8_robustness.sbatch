#!/bin/bash
# Quantization robustness across the exact target roster in Table 2's
# run_local_vanilla_audit.py (one 1B target and five 1.4B targets).
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --account=gts-vzikas3
#SBATCH --job-name=quantization_table2
#SBATCH --output=experiments/robustness_quantization/logs/%x_%j.out
#SBATCH --error=experiments/robustness_quantization/logs/%x_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"
[ -f .env ] && source .env
bash scripts/cluster/install_pace.sh
module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/model_provenance_venv
unset -f python || true
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$REPO_ROOT/.cache/huggingface"

# parent|target pairs copied from Table 2's local vanilla audit.
PAIRS=(
  "EleutherAI/pythia-1b|Leogrin/eleuther-pythia1b-hh-sft"
  "EleutherAI/pythia-1.4b|herMaster/pythia1.4B-finetuned-on-lamini-docs"
  "EleutherAI/pythia-1.4b|kykim0/pythia-1.4b-tulu-v2-mix"
  "EleutherAI/pythia-1.4b|LinguaCustodia/fin-pythia-1.4b"
  "EleutherAI/pythia-1.4b|lomahony/pythia-1.4b-helpful-dpo"
  "EleutherAI/pythia-1.4b|lomahony/pythia-1.4b-helpful-sft"
)

for PAIR in "${PAIRS[@]}"; do
  PARENT="${PAIR%%|*}"
  TARGET="${PAIR#*|}"
  python experiments/robustness_quantization/runners/run_int8_robustness.py \
    --parent-model "$PARENT" \
    --target-model "$TARGET" \
    --artifact-root "artifacts/robustness_quantization/table2_targets_seed0" \
    --resume \
    --skip-individual-report
done

python experiments/robustness_quantization/reports/compile_table2_targets.py

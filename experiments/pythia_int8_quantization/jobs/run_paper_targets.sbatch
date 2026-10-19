#!/bin/bash
# Appendix Table 7: INT8 robustness for one 1B and five 1.4B targets.
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --job-name=pythia_int8_paper
#SBATCH --output=experiments/pythia_int8_quantization/logs/%x_%j.out
#SBATCH --error=experiments/pythia_int8_quantization/logs/%x_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
cd "$REPO_ROOT"
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi
bash scripts/cluster/install_pace.sh
module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
ENV_DIR="${ENV_DIR:-$REPO_ROOT/.venv_pace}"
conda activate "$ENV_DIR"
unset -f python || true
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$REPO_ROOT/.cache/huggingface"

# Parent/target pairs reported in submitted Appendix Table 7.
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
  python experiments/pythia_int8_quantization/runners/run_int8_robustness.py \
    --parent-model "$PARENT" \
    --target-model "$TARGET" \
    --artifact-root "artifacts/pythia_int8_quantization/paper_targets_seed0" \
    --resume \
    --skip-individual-report
done

python experiments/pythia_int8_quantization/reports/compile_results.py

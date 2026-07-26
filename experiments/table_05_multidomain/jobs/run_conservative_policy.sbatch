#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 8:00:00
#SBATCH --account=gts-vzikas3
#SBATCH --job-name=table05_conservative_policy
#SBATCH --output=experiments/table_05_multidomain/logs/%x_%j.out
#SBATCH --error=experiments/table_05_multidomain/logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
fi
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/scripts/cluster/install_pace.sh" ]; then
  echo "ERROR: missing $REPO_ROOT/scripts/cluster/install_pace.sh" >&2
  ls -ld "$REPO_ROOT" "$REPO_ROOT/scripts" "$REPO_ROOT/scripts/cluster" || true
  exit 1
fi

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

# Paper Table 5 uses Pythia-1.4B and its SFT-derived target. For every target
# forward pass, logits are produced by the parent Pythia output head.
PARENT_MODEL="EleutherAI/pythia-1.4b"
TARGET_MODEL="herMaster/pythia1.4B-finetuned-on-lamini-docs"
PARENT_SLUG="parent_pythia_1_4b"
TARGET_SLUG="target_hermaster_pythia1_4b_lamini_docs"

python scripts/setup/verify_hf_model_access.py --model "$PARENT_MODEL" --model "$TARGET_MODEL"

DOMAINS_HF=("arxiv" "dm_mathematics" "github" "pile_cc" "wikipedia_(en)")
DOMAINS_CLEAN=("arxiv" "dm_mathematics" "github" "pile_cc" "wikipedia_en")
ARTIFACT_ROOT="artifacts/table_05_multidomain/conservative_policy_parent_head_seed0"
N_TRAIN=300
N_TEST=400
MIMIR_SPLIT="ngram_13_0.2"

mkdir -p "$ARTIFACT_ROOT"
cp experiments/table_05_multidomain/configs/augmented.yaml "$ARTIFACT_ROOT/config.lock.yaml"

for i in "${!DOMAINS_CLEAN[@]}"; do
  HF_DOMAIN="${DOMAINS_HF[$i]}"
  DOMAIN="${DOMAINS_CLEAN[$i]}"
  DATA_DIR="$ARTIFACT_ROOT/prepared/$DOMAIN"

  python scripts/data/prepare_mimir_domain.py \
    --config "$HF_DOMAIN" \
    --split "$MIMIR_SPLIT" \
    --num-train-per-class "$N_TRAIN" \
    --num-test-per-class "$N_TEST" \
    --seed 0 \
    --allow-overlap \
    --token "${HF_TOKEN:-}" \
    --output-dir "$DATA_DIR"

  for MODEL_SLUG in "$PARENT_SLUG" "$TARGET_SLUG"; do
    if [ "$MODEL_SLUG" = "$PARENT_SLUG" ]; then
      TARGET_BODY="$PARENT_MODEL"
    else
      TARGET_BODY="$TARGET_MODEL"
    fi

    SCORE_DIR="$ARTIFACT_ROOT/scores/$DOMAIN/$MODEL_SLUG"
    python scripts/scoring/score_table5_policy_variants.py \
      --parent-model "$PARENT_MODEL" \
      --target-model "$TARGET_BODY" \
      --train-file "$DATA_DIR/train.jsonl" \
      --test-file "$DATA_DIR/test.jsonl" \
      --output-dir "$SCORE_DIR" \
      --domain "$DOMAIN" \
      --min-k-pcts 20,40 \
      --dtype float16

    case "$DOMAIN" in
      github)
        ABLATIONS="ablation_code_line_endings,ablation_python_ast_normalize,ablation_code_typography" ;;
      arxiv)
        ABLATIONS="ablation_latex_typography,ablation_latex_control_spacing,ablation_latex_comments" ;;
      dm_mathematics)
        ABLATIONS="ablation_math_operator_spacing,ablation_math_alpha_rename,ablation_math_parenthesis_spacing" ;;
      wikipedia_en)
        ABLATIONS="ablation_prose_typography,ablation_prose_whitespace,ablation_wiki_bracket_spacing" ;;
      pile_cc)
        ABLATIONS="ablation_prose_typography,ablation_prose_whitespace,ablation_pile_html_entities" ;;
    esac
    SCORE_KEYS="min_k_20_logprob,min_k_40_logprob,safe_full_min_k_20_logprob_mean,safe_full_min_k_40_logprob_mean"
    for ABLATION in ${ABLATIONS//,/ }; do
      SCORE_KEYS="$SCORE_KEYS,${ABLATION}_min_k_20_logprob,${ABLATION}_min_k_40_logprob"
    done
    RUN_DIR="$ARTIFACT_ROOT/results/$DOMAIN/$MODEL_SLUG"
    python scripts/audit/run_threshold_audit.py \
      --train-scores "$SCORE_DIR/train_scores.jsonl" \
      --test-scores "$SCORE_DIR/test_scores.jsonl" \
      --output-dir "$RUN_DIR" \
      --score-keys "$SCORE_KEYS" \
      --primary-score min_k_20_logprob
  done
done

python experiments/table_05_multidomain/reports/compile_safe_policy.py

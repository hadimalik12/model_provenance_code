#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --account=gts-vzikas3
#SBATCH --job-name=table01_augmentation
#SBATCH --output=experiments/table_01_imagenet/logs/%x_%j.out
#SBATCH --error=experiments/table_01_imagenet/logs/%x_%j.err

echo "=========================================================="
echo " Starting Slurm Batch Job for Table 1 Reproduction"
echo " Date: $(date)"
echo " Hostname: $(hostname)"
echo "=========================================================="

# Resolve the repository from this file, independent of submit directory.
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

if [ ! -f "$REPO_ROOT/scripts/cluster/install_pace.sh" ]; then
  echo "ERROR: missing $REPO_ROOT/scripts/cluster/install_pace.sh" >&2
  exit 1
fi
if [ -f "$REPO_ROOT/.env" ]; then
  # Optional credentials for gated/private Hub models and datasets.
  source "$REPO_ROOT/.env"
fi

# Run install_pace.sh to install/reverify the environment on the node-local /tmp
echo "=== Installing environment on compute node local /tmp ==="
for ATTEMPT in 1 2 3; do
  bash scripts/cluster/install_pace.sh && break
  if [ "$ATTEMPT" -eq 3 ]; then
    echo "ERROR: environment setup failed after $ATTEMPT attempts" >&2
    exit 1
  fi
  echo "Environment setup failed; retrying ($((ATTEMPT + 1))/3)..." >&2
  sleep 10
done

# Auto-activate cluster environment modules and python environment
echo "=== Loading modules & activating environment ==="
module load pytorch/25 anaconda3 || true
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/model_provenance_venv
unset -f python || true
export PATH="/tmp/python-venv/model_provenance_venv/bin:$PATH"

PARENT="google/vit-base-patch16-224"
ARTIFACT_ROOT="artifacts/table_01_imagenet/augmented_logprob_seed0"
DATA="$ARTIFACT_ROOT/prepared"
SCORES="$ARTIFACT_ROOT/scores"
PREFIX="imagenet_shard_paper__"

# Step 0: Verification
SCORER="experiments/table_01_imagenet/scoring/score_augmented_parent_logprob.py"
if [ ! -f "$SCORER" ]; then
  echo "Error: $SCORER does not exist."
  echo "Please write the logprob scoring script first before running."
  exit 1
fi

# Step 1: Prepare the class-balanced dataset
echo ""
echo "=== Step 1: Preparing class-balanced dataset (77 classes, 7700 images) ==="
python experiments/table_01_imagenet/runners/prepare_shards.py \
    --train-shard-idxs 0,1,2,3,4 \
    --val-shard-idxs 0,1 \
    --members-per-class 50 \
    --nonmembers-per-class 50 \
    --max-classes 77 \
    --train-frac 0.5 \
    --seed 0 \
    --output-dir "$DATA"

# Step 2: Score target models and controls
echo ""
echo "=== Step 2: Scoring Target and Control Models ==="

declare -A TARGETS=(
  ["self"]="google/vit-base-patch16-224"
  ["pets"]="ISxOdin/vit-base-oxford-iiit-pets"
  ["cifar10"]="MatanBT/vit-base-patch16-224-cifar10"
  ["cifar100"]="MatanBT/vit-base-patch16-224-cifar100"
  ["eurosat"]="Abhiram4/vit-base-patch16-224-finetuned-eurosat"
  ["celeba"]="MatanBT/vit-base-patch16-224-celeba-smiling"
  ["har"]="rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224"
)

python scripts/setup/verify_hf_model_access.py --model "$PARENT" --model "facebook/deit-base-patch16-224"
for MODEL in "${TARGETS[@]}"; do
  python scripts/setup/verify_hf_model_access.py --model "$MODEL"
done

for NAME in "${!TARGETS[@]}"; do
  echo "Scoring target model: $NAME (${TARGETS[$NAME]})"
  python "$SCORER" \
      --target-model "${TARGETS[$NAME]}" \
      --parent-model "$PARENT" \
      --train-file "$DATA/train.jsonl" \
      --test-file "$DATA/test.jsonl" \
      --output-dir "$SCORES/${PREFIX}${NAME}_logprob" \
      --n-aug 8 --batch-size 16
done

# Control 1: Random Init
echo "Scoring control: random_init"
python "$SCORER" \
    --target-model "$PARENT" \
    --parent-model "$PARENT" \
    --random-init --random-seed 0 \
    --train-file "$DATA/train.jsonl" \
    --test-file "$DATA/test.jsonl" \
    --output-dir "$SCORES/${PREFIX}random_init_logprob" \
    --n-aug 8 --batch-size 16

# Control 2: DeiT
echo "Scoring control: deit"
python "$SCORER" \
    --target-model "facebook/deit-base-patch16-224" \
    --parent-model "$PARENT" \
    --train-file "$DATA/train.jsonl" \
    --test-file "$DATA/test.jsonl" \
    --output-dir "$SCORES/${PREFIX}deit_logprob" \
    --n-aug 8 --batch-size 16

# Step 3: Run Sweeps
echo ""
echo "=== Step 3: Running Advantage Sweeps (Target vs Controls) ==="
for NAME in self pets cifar10 cifar100 eurosat celeba har; do
  for CTRL in random_init_logprob deit_logprob; do
    echo "Sweep for ${NAME}_logprob vs $CTRL"
    python experiments/table_01_imagenet/runners/sweep_shard_size.py \
        --scores-root "$SCORES" \
        --prefix "$PREFIX" \
        --positive-run "${NAME}_logprob" \
        --negative-run "$CTRL" \
        --score-key aug_logprob_mean \
        --m-values "40,80,160,320,640,1280,1920,3850" \
        --n-seeds 30 \
        --output-dir "$ARTIFACT_ROOT/results/replicated_sweep_aug_logprob_${NAME}_vs_${CTRL}"
  done
done

# Step 4: Generate Report
echo ""
echo "=== Step 4: Compiling Replicated Table 1 ==="
if [ -f "experiments/table_01_imagenet/reports/compile.py" ]; then
  python experiments/table_01_imagenet/reports/compile.py
else
  echo "Warning: Table 1 compiler not found."
  echo "Please write the reporting script to view the compiled table."
fi

echo "=========================================================="
echo " Table 1 Replication Pipeline Completed Successfully!"
echo "=========================================================="

# Causal Model Provenance

This repository audits whether a model retains evidence of a candidate
pretraining shard. The paper calls this a causal model provenance audit: create
member and control shards, score a parent or target model, calibrate a
threshold on one split, and measure held-out shard advantage on another.

## Start Here

Choose the research question, then open its experiment folder. Each folder
contains its own README, configuration, Slurm jobs, and table compiler.

| Question | Experiment folder | Paper table |
|---|---|---|
| Do downstream ViTs retain ImageNet shard membership? | `experiments/table_01_imagenet/` | 1 |
| Do fine-tuned Pythia targets retain GitHub membership? | `experiments/table_02_target_provenance/` | 2 |
| Do base Pythia models show the expected parent signal? | `experiments/table_03_parent_signal/` | 3 |
| Which membership score works on WikiMIA? | `experiments/table_04_wikimia_score_selection/` | 4 |
| Does the signal persist across MIMIR domains? | `experiments/table_05_multidomain/` | 5 |

Read [the experiment map](docs/EXPERIMENTS.md) for the precise paper-to-code
mapping and [the architecture guide](docs/architecture/repository-map.md) for
the repository layout.

## Running On PACE

```bash
bash scripts/cluster/install_pace.sh
export HF_TOKEN=<your_huggingface_token>
python scripts/setup/check_env.py
```

Then validate an experiment configuration and submit its focused job:

```bash
python scripts/setup/validate_experiment_config.py \
  experiments/table_02_target_provenance/configs/augmented.yaml
sbatch experiments/table_02_target_provenance/jobs/run_augmented_targets_1b_1_4b.sbatch
```

## Repository Layout

```text
experiments/        Human-facing research packages: Table 1-5 and exploration
scripts/data/       Reusable dataset preparation commands
scripts/scoring/    Reusable language-model scoring commands
scripts/audit/      Reusable threshold-audit commands
scripts/cluster/    HPC environment bootstrap
scripts/setup/      Local setup and configuration validation
src/shard_audit/    Shared audit, scoring, dataset, and reporting logic
artifacts/          New generated runs (ignored by Git)
.cache/             Downloaded model and dataset cache (ignored by Git)
```

All generated run state belongs in `artifacts/<experiment-id>/<run-id>/`; see
[artifact conventions](docs/architecture/artifacts.md).

## Core Terms

- **Member:** an example drawn from a candidate training shard.
- **Control / nonmember:** an example outside that shard; a null control may use
  two nonmember groups.
- **Score:** evidence that an example is member-like, such as MIN-K log-prob.
- **Shard advantage:** held-out TPR minus FPR after threshold calibration.
- **Paper replication:** follows a paper table's intended setup.
- **Augmented variant:** modifies scoring or data augmentation and must not be
  interpreted as an exact paper result.

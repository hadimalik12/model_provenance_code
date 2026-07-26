# Experiment Map

The repository is organized by research question. An experiment folder is the
single place to find its motivation, configuration, Slurm jobs, and report
compiler. Shared implementation belongs in `src/shard_audit/` and reusable CLI
commands remain under `scripts/`.

| Paper table | Research question | Canonical folder | Slurm entrypoint | Result type |
|---|---|---|---|---|
| 1 | ImageNet ViT shard provenance | `experiments/table_01_imagenet/` | `jobs/run_augmentation_stability.sbatch` | Augmented paper-style replication |
| 2 | Fine-tuned Pythia target provenance | `experiments/table_02_target_provenance/` | `jobs/targets_*_aug.sbatch` | Augmented variant; vanilla compiler retained |
| 3 | Parent Pythia shard-signal sanity check | `experiments/table_03_parent_signal/` | `jobs/parents_*_aug.sbatch` | Augmented variant with nonmember control |
| 4 | WikiMIA score selection | `experiments/table_04_wikimia_score_selection/` | None yet | Per-model runs and compilers |
| 5 | MIMIR multi-domain signal | `experiments/table_05_multidomain/` | `jobs/run_conservative_policy.sbatch` | Conservative-policy variant |

## Important Naming Correction

Older files named the MIMIR GitHub parent-signal work as "Table 4." That was
a historical filename, not a correct paper mapping. It is now Table 3 under
`experiments/table_03_parent_signal/`. Paper Table 4 is the WikiMIA experiment.

## Running Pattern

1. Read the experiment README.
2. Validate the selected YAML configuration.
3. Submit the job from that same experiment folder.
4. Find generated data, scores, result JSON, and the compiled table in the
   configured artifact directory. Historical reports still read legacy paths.

The core audit is always the same: prepare a labeled member/control split,
score a model, calibrate a threshold on training/calibration data, and report
held-out shard advantage plus a null control where applicable.

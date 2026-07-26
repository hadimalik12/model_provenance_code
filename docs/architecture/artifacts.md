# Artifact Conventions

Every new experiment run has one directory:

```text
artifacts/<experiment-id>/<run-id>/
  config.lock.yaml   Exact config used for the run
  prepared/          Dataset manifests and train/test JSONL
  scores/            Per-example score files and score manifest
  results/           Distinguisher JSON and metrics
  reports/           Compiled table, CSV, figures, summary
  logs/              Slurm stdout/stderr and command logs
```

The `experiment-id` comes from `experiment.id` in the YAML. The `run-id`
identifies the model group, score variant, and seed, for example
`augmented_seed0`. The artifact tree is ignored by Git, but every run should
write a copy of its configuration as `config.lock.yaml` before scoring starts.

The older `data/processed`, `data/scores`, and `outputs/runs` layouts were
removed. Do not recreate them; use the artifact structure above for every run.

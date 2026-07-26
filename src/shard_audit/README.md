# Shared Python Modules

`shard_audit` contains implementation used across experiments. It should not
contain paper-table-specific model lists, Slurm behavior, or hard-coded output
directories.

| Module | Responsibility |
|---|---|
| `data/` | Data loading, normalization, deterministic splits, and checks |
| `scoring/` | Model scoring primitives and augmentation policies |
| `auditing/` | Threshold calibration, evaluation, and metrics |
| `reporting/` | Result readers and report helpers |
| `experiments/` | Shared paper metadata and artifact-directory layout |

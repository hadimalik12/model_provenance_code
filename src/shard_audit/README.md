# Shared Python Modules

`shard_audit` contains implementation used across experiments. It should not
contain paper-table-specific model lists, Slurm behavior, or hard-coded output
directories.

| Module | Responsibility |
|---|---|
| `data/` | Data loading, normalization, deterministic splits, and checks |
| `scoring/` | Language-model and vision scoring primitives |
| `auditing/` | Threshold calibration, evaluation, and metrics |
| `experiments/` | Shared submitted-paper metadata |

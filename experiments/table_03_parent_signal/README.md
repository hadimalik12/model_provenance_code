# Table 3: Parent Shard Signal

This is the parent-model sanity check: base Pythia models are audited directly
against their MIMIR GitHub shard, with a nonmember-versus-nonmember control.
The previous filenames called this "Table 4"; that label was incorrect and is
retired here. WikiMIA score selection has its own Table 4 folder.

```bash
python scripts/setup/validate_experiment_config.py experiments/table_03_parent_signal/configs/augmented.yaml
sbatch experiments/table_03_parent_signal/jobs/run_augmented_parents_1b_1_4b.sbatch
```

`compile.py` is the canonical report command. It writes both the historical
output filenames and clearly named parent-signal aliases for existing results.

# Table 2: Fine-Tuned Target Provenance

This experiment measures whether fine-tuned Pythia targets retain their
parent's MIMIR GitHub membership signal. The three jobs partition targets by
model size so they can be scheduled independently.

```bash
python scripts/setup/validate_experiment_config.py experiments/table_02_target_provenance/configs/augmented.yaml
sbatch experiments/table_02_target_provenance/jobs/run_augmented_targets_1b_1_4b.sbatch
```

Use `compile_paper.py` only for paper-style vanilla results. The active jobs
produce the augmented variant and compile it with `compile_augmented.py`.

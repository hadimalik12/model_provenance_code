# Table 2: Fine-Tuned Target Provenance

This experiment measures whether fine-tuned Pythia targets retain their
parent's MIMIR GitHub membership signal. The three jobs partition targets by
model size so they can be scheduled independently.

## Reproduction

The paper-style Table 2 report is produced by the local vanilla runner. Run it
on an allocated GPU node, then compile the Markdown and CSV report:

```bash
python experiments/table_02_target_provenance/runners/run_local_vanilla_audit.py \
  --seeds 0 --batch-size 4 --dtype float16
python experiments/table_02_target_provenance/reports/compile_local_vanilla_audit.py
```

The runner fixes the six target checkpoints, their declared 1B/1.4B parents,
the MIMIR GitHub split seed, MIN-K20 and mean-log-probability scorers, matched
nonmember controls, and shuffled-label controls. Its report is written to
`artifacts/table_02_target_provenance/local_vanilla/reports/`.

For a resumable rerun, add `--resume`. To run additional pre-specified split
seeds, pass `--seeds 0,1,2` and compile after all runs finish.

## Augmented variant

```bash
python scripts/setup/validate_experiment_config.py experiments/table_02_target_provenance/configs/augmented.yaml
sbatch experiments/table_02_target_provenance/jobs/run_augmented_targets_1b_1_4b.sbatch
```

The Slurm command above runs a separate augmentation-based variant and compiles
it with `compile_augmented.py`; it does not reproduce the local-vanilla Table 2
report described above.

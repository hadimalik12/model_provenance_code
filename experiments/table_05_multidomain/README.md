# Table 5: Multi-Domain Signal

This is a conservative augmentation rerun of paper Table 5. For every domain, the target
transformer body is evaluated through the parent Pythia output head, matching
the paper's parent-head protocol. Each example receives four domain-specific
text augmentations before scoring. It also retains the paper-style vanilla
MIN-K scores and reports one ablation per domain-specific transformation.

```bash
python scripts/setup/validate_experiment_config.py experiments/table_05_multidomain/configs/augmented.yaml
sbatch experiments/table_05_multidomain/jobs/run_conservative_policy.sbatch
```

The job uses the paper's parent/target pair, 300 construction examples per
class, and 400 held-out examples per shard. It writes two independent tables:
`table_5_mink20` and `table_5_mink40`. Each table contains vanilla MIN-K, the
four-view conservative full policy, and its per-domain ablations. The full
policy uses only deterministic, snippet-safe transformations and never changes
numbers, deletes prose, masks entities, or reorders text.

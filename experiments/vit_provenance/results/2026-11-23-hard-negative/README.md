# AISTATS 2027 independent ImageNet hard-negative control

## Status

Observed production result: the independently published DeiT-Base hard
negative was not flagged. Its signed held-out shard advantage was 0.0083,
below the level-0.05 cutoff of 0.04378 over 1,925 evaluation pairs. The
shuffled-label diagnostic had advantage 0.00208.

Completed on 2026-11-23:

- smoke: Slurm `16063098`;
- production: Slurm `16063127` (9:15 on one H200, or 0.154 H200-hours);
- resumability verification: Slurm `16063135`.

The compact observed values are recorded in `observed_summary.json`. Full
per-example scores remain under the ignored artifact directory rather than
being copied into Git.

## Design

- Target: `facebook/deit-base-patch16-224` at immutable revision
  `fb2c78a54a5637dec350432794f7b93e31f910c9`.
- Rationale: Meta DeiT-Base is a separately published ImageNet-1K transformer
  checkpoint and not a fine-tune of `google/vit-base-patch16-224`. It is a hard
  negative because it shares ImageNet-1K training exposure.
- Data: the exact class-balanced ImageNet train/validation shard, construction
  split, evaluation split, and seed used by the submitted public-ViT audit.
- Audit: the same eight-view CLV scorer, construction-only calibration,
  held-out evaluation, level 0.05 cutoff, and shuffled-label diagnostic.

The runner validates the immutable data manifest and refuses to silently use a
different split. The smoke output is explicitly marked as non-scientific.

## Commands

```bash
python scripts/setup/validate_experiment_config.py \
  experiments/vit_provenance/configs/aistats27_hard_negative.yaml

sbatch -A gts-vzikas3 --export=ALL,HARD_NEGATIVE_MODE=smoke \
  experiments/vit_provenance/jobs/run_hard_negative.sbatch

sbatch -A gts-vzikas3 experiments/vit_provenance/jobs/run_hard_negative.sbatch
```

## Interpretation fixed before evaluation

A positive result indicates detection of a shared ImageNet data signal rather
than exact lineage. A negative result supports specificity for this hard
negative, but does not prove lineage. This is one independent ImageNet control,
so it is not a precise estimate of a population false-positive rate.

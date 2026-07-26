# Table 1: ImageNet Shard Provenance

This experiment asks whether a ViT fine-tuned on a downstream task retains a
membership signal for images from its ImageNet pretraining shards. It is the
repository home for paper Table 1 and the current augmentation-based variant.

Start here:

```bash
python scripts/setup/validate_experiment_config.py experiments/table_01_imagenet/configs/augmented.yaml
sbatch experiments/table_01_imagenet/jobs/run_augmentation_stability.sbatch
```

The job prepares the balanced ImageNet shard split, scores target and control
models, runs the held-out advantage sweep, then compiles the table. Results are
currently read from the legacy `data/` and `outputs/` paths; new runs should use
the artifact layout described in `docs/architecture/artifacts.md`.

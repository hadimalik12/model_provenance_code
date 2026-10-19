# ViT provenance-witness audit

This package corresponds to Section 5.2.2 and the submitted paper's ViT
results:

- main-paper Table 3;
- Appendix D.3 Table 8, public ViT derivatives; and
- Appendix D.4 Table 9, model transformations.

The candidate parent is `google/vit-base-patch16-224`. The secret and control
shards are drawn from ImageNet-1K training and validation images over 77
classes. Construction and evaluation are disjoint; held-out evaluation has 25
images per class from each shard, so `m = 1,925` and the level-0.05 threshold is
`0.0438`.

The black-box centered-logit-variance (CLV) scorer queries each target under
eight fixed deterministic weak views. It centers log probabilities within each
view and reports the negative population variance across views, averaged over
the target's output classes. It uses only output score vectors—never weights,
activations, the parent classifier head, or ground-truth labels.

## Public derivatives (Appendix Table 8)

Run the parent reference, six public derivatives, and random-initialization
sanity check:

```bash
python scripts/setup/validate_experiment_config.py \
  experiments/vit_provenance/configs/clv.yaml
sbatch experiments/vit_provenance/jobs/run_public_derivatives.sbatch
```

Site-specific scheduler settings can be supplied through normal `sbatch`
overrides such as `--account` and `--partition`. Generated files are written to
`artifacts/vit_provenance/public_derivatives_seed0/`.

The submitted values are versioned in
[`results/submitted_table_8.csv`](results/submitted_table_8.csv).

## Model transformations (Appendix Table 9)

The submitted transformation values are versioned in
[`results/submitted_table_9.csv`](results/submitted_table_9.csv). They cover
Linear Soup, TIES, DARE-TIES, SLERP, simulated 8/4/3-bit quantization, and
30/50/70/90% magnitude pruning of the CIFAR-10 derivative.

No transformation-construction source was present in this repository or in the
teammate repository used to recover the watermark experiment. The submitted
PDF does not specify the three exact merge inputs or all merge hyperparameters,
so this cleanup does not invent a purportedly exact reproduction job. The CLV
audit implementation and fixed evaluation protocol are shared with Table 8 and
are available here; reconstructing the transformed checkpoints requires the
original run configuration.

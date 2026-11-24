# AISTATS 2027 Q3 scorer and split-diagnostic run

## Observed scorer comparison

The scorer comparison is complete and reuses the same cached per-example model
outputs and matched split. It performs no additional model forward passes.

| Fixed scorer | Available public targets positive | Nonmember-control false alarms |
|---|---:|---:|
| MIN-K 20% | 13/13 | 0/13 |
| Mean log probability | 12/13 | 0/13 |

`kykim0/pythia-1.4b-tulu-v2-mix` is positive with MIN-K and does not reject
with mean log probability. The four parent sanity checks are positive under
both scorers. The submitted fourteenth target,
`nnheui/pythia-1.4b-sft-full`, is unavailable at its immutable checkpoint and
was skipped without substitution.

Source:
`artifacts/public_pythia_derivatives/aistats27_matched/reports/public_pythia_audit.csv`.

## New split diagnostics

Both production diagnostics are now complete.

| Dataset | Examples per side | Semantic C2ST balanced accuracy | Permutation p-value | Main measured shift |
|---|---:|---:|---:|---|
| Pile/MIMIR GitHub | 740 | 0.610 | 0.00498 | character-count SMD -0.176 |
| ImageNet | 3,850 | 0.544 | 0.00498 | width SMD 0.412 |

The Pile comparison contained no exact or near cross-shard duplicates. The
ImageNet comparison contained two exact cross-shard duplicate pairs and no
additional perceptual near duplicates. ImageNet class frequencies were exactly
balanced. Both semantic classifiers nevertheless distinguished the public
training and validation shards better than their permutation nulls.

MIMIR GitHub:

```bash
.venv_pace/bin/python \
  experiments/public_pythia_derivatives/runners/diagnose_mimir_splits.py \
  --resume
```

Fast MIMIR smoke run:

```bash
.venv_pace/bin/python \
  experiments/public_pythia_derivatives/runners/diagnose_mimir_splits.py \
  --max-per-class 32 --permutations 20 \
  --output-dir artifacts/split_diagnostics/aistats27_q3/mimir_github_smoke \
  --resume
```

ImageNet, after preparing the exact split used by the ViT audit:

```bash
.venv_pace/bin/python \
  experiments/vit_provenance/runners/diagnose_imagenet_splits.py \
  --resume
```

Cluster submissions:

```bash
sbatch --account=gts-vzikas3 --partition=cpu-small \
  experiments/public_pythia_derivatives/jobs/smoke_split_diagnostics_cpu.sbatch
sbatch --account=gts-vzikas3 --partition=gpu-h200 \
  --export=ALL,SPLIT_DIAG_SMOKE=1 \
  experiments/public_pythia_derivatives/jobs/run_split_diagnostics.sbatch
sbatch --account=gts-vzikas3 --partition=gpu-h200 \
  experiments/public_pythia_derivatives/jobs/run_split_diagnostics.sbatch
sbatch --account=gts-vzikas3 --partition=gpu-h200 \
  experiments/vit_provenance/jobs/run_split_diagnostics.sbatch
```

Both runners cache frozen embeddings and validate record identities when
`--resume` is used. No Pythia inference is performed. The ImageNet runner
refuses to run if the original prepared image split is absent rather than
silently downloading or substituting different data.

These diagnostics can detect measured structural, lexical/visual, duplicate,
or semantic differences. Failure to detect a difference cannot prove equality
or exchangeability and does not establish exact parentage.

The watermark experiment does not receive another embedding test. Its
construction already verifies fixed prompts/configuration, independent
sampling seeds, and the fresh S-double-prime draw. Those checks are the
appropriate lightweight construction verification for S, S-prime, and
S-double-prime.

## Reconciliation notes

The matched-v3 sensitivity artifact contains 340 member and 340 nonmember
construction examples plus 400 per side for evaluation. This differs from the
public package's nominal 560-member/340-nonmember construction description and
must be described as the matched sensitivity run, not silently as the exact
submitted construction split. The scorer comparison above uses this same
matched-v3 data for both scorers, so the within-run comparison remains valid.

No prepared ImageNet artifact was present when this work began. The exact
submitted preparation was reconstructed from the pinned dataset revision; no
substitute image checkpoint or dataset was used.

## Run status

The full MIMIR semantic diagnostic completed in Slurm job `16063140`, and the
full ImageNet diagnostic completed in job `16064818`. Compact results are in
`mimir_structural_summary.json`, `mimir_semantic_summary.json`, and
`imagenet_observed_summary.json`. The observed split differences mean that the
public audits must be described as data-signal sensitivity results. Matching
measured properties does not restore a general exchangeability or lineage
guarantee.

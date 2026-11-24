# AISTATS-2027 reviewer analysis ledger

This directory separates observed reanalysis from jobs that have only been
prepared.  It does not modify the submitted `2026-10-06` result bundles.

## Compact results mapped to reviewer concerns

| Concern | Analysis unit | Status | Result or deliverable | Establishes | Does not establish |
|---|---|---|---|---|---|
| Multiple audits | 12 primary final-checkpoint audits; 48 exploratory checkpoint audits | **Observed reanalysis** | Holm rejects 6/6 primary targets and 0/6 primary controls; in the 48-test family Holm and Bonferroni reject 27/27 target rows and 0/21 control rows. Bonferroni paired cutoff is 0.04347. | Submitted decisions survive FWER control valid under arbitrary dependence. | The 48 correlated rows are not 48 independent model/shard trials. |
| False positives | One independently seeded model and shard per unit | **Configured, unrun** | 12 new Dolly controls and 3 new UltraChat controls; separate model/shard seeds. If there are zero alarms, the one-sided 95% bounds would be 18.1% for 15 Dolly units and 39.3% for 6 UltraChat units, including submitted units. | A direct estimate of false alarms across independently trained controls once run. | Planned precision is not observed performance; it cannot prove a 5% population rate. |
| Sample size/checkpoints | Whole held-out prompt pairs; repeat subsets are sensitivity analyses | **Runnable, unrun** | Frozen-cache pipeline for 250, 500, 1,000, 2,000, 4,000, and 8,000 pairs, Dolly and UltraChat checkpoints, watermark and MIN-K20. | Sensitivity to query count and training exposure once canonical caches exist. | Repeated subsets are not independent estimates of population power. |
| Scorer/cost | Same model, split, and cached forward output | **Controlled comparison observed; public comparison unrun** | At final Dolly and UltraChat checkpoints, watermark and MIN-K20 each reject 3/3 targets and 0/3 controls per dataset. They share forward passes. New manifests retain passes, sequences, tokens, and wall time. | Controlled scorer dependence without confounding by a new model or split. | Historical token/wall-time totals were not retained; public scorer agreement is not yet observed and would not establish lineage. |
| Public Pile split | 740 matched member/nonmember pairs from pinned MIMIR GitHub files | **Preparation observed; model audit unrun** | Exact duplicates: 0; near duplicates at token-5-shingle Jaccard ≥0.8: 0. Character-count SMD improves from −0.309 to −0.176; line-count from −0.180 to −0.104; code-symbol fraction from 0.085 to 0.054. | A documented measured-covariate sensitivity analysis. | Matching cannot establish exchangeability, exact parentage, or the route by which witness data was learned. |

Public Pythia results are interpreted only as witness/data-signal detection.
When an independently trained model can access the same witness data, a
positive result cannot distinguish inheritance from independent exposure.

## Exact commands

Static configuration and the observed multiple-testing reanalysis:

```bash
.venv_pace/bin/python -m experiments.watermark_inheritance.reviewer_configure
.venv_pace/bin/python -m experiments.watermark_inheritance.reviewer_report \
  multiplicity \
  --output-dir experiments/watermark_inheritance/results/2026-11-21-reviewer-analysis
.venv_pace/bin/python -m experiments.watermark_inheritance.reviewer_report \
  scorers \
  --output-dir experiments/watermark_inheritance/results/2026-11-21-reviewer-analysis
```

GPU pilot and, only after checking its throughput and outputs, production:

```bash
sbatch experiments/watermark_inheritance/jobs/aistats27_false_positive_pilot.sbatch
# Inspect the pilot, update estimates if needed, then:
bash experiments/watermark_inheritance/submit_aistats27_controls.sh
```

Frozen-cache curves use one argument per checkpoint:

```bash
.venv_pace/bin/python -m experiments.watermark_inheritance.reviewer_report curves \
  --run-spec RUN_DIR:CHECKPOINT:target \
  --run-spec CONTROL_RUN_DIR:CHECKPOINT:control \
  --output-dir artifacts/watermark_inheritance/aistats27_reviewer/reports
```

## Compute and query budget

The submitted controlled production job measured 3.934 H200 GPU-hours (two
H200s for 1:58:04) for three shards, parent/target/control training, and 18
checkpoint prediction caches.  The selected new design estimates 7.2 H200
GPU-hours for Dolly controls and 9.0 for UltraChat controls: 16.2 total, with a
12–24 H200-hour planning range and about 110 GB of checkpoints.  Each scored
model cache records exact forward passes, sequences, input/scored tokens, model
load time, scoring time, and total wall time.  No production GPU jobs were
submitted in this work session because scheduler accounting was unavailable;
these figures are estimates, not observed new costs.

## Discrepancies and missing immutable artifacts

1. The paper describes the public comparison as Pile training versus
   validation.  The pinned MIMIR inputs are member and nonmember JSON-string
   caches and expose no repository, path, author, time, commit, or license
   metadata.
2. The pinned public files contain 963 eligible unique member snippets and 740
   nonmembers after preprocessing.  Therefore the submitted asymmetric
   560/340 construction plus 400/400 evaluation fits, but the matched
   sensitivity must use 340/340 plus 400/400.
3. `nnheui/pythia-1.4b-sft-full` is no longer served.  The reviewer job records
   it in `unavailable_checkpoints.json` and skips it without substitution.
4. The committed controlled bundles contain compact aggregate tables but not
   the canonical per-example score caches or checkpoints.  New sample-size
   curves cannot honestly be reported until those caches are restored or new
   jobs complete.
5. The submitted run manifest records PyTorch 2.8.0 and Transformers 4.57.1,
   while the repository requirements pin 2.2.0 and 4.44.2.  The submitted
   `packages.txt` and input revision manifest, not the generic requirements,
   describe that observed run.
6. The 48 checkpoint rows reuse only three controlled witnesses and model
   trajectories.  They are a multiplicity family, not independent false-alarm
   or power trials.

The loose Type-II bound is a separate theory issue.  These empirical curves
measure sensitivity and do not repair or replace its proof.

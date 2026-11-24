# Reviewer 3 Q4: false-alarm calibration

This is an observed CPU-only reanalysis of the frozen per-example watermark
scores. It did not train or load a model and did not run new GPU inference.
The construction-fitted detector threshold stays fixed. Only whole held-out
prompt pairs are subsampled, and the paired hypothesis-test cutoff is
recomputed for each prespecified alpha and sample size.

## Primary observed results

| Family | Independent units | Primary false alarms at alpha=.05, m=8,000 | Holm FWER alarms | One-sided 95% upper bound |
|---|---:|---:|---:|---:|
| Direct-base Dolly/UltraChat model controls | 15 (12 + 3) | 0/15 | 0/15 | 18.1% |
| Fresh S'-versus-S'' controls | 6 | 0/6 | 0/6 | 39.3% |
| Cached checkpoint sensitivity | 15 clusters, 54 correlated rows | 0/54 | 0/54 | Not a binomial sample |

The manuscript's primary decision is the conservative paired Hoeffding rule
`abs(gap) > sqrt(2 log(2/alpha) / m)`, whose alpha=.05, m=8,000 cutoff is
0.030368. The largest endpoint control advantage was 0.01575. An exploratory
exact sign test flagged that endpoint before multiplicity correction
(`p=0.04013`), but no exact-test result survived Holm correction. Both results
are retained because the exact test and conservative manuscript rule must not
be silently conflated.

This rule follows the repository's reported controlled paired protocol and its
saved `advantage = abs(signed_gap)` field. The paper's generic Figure 2 instead
appears to write a signed, one-sided decision (`estimated advantage > gamma`)
for the unpaired generic audit. That prose and the controlled paired reporting
should be reconciled in the rebuttal/manuscript. This reanalysis does not
reinterpret or overwrite any old saved decision.

The zero-event results do not show that the population false-positive rate is
below 5%. With only 15 independent model controls, the exact one-sided 95%
upper bound remains 18.1%. The six fresh-shard controls answer a different
exchangeability question and are not pooled with the 15 model controls.

## Sample-size, alpha, and contamination sensitivity

The prespecified grid is alpha in `{.01,.025,.05,.10}`, `m` in
`{250,500,1000,2000,4000,8000}`, and contamination in
`{0,5,10,20,40}%`. There are 25 fixed-seed subsamples per unit below 8,000;
the full 8,000-pair set is evaluated once. At alpha=.05 without contamination,
the conservative rule fired in 1/375 correlated model-control subsamples and
1/150 correlated fresh-shard subsamples at m=250, and 0 times for every
prespecified m from 500 through 8,000. These repeated subsets are sensitivity
analyses, not independent false-positive trials.

Contamination replaces the second/control-side score with the corresponding
first-side score for exactly `floor(rate*m)` fixed-seed pairs. This models
witness leakage into the control side and sets those paired differences to
zero. It is a non-adversarial attenuation analysis, not a model of arbitrary
semantic or adversarial contamination. Full unit-clustered results are in
`unit_clustered_sensitivity.csv`; aggregate sensitivity frequencies are in
`sensitivity_summary.csv`.

## Witness and checkpoint reuse

The available caches contain 54 production checkpoint audits grouped into 15
model-seed/shard-seed trajectories. They have 0/54 conservative false alarms,
but the rows are correlated and do not provide 54 independent trials. The
caches do not contain several independently trained models scored against the
identical witness shard. That shared-witness cross-model experiment remains
unrun because it requires new model inference.

## Reproduction

From the repository root:

```bash
.venv_pace/bin/python -m unittest \
  experiments.watermark_inheritance.tests.test_calibration_analysis \
  experiments.watermark_inheritance.tests.test_null_control -v

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv_pace/bin/python -m experiments.watermark_inheritance.calibration_analysis \
  --root artifacts/watermark_inheritance/aistats27_reviewer \
  --config experiments/watermark_inheritance/configs/aistats27_q4_calibration.json \
  --output-dir experiments/watermark_inheritance/results/2026-11-23-reviewer-q4-calibration
```

The exact cache hashes are recorded in `summary.json`.

## Discrepancy found

The previously saved S'/S'' metrics used `sqrt(log(2/alpha)/m)` for a field
named `single_audit_advantage_cutoff`, missing the manuscript's factor of two
inside the square root. The helper is corrected for future runs. This analysis
recomputed the manuscript cutoff from frozen decisions. All six old and new
S'/S'' decisions remain negative, so the correction changes no observed
conclusion. Immutable old metrics were not overwritten.

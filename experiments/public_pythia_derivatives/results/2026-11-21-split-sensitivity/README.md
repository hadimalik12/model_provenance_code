# MIMIR GitHub matched-split preparation

Status: data inspection and matching **observed**; model scoring **not run**.

The pinned MIMIR files contain JSON strings only.  No repository, file path,
author, timestamp, commit, or license field is available.  After the existing
8–64-word filter and within-side deduplication there are 963 member and 740
nonmember snippets.  There are no cross-label exact duplicates and no
cross-label pairs with normalized token 5-shingle Jaccard similarity at least
0.8.

One-to-one matching uses word count, character count, line count, and code
symbol fraction.  The sensitivity split contains 340/340 construction and
400/400 evaluation examples.  Standardized mean differences change as follows:

| Feature | Before | After matching |
|---|---:|---:|
| Word count | 0.000 | 0.000 |
| Character count | −0.309 | −0.176 |
| Line count | −0.180 | −0.104 |
| Code-symbol fraction | 0.085 | 0.054 |

Reproduce preparation and then run all available checkpoints with:

```bash
.venv_pace/bin/python experiments/public_pythia_derivatives/runners/run_audit.py \
  --families 1b,1.4b,6.9b,12b --seeds 0 \
  --artifact-root artifacts/public_pythia_derivatives/aistats27_matched \
  --data-variant matched --calibration-members 340 \
  --calibration-nonmembers 340 --prepare-only --resume

bash experiments/public_pythia_derivatives/submit_aistats27_matched.sh
```

Regenerate the measured-covariate balance figure with:

```bash
.venv_pace/bin/python \
  experiments/public_pythia_derivatives/reports/plot_split_diagnostics.py \
  --diagnostics artifacts/public_pythia_derivatives/aistats27_matched/seed_0/prepared/mimir_github_matched_v3/diagnostics.json \
  --output-dir experiments/public_pythia_derivatives/results/2026-11-21-split-sensitivity
```

The production job uses MIN-K20 as the frozen primary scorer and mean log
probability as the frozen secondary scorer, both from identical forward
passes.  It includes shuffled-label and nonmember-versus-nonmember diagnostics.
It explicitly skips and records the unavailable `nnheui` checkpoint; it does
not substitute another model.

For 13 available targets and four parents, the plan creates 30 score caches
(main plus target controls): 44,400 queried sequences, 20,720 model forward
passes at the configured family batch sizes, and at most 22,732,800 input
tokens at length 512. The planning range is 12–36 H200 GPU-hours; the four job
reservations have a conservative 96 H200-hour ceiling. Exact tokens and wall
time will come from manifests and are not reported as observed yet.

Matching observed properties does not establish full exchangeability.  A
positive public-model audit detects the witness/data signal and cannot establish
exact parentage or the route by which the model encountered that signal.

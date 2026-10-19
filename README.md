# Causal Model Provenance

This repository contains the experiment code and compact published results for
the submitted AISTATS 2027 paper *Causal Model Provenance via Shard Membership
Auditing*. The audit asks whether a target retains a witness carried by a
candidate parent's training shard.

## Submitted experiments

Directory names describe the scientific experiment rather than a table number:

| Experiment | Paper coverage | Entry point |
|---|---|---|
| [`watermark_inheritance/`](experiments/watermark_inheritance/) | Table 1; Appendix Tables 4–5 | `submit.sh`, `submit_ultrachat.sh` |
| [`public_pythia_derivatives/`](experiments/public_pythia_derivatives/) | Table 2 public derivatives; Appendix Table 6 | `submit.sh` |
| [`pythia_int8_quantization/`](experiments/pythia_int8_quantization/) | Table 2 INT8 summary; Appendix Table 7 | `jobs/run_paper_targets.sbatch` |
| [`vit_provenance/`](experiments/vit_provenance/) | Table 3; Appendix Tables 8–9 | `jobs/run_public_derivatives.sbatch` |

[The experiment map](docs/EXPERIMENTS.md) records the exact main/appendix
mapping and the availability of each reproduction pipeline.

The [controlled watermark-inheritance manuscript fragment](docs/manuscript/watermark_inheritance/README.md)
provides the proposed Section 5.3.1 with verified run-by-run Dolly and UltraChat
tables and instructions for insertion into the manuscript source.

## Setup on PACE

```bash
bash scripts/cluster/install_pace.sh
export HF_TOKEN=<your_huggingface_token>
python scripts/setup/check_env.py
```

`requirements.txt` pins the validated package versions. Each experiment keeps
its immutable model, dataset, and external-code revisions directly beside the
commands that download them.

Each experiment README gives its full command. Scheduler account and partition
choices are supplied through documented environment variables or normal
`sbatch` command-line overrides; no personal account is committed.

## Repository layout

```text
experiments/        Submitted-paper experiments and compact result tables
scripts/data/       Reusable MIMIR data preparation
scripts/scoring/    Reusable language-model scoring
scripts/audit/      Threshold calibration and held-out evaluation
scripts/cluster/    PACE environment bootstrap
scripts/setup/      Environment and configuration checks
src/shard_audit/    Shared audit, data, scoring, and experiment metadata
artifacts/          Generated runs and checkpoints (ignored by Git)
.cache/             Downloaded models and datasets (ignored by Git)
```

Generated datasets, checkpoints, and raw scores belong under `artifacts/`.
Only compact result tables and validation manifests needed to audit paper claims
are versioned.

## Verification

```bash
.venv_pace/bin/python -m pytest -q
find experiments scripts -type f \( -name '*.sh' -o -name '*.sbatch' \) \
  -exec bash -n {} +
```

The watermark package additionally contains tiny-model compatibility, resume,
scoring, and exact published-result tests under
`experiments/watermark_inheritance/tests/`.

## Core terms

- **Secret shard:** examples that carry the candidate witness.
- **Control shard:** matched examples that do not carry that witness.
- **Construction split:** data used to select and freeze the distinguisher.
- **Evaluation split:** held-out data used only for the final test.
- **Shard advantage:** held-out true-positive rate minus false-positive rate.
- **Witness detected:** the signed held-out advantage exceeds the paper's fixed
  level-0.05 threshold.

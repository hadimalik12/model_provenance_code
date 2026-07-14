# Phase 1 — Repository Scaffold

**Status:** Complete  
**Date:** 2026-05-06

---

## Purpose

Phase 1 establishes the repository skeleton for the MIMIR GitHub / Pythia
membership-auditing sanity check.  No model weights are downloaded and no
experiments are run in this phase.

---

## Directory Structure

```
model_provenance/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── configs/
│   ├── env/local.yaml                         ← runtime paths and device config
│   └── experiments/mimir_github_pythia_min_k.yaml  ← planned experiment config
│
├── docs/
│   ├── notes/                                 ← planning documents
│   └── paper/                                 ← reference PDFs
│
├── scripts/
│   ├── setup/
│   │   ├── setup_local_env.sh                 ← Phase 1 entrypoint
│   │   └── check_env.py                       ← environment checker
│   ├── local_scripts/
│   │   └── install_pace.sh                    ← HPC/PACE conda setup (existing)
│   ├── data/          ← Phase 2: data download scripts
│   ├── scoring/       ← Phase 2: scoring scripts
│   ├── experiments/   ← Phase 2: experiment runners
│   └── reports/       ← Phase 3: report generation scripts
│
├── src/
│   ├── __init__.py
│   └── shard_audit/                           ← membership auditing package
│       ├── __init__.py
│       ├── datasets.py       ← Phase 2
│       ├── preprocessing.py  ← Phase 2
│       ├── logprobs.py       ← Phase 2
│       ├── mia_scores.py     ← Phase 2  (MIN-K% PROB)
│       ├── distinguishers.py ← Phase 2
│       ├── metrics.py        ← Phase 2
│       ├── splitting.py      ← Phase 2
│       ├── reporting.py      ← Phase 3
│       └── sanity_checks.py  ← Phase 2
│
├── data/
│   ├── raw/       ← gitignored; populated in Phase 2
│   ├── processed/ ← gitignored
│   └── scores/    ← gitignored
│
└── outputs/
    ├── runs/      ← gitignored
    ├── reports/   ← gitignored
    └── figures/   ← gitignored
```

---

## Local Environment Setup

### Option A — PACE/HPC (recommended on cluster)

```bash
bash scripts/local_scripts/install_pace.sh
```

This script loads the `pytorch/25` and `anaconda3` modules and creates a
conda environment at `/tmp/python-venv/model_provenance_venv`.

### Option B — Local machine / generic Linux

```bash
bash scripts/setup/setup_local_env.sh
```

The wrapper auto-detects whether the HPC module system is present.
If not, it falls back to plain `pip install -r requirements.txt`.

### Standalone environment check

```bash
python scripts/setup/check_env.py
```

Prints Python version, installed package versions, GPU status, and
whether required directories exist.  Exits 0 on success, 1 if an
essential package is missing.

---

## Experiment Config (planned, not yet implemented)

See [configs/experiments/mimir_github_pythia_min_k.yaml](../../configs/experiments/mimir_github_pythia_min_k.yaml).

Key parameters:
- **Parent model:** `EleutherAI/pythia-1.4b`
- **Dataset:** MIMIR GitHub (member and nonmember splits)
- **Primary score:** MIN-K=20 log-prob
- **Primary metric:** held-out test accuracy

---

## What Phase 2 Will Implement

1. Load MIMIR GitHub member/nonmember examples (`src/shard_audit/datasets.py`).
2. Filter by word count (`src/shard_audit/preprocessing.py`).
3. Extract per-token log-probs from Pythia-1.4b (`src/shard_audit/logprobs.py`).
4. Compute MIN-K% PROB scores (`src/shard_audit/mia_scores.py`).
5. Split into calibration and held-out test sets (`src/shard_audit/splitting.py`).
6. Evaluate distinguisher accuracy (`src/shard_audit/metrics.py`).
7. Run pre-flight sanity checks (`src/shard_audit/sanity_checks.py`).

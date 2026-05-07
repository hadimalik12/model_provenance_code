# Model Provenance — Shard Membership Auditing

This repository implements a membership inference pipeline to audit whether a
fine-tuned language model inherited pretraining membership signals from its
parent model. The core question: can we tell that a fine-tuned model was built
on top of a specific parent that was trained on a specific data shard?

The pipeline supports the entire **Pythia family** (1B, 1.4B, 6.9B, and 12B) and
includes automated orchestrators for 14 different fine-tuned targets.

**Scale Support:** 1B, 1.4B, 6.9B, 12B  
**Primary Dataset:** MIMIR GitHub (`iamgroot42/mimir`, config=`github`)  
**Primary Metric:** `min_k_20_logprob`

---

## Requirements

- Python 3.11
- CUDA GPU (experiments run on Tesla V100-16GB)
- HuggingFace account with access to [`iamgroot42/mimir`](https://huggingface.co/datasets/iamgroot42/mimir)

---

## Installation

### On a PACE/HPC cluster

```bash
bash scripts/local_scripts/install_pace.sh
```

This creates a conda environment at `/tmp/python-venv/model_provenance_venv`.

### Activate the environment (every session)

```bash
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate /tmp/python-venv/model_provenance_venv
```

### Set HuggingFace credentials
```bash
export HF_HOME=$(pwd)/data/.hf_home
export HF_TOKEN=<your_hf_token>
```

### Verify the environment
```bash
python scripts/setup/check_env.py
```

---

## High-Scale Auditing (6.9B & 12B)

Running audits on 6.9B and 12B models requires significant resources. This repository includes several built-in protections:

1.  **Half-Precision (FP16)**: The model loader in `src/shard_audit/logprobs.py` forces `torch.float16` by default, cutting VRAM and System RAM usage in half.
2.  **Safe Pre-Downloading**: Each orchestrator script uses `snapshot_download(max_workers=1)` to ensure weights are safely on disk before loading starts. This prevents `SIGKILL` errors during heavy multi-threaded downloads.
3.  **Batch Size Control**: Scripts are pre-configured with optimal batch sizes:
    *   1.4B Models: Batch Size 4
    *   6.9B Models: Batch Size 2
    *   12B Models: Batch Size 1

### Managing the Model Cache
The 12B and 6.9B weight files are massive (14GB–48GB per model). If you run out of disk space, you should delete the weights of **finished** target models:

```bash
# Example: Remove a finished 6.9B target to free up 14GB
rm -rf data/.hf_home/hub/models--pkarypis--pythia-ultrachat
```
*Note: Keep the `EleutherAI/pythia-...` parent models until you have finished all targets of that size.*

## Repository Layout

```
scripts/
  data/
    prepare_mimir_github.py              # Prepare main member/nonmember dataset
    prepare_mimir_github_nonmember_control.py  # Prepare null-control dataset
  scoring/
    extract_logprob_scores.py            # Score a model on a prepared dataset
  experiments/
    run_mia_experiment.py                # Calibrate threshold and evaluate
  reports/
    compare_parent_target_advantage.py   # Parent vs target comparison report
    report_nonmember_control.py          # Null-control report

src/shard_audit/
  datasets.py        # MIMIR dataset loading
  preprocessing.py   # Text normalization and truncation
  splitting.py       # Stratified train/test splitting
  logprobs.py        # Per-token log-probability extraction
  mia_scores.py      # MIN-K% PROB and mean log-prob scoring
  distinguishers.py  # Threshold sweep, calibration, and evaluation
  sanity_checks.py   # Hash-overlap and balance checks
  metrics.py         # Score diagnostics (AUC, direction check)
```

---

## Reproducing the Experiments

The experiments are fully automated via orchestrator scripts in `scripts/experiments/`. Each script handles data preparation, model scoring (Main and Control), and final provenance reporting in a single command.

### 1. Run Parent Baselines
Establish the "ground truth" memorization for each scale:

```bash
python scripts/experiments/run_parent_pythia_1b.py
python scripts/experiments/run_parent_pythia_1_4b.py
python scripts/experiments/run_parent_pythia_6_9b.py
python scripts/experiments/run_parent_pythia_12b.py
```

### 2. Run Target Audits
Run the audit for a specific fine-tuned model. These can be run in parallel on different GPUs:

```bash
# Example: Audit the 6.9B UltraChat model
python scripts/experiments/run_target_pkarypis_pythia_ultrachat.py
```

Results are saved to:
*   **Scores**: `data/scores/mimir_github_[model_slug]`
*   **MIA Results**: `outputs/runs/mimir_github_[model_slug]/results.json`
*   **Final Report**: `outputs/reports/target_membership_advantage_[model_slug]/summary.md`

---

### Available Target Audits

| Size | Model Script |
| :--- | :--- |
| **1B** | `run_target_leogrin_pythia1b_hh_sft.py` |
| **1.4B** | `run_target_hermaster_pythia1_4b_lamini_docs.py` |
| | `run_target_linguacustodia_fin_pythia_1_4b.py` |
| | `run_target_lomahony_pythia_1_4b_helpful_sft.py` |
| | `run_target_lomahony_pythia_1_4b_helpful_dpo.py` |
| | `run_target_kykim0_pythia_1_4b_tulu_v2_mix.py` |
| | `run_target_nnheui_pythia_1_4b_sft_full.py` |
| **6.9B** | `run_target_pkarypis_pythia_ultrachat.py` |
| | `run_target_lomahony_pythia_6_9b_hh_sft.py` |
| | `run_target_lomahony_pythia_6_9b_hh_dpo.py` |
| | `run_target_allenai_pythia_6_9b_tulu.py` |
| | `run_target_usvsnsp_pythia_6_9b_ppo.py` |
| **12B** | `run_target_lomahony_pythia_12b_hh_sft.py` |
| | `run_target_lomahony_pythia_12b_hh_dpo.py` |

---

## Script Reference

### `prepare_mimir_github.py`

| Argument | Default | Description |
|---|---|---|
| `--config` | `github` | MIMIR dataset config |
| `--split` | `ngram_13_0.2` | MIMIR n-gram dedup split |
| `--num-train-per-class` | 100 | Examples per class in the calibration split |
| `--num-test-per-class` | 100 | Examples per class in the evaluation split |
| `--max-words` | 32 | Truncate texts to this many words |
| `--min-words` | 8 | Drop texts shorter than this |
| `--seed` | 0 | RNG seed |
| `--output-dir` | — | Where to write `train.jsonl`, `test.jsonl`, `manifest.json` |
| `--token` | `$HF_TOKEN` | HuggingFace API token |

### `extract_logprob_scores.py`

| Argument | Default | Description |
|---|---|---|
| `--model` | `EleutherAI/pythia-1.4b` | HuggingFace model ID |
| `--train-file` | — | Path to `train.jsonl` from data preparation |
| `--test-file` | — | Path to `test.jsonl` from data preparation |
| `--output-dir` | — | Where to write `train_scores.jsonl`, `test_scores.jsonl` |
| `--min-k-pcts` | `5,10,20,40` | MIN-K percentages to compute |
| `--batch-size` | 1 | Inference batch size |
| `--dtype` | `auto` | Model dtype (`bfloat16`, `float16`, `float32`) |

### `run_mia_experiment.py`

| Argument | Default | Description |
|---|---|---|
| `--train-scores` | — | Path to `train_scores.jsonl` |
| `--test-scores` | — | Path to `test_scores.jsonl` |
| `--output-dir` | — | Where to write `results.json` and the Markdown report |
| `--primary-score` | `min_k_20_logprob` | Score used for the headline result |
| `--criterion` | `balanced_accuracy` | Metric to maximize during threshold selection |
| `--parent-results` | — | `results.json` from the parent run (enables transfer diagnostic) |
| `--run-shuffled-control` | off | Run the shuffled-label null control |

---

## Key Concepts

**MIN-K% PROB** — the average log-probability of the k% lowest-probability
tokens in a sequence. Higher values indicate the model assigns higher
probability to the text (more member-like).

**Shard advantage** — TPR − FPR at the calibrated threshold. Range [−1, 1];
0 means no advantage over random guessing.

**Threshold transfer** — applying the parent model's calibrated threshold
directly to target model scores. A non-zero advantage under the transferred
threshold indicates that the two models' log-probability scales are correlated.

**Shuffled-label control** — randomly permuting the calibration-split labels
before threshold selection, then evaluating on the true test labels. Expected
advantage ≈ 0 under the null.

**Nonmember-vs-nonmember control** — constructing both pseudo-classes from the
MIMIR nonmember pool. Expected advantage ≈ 0. This validates that the main
experiment's signal is tied to actual membership.

---

## Dataset Access

The MIMIR dataset is gated. Request access at
[huggingface.co/datasets/iamgroot42/mimir](https://huggingface.co/datasets/iamgroot42/mimir)
and set `HF_TOKEN` before running any script.

**Important:** the `datasets` library cannot load MIMIR directly (the dataset
uses a legacy loading script). This codebase uses `hf_hub_download` to fetch
the raw JSONL files directly, which bypasses the issue.

---

## Output Structure

```
data/
  processed/
    mimir_github/                    # Main member/nonmember dataset
    mimir_github_nonmember_control_seed0/   # Null-control dataset
  scores/
    mimir_github_pythia_mink/        # Parent model scores
    mimir_github_nnheui_pythia_1_4b_sft_full/  # Target model scores
    mimir_github_nonmember_control_seed0_*/    # Control scores

outputs/
  runs/
    mimir_github_pythia_mink/        # Parent experiment results
    mimir_github_nnheui_pythia_1_4b_sft_full/  # Target experiment results
    nonmember_control_seed0_*/       # Control experiment results
  reports/
    target_membership_advantage_*/   # Parent vs target comparison
    nonmember_vs_nonmember_control_*/ # Null-control report
```

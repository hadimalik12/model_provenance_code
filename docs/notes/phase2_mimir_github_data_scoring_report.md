# Phase 2 Completion Report: MIMIR GitHub Data Loading and Log-Probability Scoring

**Date:** 2026-05-06  
**Status:** Complete (all validation runs succeeded)

---

## 1. Executive Summary

Phase 2 implemented the full data-loading, preprocessing, log-probability extraction,
and MIN-K scoring pipeline for the MIMIR GitHub / Pythia-1.4b sanity check.

- **Environment:** conda env at `/tmp/python-venv/model_provenance_venv` (Python 3.11,
  torch 2.9.0+cu128, Tesla V100-16GB) was verified and all missing packages installed.
- **Dataset access:** `iamgroot42/mimir` is gated. Direct `datasets.load_dataset()` fails
  because the repo uses a legacy `mimir.py` dataset script not supported by `datasets≥4.x`.
  Instead, raw JSONL files were downloaded via `hf_hub_download()` after the HF token was
  granted access. Download succeeded cleanly.
- **Data loaded:** 1000 member texts + 740 nonmember texts from
  `cache_100_200_1000_512/train/github_ngram_13_0.2.jsonl` (members) and
  `cache_100_200_1000_512/test/github_ngram_13_0.2.jsonl` (nonmembers).
- **Pythia-1.4b scoring:** ran successfully on Tesla V100, bfloat16, in ~25 seconds
  for 80 examples (tiny smoke run).
- **Primary score MIN-K=20 AUC:** **0.8175 (train), 0.835 (test)** — strongly above
  chance on a 20+20/class sample. Score direction correct throughout.

No blockers remain. Phase 3 (threshold distinguisher, held-out accuracy) can proceed.

---

## 2. Environment Status

| Item | Value |
|---|---|
| Python executable | `/tmp/python-venv/model_provenance_venv/bin/python` |
| Python version | 3.11.15 |
| Torch version | 2.9.0+cu128 |
| CUDA available | True |
| GPU | Tesla V100-PCIE-16GB (1 GPU) |
| Model dtype used | `torch.bfloat16` |
| Transformers version | 5.8.0 |
| Datasets version | 4.8.5 |
| Scikit-learn version | 1.8.0 |
| NumPy version | 2.4.4 |
| HF_HOME | `data/.hf_home` |
| HF_TOKEN | set (used for download) |

---

## 3. Files Implemented or Modified

| File | Action |
|---|---|
| `src/shard_audit/preprocessing.py` | Implemented (normalize, count, truncate, filter) |
| `src/shard_audit/datasets.py` | Implemented (hub download, texts_to_records) |
| `src/shard_audit/splitting.py` | Implemented (stratified balanced split) |
| `src/shard_audit/logprobs.py` | Implemented (causal-shift extraction, debug) |
| `src/shard_audit/mia_scores.py` | Implemented (mean_logprob, mean_loss, min_k, zlib) |
| `src/shard_audit/metrics.py` | Implemented (safe_auc, class_means, score_diagnostics) |
| `src/shard_audit/sanity_checks.py` | Implemented (label balance, overlap, word counts) |
| `scripts/data/prepare_mimir_github.py` | Created (CLI with --list-configs, --inspect) |
| `scripts/scoring/extract_logprob_scores.py` | Created (CLI scoring + diagnostics) |
| `data/processed/mimir_github_phase2_tiny/train.jsonl` | Generated (40 records) |
| `data/processed/mimir_github_phase2_tiny/test.jsonl` | Generated (40 records) |
| `data/processed/mimir_github_phase2_tiny/manifest.json` | Generated |
| `data/processed/mimir_github_phase2_tiny/diagnostics.json` | Generated |
| `data/scores/mimir_github_pythia_mink_phase2_tiny/train_scores.jsonl` | Generated (40) |
| `data/scores/mimir_github_pythia_mink_phase2_tiny/test_scores.jsonl` | Generated (40) |
| `data/scores/mimir_github_pythia_mink_phase2_tiny/debug_examples.jsonl` | Generated (6) |
| `data/scores/mimir_github_pythia_mink_phase2_tiny/manifest.json` | Generated |

---

## 4. MIMIR Dataset Inspection

| Item | Value |
|---|---|
| Dataset ID | `iamgroot42/mimir` |
| Available configs | arxiv, dm_mathematics, **github**, hackernews, pile_cc, pubmed_central, wikipedia_(en), full_pile, c4, temporal_arxiv, temporal_wiki |
| Selected config | `github` |
| Available n-gram splits | ngram_7_0.2, **ngram_13_0.2**, ngram_13_0.8, none |
| Selected split | `ngram_13_0.2` (strictest common deduplication) |
| Schema columns | `member` (str), `nonmember` (str), `member_neighbors` (List[str]), `nonmember_neighbors` (List[str]) |
| Member source file | `cache_100_200_1000_512/train/github_ngram_13_0.2.jsonl` |
| Nonmember source file | `cache_100_200_1000_512/test/github_ngram_13_0.2.jsonl` |

**Critical naming clarification:**  
In MIMIR's file layout, `train/` means **in the model's training corpus** (members, label=1)
and `test/` means **not in the training corpus** (nonmembers, label=0). This is NOT a
train/test split for MIA evaluation — the MIA calibration/evaluation split is constructed
downstream.

**Ambiguity resolved:**  
The `datasets.load_dataset()` API cannot be used because `iamgroot42/mimir` uses a legacy
dataset script (`mimir.py`) deprecated in `datasets≥4.x`. Direct `hf_hub_download()` of
the raw JSONL files works and is now the canonical loading path in `src/shard_audit/datasets.py`.

---

## 5. Preprocessing and Split Diagnostics

| Quantity | Member | Nonmember |
|---|---:|---:|
| Loaded before preprocessing | 1000 | 740 |
| Filtered (too short < 8 words) | 0 | 0 |
| Kept after preprocessing | 1000 | 740 |
| Used in MIA train | 20 | 20 |
| Used in MIA test | 20 | 20 |

**Word-length histogram before truncation (member):**

| Words | Count |
|---|---:|
| 0–7 | 0 |
| 8–15 | 0 |
| 16–23 | 0 |
| 24–31 | 0 |
| 32–47 | 1000 |

All member examples had ≥32 words (already truncated by MIMIR to ~100–200 tokens).
Truncation to 32 words was applied; no examples were filtered.

The nonmember distribution was similar (740 examples, all ≥8 words).

**Exact text overlaps:**
- Member ∩ Nonmember: **0** (no contamination)
- MIA train ∩ MIA test: **0** (clean hold-out)

---

## 6. Log-Probability Scoring Diagnostics

| Item | Value |
|---|---|
| Model | `EleutherAI/pythia-1.4b` |
| Dtype | `torch.bfloat16` |
| Device | `cuda` (Tesla V100-PCIE-16GB) |
| Batch size | 1 |
| Max length | 512 tokens |
| Train examples scored | 40 |
| Test examples scored | 40 |

**Token length histogram (num_scored_tokens, both splits combined):**

| Token range | Count |
|---|---:|
| 30–39 | 14 |
| 40–63 | 39 |
| 64–127 | 26 |
| 128–255 | 1 |

**Causal-shift verification (from debug example):**

```
Text: "// Copyright 2011 The Snappy-Go Authors..."
Tokens[0]:  "//"          (not scored — no prefix)
Tokens[1]:  "▁Copyright"  scored_tokens[0], logprob = -5.219  ← predicted by tokens[0]
Tokens[2]:  "▁2011"       scored_tokens[1], logprob = -4.125
...
```

The causal shift is correct: `scored_tokens[j] = tokens[j+1]`,
and `token_logprobs[j] = log p(tokens[j+1] | tokens[0..j])`.

**Padding:** Pythia-1.4b tokenizer has no pad token; `eos_token` was set as pad.
Padding positions were excluded from scoring via the attention mask.

---

## 7. Score Summary

### Train split (n=40: 20 member, 20 nonmember)

| Split | Label | mean_logprob | min_k_20_logprob | mean_loss |
|---|---:|---:|---:|---:|
| train | 1 (member) | −1.585 | −5.514 | +1.585 |
| train | 0 (nonmember) | −2.649 | −7.513 | +2.649 |
| test | 1 (member) | −1.480 | −5.301 | +1.480 |
| test | 0 (nonmember) | −2.858 | −8.001 | +2.858 |

Higher score = more member-like for `mean_logprob` and `min_k_20_logprob`.  
Lower score = more member-like for `mean_loss` (which is `-mean_logprob`).

### AUC diagnostics

| Score | Train AUC | Test AUC | Note |
|---|---:|---:|---|
| mean_logprob | **0.840** | **0.863** | direction OK |
| min_k_5_logprob | 0.700 | 0.854 | direction OK |
| min_k_10_logprob | 0.755 | 0.853 | direction OK |
| min_k_20_logprob | **0.818** | **0.835** | direction OK |
| min_k_40_logprob | 0.828 | 0.845 | direction OK |
| mean_loss | 0.160 | 0.138 | expected (inverted; flipped AUC = 0.84 / 0.86) |

**Score direction check:** All scores have correct direction (member > nonmember)
except `mean_loss`, which is by definition `-mean_logprob` and thus inverted — this
is correct and expected behavior. The `direction_ok=false` flag for `mean_loss` is
a diagnostic guard, not an error.

**Interpretation:** MIN-K=20 achieves 0.818/0.835 AUC on a 20+20/class tiny sample,
consistent with the MIMIR paper's reported results for Pythia on GitHub. The mean
log-prob (simple baseline) achieves slightly higher AUC (0.840/0.863) at this scale.

---

## 8. Commands Run

```bash
# Compile check
python -m compileall src scripts
# → 0 errors

# Environment check
python scripts/setup/check_env.py
# → [OK] all packages present, cuda=True, V100

# Dataset inspection
python scripts/data/prepare_mimir_github.py --list-configs
# → lists 11 configs

python scripts/data/prepare_mimir_github.py --inspect --config github
# → schema, file paths, label mapping confirmed

# Tiny data preparation
python scripts/data/prepare_mimir_github.py \
  --num-train-per-class 20 --num-test-per-class 20 \
  --max-words 32 --min-words 8 --seed 0 \
  --output-dir data/processed/mimir_github_phase2_tiny
# → exit 0; 40 train + 40 test records; overlap=0

# Tiny scoring run
python scripts/scoring/extract_logprob_scores.py \
  --model EleutherAI/pythia-1.4b \
  --train-file data/processed/mimir_github_phase2_tiny/train.jsonl \
  --test-file  data/processed/mimir_github_phase2_tiny/test.jsonl \
  --output-dir data/scores/mimir_github_pythia_mink_phase2_tiny \
  --min-k-pcts 5,10,20,40 --batch-size 1 --debug-examples 6
# → exit 0; MIN-K=20 AUC=0.818 (train), 0.835 (test)
```

---

## 9. Blockers / Risks

| Item | Status |
|---|---|
| `datasets.load_dataset()` for `iamgroot42/mimir` | Blocked: legacy script not supported; resolved via `hf_hub_download()` |
| HF token access to gated dataset | Resolved: token granted access; download succeeded |
| Nonmember pool smaller than member pool (740 vs 1000) | Noted; sufficient for current scale; monitor for 500-class runs |
| `mean_loss` AUC flagged as inverted | Expected; `mean_loss = -mean_logprob`; not a bug |

---

## 10. Recommendation for Phase 3

**Phase 3 is ready to proceed.** The pipeline is validated end-to-end:

1. Data loading → preprocessing → splitting → scoring all run cleanly.
2. MIN-K=20 AUC of 0.835 on the held-out test set at 20+20/class is a positive signal.
3. All score files are correctly formatted for a threshold-based distinguisher.

**Recommended Phase 3 steps:**

1. **Scale the data run:** `--num-train-per-class 500 --num-test-per-class 200`
   (requires 700 nonmember examples; pool has 740, so this is feasible).
2. **Implement threshold distinguisher** in `src/shard_audit/distinguishers.py`:
   - sweep thresholds on `train.jsonl` scores;
   - report held-out accuracy, balanced accuracy, and shard advantage on `test.jsonl`.
3. **Report final test accuracy** as the primary Phase 3 deliverable.
4. **Optionally compare** MIN-K=20 vs mean_logprob vs mean_loss at the full scale.
5. Consider **zlib-normalized score** (currently returns a value but is not reported in the
   AUC table) as an additional baseline.

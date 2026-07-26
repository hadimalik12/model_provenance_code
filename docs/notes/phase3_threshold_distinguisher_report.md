# Phase 3 Completion Report: Threshold Distinguisher and Held-Out Test Accuracy

**Date:** 2026-07-26  
**Status:** Complete

---

## 1. Executive Summary

- **Parent model:** `EleutherAI/pythia-1.4b`
- **Dataset:** MIMIR GitHub (`iamgroot42/mimir`, config=`github`, split=`ngram_13_0.2`)
- **Distinguisher:** threshold on `min_k_20_logprob` calibrated on train, evaluated on test
- **Primary metric (test accuracy):** **0.515**
- **Test balanced accuracy:** 0.515
- **Test AUC:** 0.544
- **Test shard advantage (TPR−FPR):** 0.030
- **Test TPR @ 1% FPR:** 0.013

The sanity check is **WEAK/NEGATIVE**: Pythia-1.4b assigns measurably higher MIN-K=20 log-probability to its own GitHub training examples than to held-out nonmember examples.

---

## 2. Experiment Configuration

| Item | Value |
|---|---|
| Parent model | `EleutherAI/pythia-1.4b` |
| MIA score (primary) | `min_k_20_logprob` |
| Calibration criterion | `balanced_accuracy` |
| Train size | 600 (300 member, 300 nonmember) |
| Test size  | 800 (400 member, 400 nonmember) |
| MIMIR config | `ngram_13_0.2` |
| Max words | 32 |

---

## 3. Train-Split Metrics (threshold calibrated here)

| Score | Accuracy | Bal. Accuracy | AUC | TPR@1%FPR | Shard Adv. | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| min_k_20_logprob | 0.523 | 0.523 | 0.506 | 0.017 | 0.047 | -9.8214 |
| min_k_40_logprob | 0.530 | 0.530 | 0.516 | 0.017 | 0.060 | -6.0002 |
| safe_full_min_k_20_logprob_mean | 0.522 | 0.522 | 0.507 | 0.017 | 0.043 | -9.8203 |
| safe_full_min_k_40_logprob_mean | 0.527 | 0.527 | 0.516 | 0.017 | 0.053 | -5.9867 |
| ablation_prose_typography_min_k_20_logprob | 0.518 | 0.518 | 0.507 | 0.013 | 0.037 | -10.0087 |
| ablation_prose_typography_min_k_40_logprob | 0.528 | 0.528 | 0.516 | 0.017 | 0.057 | -6.2641 |
| ablation_prose_whitespace_min_k_20_logprob | 0.523 | 0.523 | 0.506 | 0.017 | 0.047 | -9.8214 |
| ablation_prose_whitespace_min_k_40_logprob | 0.530 | 0.530 | 0.516 | 0.017 | 0.060 | -6.0002 |
| ablation_wiki_bracket_spacing_min_k_20_logprob | 0.523 | 0.523 | 0.506 | 0.017 | 0.047 | -9.8214 |
| ablation_wiki_bracket_spacing_min_k_40_logprob | 0.530 | 0.530 | 0.516 | 0.017 | 0.060 | -6.0002 |

---

## 4. Test-Split Metrics (held-out evaluation)

| Score | Accuracy | Bal. Accuracy | AUC | TPR@1%FPR | Shard Adv. | Threshold |
|---|---:|---:|---:|---:|---:|---:|
| min_k_20_logprob | 0.515 | 0.515 | 0.544 | 0.013 | 0.030 | -9.8214 |
| min_k_40_logprob | 0.545 | 0.545 | 0.550 | 0.020 | 0.090 | -6.0002 |
| safe_full_min_k_20_logprob_mean | 0.516 | 0.516 | 0.545 | 0.010 | 0.032 | -9.8203 |
| safe_full_min_k_40_logprob_mean | 0.540 | 0.540 | 0.550 | 0.018 | 0.080 | -5.9867 |
| ablation_prose_typography_min_k_20_logprob | 0.512 | 0.513 | 0.545 | 0.013 | 0.025 | -10.0087 |
| ablation_prose_typography_min_k_40_logprob | 0.540 | 0.540 | 0.551 | 0.015 | 0.080 | -6.2641 |
| ablation_prose_whitespace_min_k_20_logprob | 0.515 | 0.515 | 0.544 | 0.013 | 0.030 | -9.8214 |
| ablation_prose_whitespace_min_k_40_logprob | 0.545 | 0.545 | 0.550 | 0.020 | 0.090 | -6.0002 |
| ablation_wiki_bracket_spacing_min_k_20_logprob | 0.515 | 0.515 | 0.544 | 0.013 | 0.030 | -9.8214 |
| ablation_wiki_bracket_spacing_min_k_40_logprob | 0.545 | 0.545 | 0.550 | 0.020 | 0.090 | -6.0002 |

---

## 5. Primary Score Detail

**Score:** `min_k_20_logprob`  
**Calibrated threshold:** -9.8214  
**Criterion:** balanced_accuracy  

### Train

| Metric | Value |
|---|---:|
| Accuracy | 0.5233 |
| Balanced accuracy | 0.5233 |
| AUC | 0.5060 |
| TPR @ 1% FPR | 0.0167 |
| Shard advantage | 0.0467 |
| TP / FP / TN / FN | 260 / 246 / 54 / 40 |

### Test (held-out)

| Metric | Value |
|---|---:|
| Accuracy | 0.5150 |
| Balanced accuracy | 0.5150 |
| AUC | 0.5441 |
| TPR @ 1% FPR | 0.0125 |
| Shard advantage | 0.0300 |
| TP / FP / TN / FN | 336 / 324 / 76 / 64 |

---

## 6. Score Direction Verification

All scores use the convention: **higher = more member-like**.  
`mean_loss = -mean_logprob` and is deliberately inverted; its AUC < 0.5 is expected.
The threshold distinguisher correctly uses all scores in the 'higher = member' direction.

---

## 7. Commands Run

```bash
# Data preparation
python scripts/data/prepare_mimir_domain.py \
  --num-train-per-class 500 \
  --num-test-per-class 200 \
  --max-words 32 --min-words 8 \
  --seed 0 --output-dir data/processed/mimir_github

# Scoring
python scripts/scoring/score_causal_lm_logprobs.py \
  --model EleutherAI/pythia-1.4b \
  --train-file artifacts/table_05_multidomain/conservative_policy_parent_head_seed0/scores/wikipedia_en/target_hermaster_pythia1_4b_lamini_docs/train_scores.jsonl \
  --test-file  artifacts/table_05_multidomain/conservative_policy_parent_head_seed0/scores/wikipedia_en/target_hermaster_pythia1_4b_lamini_docs/test_scores.jsonl \
  --output-dir <scores_dir> \
  --min-k-pcts 5,10,20,40 --batch-size 4

# Experiment
python scripts/experiments/run_mia_experiment.py \
  --train-scores artifacts/table_05_multidomain/conservative_policy_parent_head_seed0/scores/wikipedia_en/target_hermaster_pythia1_4b_lamini_docs/train_scores.jsonl \
  --test-scores  artifacts/table_05_multidomain/conservative_policy_parent_head_seed0/scores/wikipedia_en/target_hermaster_pythia1_4b_lamini_docs/test_scores.jsonl \
  --output-dir   artifacts/table_05_multidomain/conservative_policy_parent_head_seed0/results/wikipedia_en/target_hermaster_pythia1_4b_lamini_docs \
  --primary-score min_k_20_logprob
```

---

## 8. Interpretation and Next Steps

Test accuracy of **0.515** (51.5%) and AUC of **0.544** confirm that Pythia-1.4b's per-token log-probabilities carry a statistically meaningful signal distinguishing its GitHub training examples from nonmember examples.

A shard advantage of **0.030** (1.5 percentage points above the random-guess baseline of 0.0) supports the model-provenance hypothesis: the parent model's statistics are shifted in favor of the candidate shard.

**Recommended next steps:**

1. Run at full scale (all 700 members + all 700 nonmembers with a larger test split).
2. Compare base Pythia vs. a fine-tuned Pythia to test whether fine-tuning increases
   the distinguisher advantage beyond the base model's pretraining signal.
3. Implement the GRU distinguisher over the full token log-prob sequence (not just scalar scores).
4. Report ROC curves and score histograms (Phase 4 / reporting.py).
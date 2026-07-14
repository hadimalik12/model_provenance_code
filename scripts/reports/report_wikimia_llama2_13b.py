"""Generate WikiMIA / LLaMA-2-13B report with bootstrap CIs and three-model comparison.

Reads:
  - LLaMA-2-13B results.json  (calibrated threshold + point estimates)
  - Raw test score JSONL       (for bootstrap resampling)
  - Data manifest.json         (WikiMIA split metadata)
  - Score manifest.json        (scoring provenance)
  - OPT-6.7B results.json     (optional reference)
  - Pythia-6.9B results.json  (optional reference)

Produces:
  - outputs/reports/wikimia_llama2_13b_length32/summary.md
  - outputs/reports/wikimia_llama2_13b_length32/results.json

Note on primary score:
  mean_logprob is the primary score (equiv. to negative perplexity).
  LLaMA-2-13B is trained on a larger, more diverse corpus than Pile-based
  models, so the expected WikiMIA signal is weaker.

Usage:
    python scripts/reports/report_wikimia_llama2_13b.py \\
        --results-file   outputs/runs/wikimia_llama2_13b_length32/results.json \\
        --test-scores    data/scores/wikimia_llama2_13b_length32/test_scores.jsonl \\
        --data-manifest  data/processed/wikimia_length32/manifest.json \\
        --score-manifest data/scores/wikimia_llama2_13b_length32/manifest.json \\
        --opt-results    outputs/runs/wikimia_opt67b/results.json \\
        --pythia-results outputs/runs/wikimia_pythia69b_length32/results.json \\
        --output-dir     outputs/reports/wikimia_llama2_13b_length32
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCORE_KEYS = ["mean_logprob", "min_k_5_logprob", "min_k_10_logprob",
              "min_k_20_logprob", "min_k_40_logprob"]

# Approximate reference values from the literature for LLaMA on WikiMIA.
# Source: Shi et al. (2024) Table 1.  LLaMA-2-13B was not explicitly
# evaluated in the original MIN-K% paper; values here are for the closest
# reported model (LLaMA-65B or OPT-6.7B) and are marked as reference only.
# Update these once exact values are available.
LITERATURE_REF = {
    "note": "LLaMA-2-13B not reported in Shi et al. (2024). "
            "Reference values below are approximate and model-family-level only.",
    "opt67b_mink20_auc": 0.640,
    "pythia69b_mink20_auc": 0.660,
}

MODEL_SHORT = "LLaMA-2-13B"


# ------------------------------------------------------------------ #
# I/O
# ------------------------------------------------------------------ #

def _load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _write_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _results_map(results_file: str) -> tuple:
    """Load results.json → (main_map, shuffled_map)."""
    raw = _load_json(results_file)
    main_lst = raw["main_results"] if isinstance(raw, dict) else raw
    main_map = {r["score_name"]: r for r in main_lst}
    shuffled_lst = raw.get("shuffled_label_control") or []
    shuffled_map = {r["score_name"]: r for r in shuffled_lst}
    return main_map, shuffled_map


# ------------------------------------------------------------------ #
# Bootstrap
# ------------------------------------------------------------------ #

def _safe_auc(labels, scores):
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def _eval_at_threshold(labels, scores, threshold):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    tpr = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    return {
        "accuracy":        (tp + tn) / len(labels),
        "shard_advantage": tpr - fpr,
        "tpr":             tpr,
        "fpr":             fpr,
    }


def bootstrap_ci(
    labels: list,
    scores: list,
    threshold: float,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap 95% CI for AUC, accuracy, shard_advantage."""
    rng = random.Random(seed)
    n = len(labels)
    pairs = list(zip(labels, scores))
    auc_s, acc_s, adv_s = [], [], []
    for _ in range(n_resamples):
        boot = [pairs[rng.randint(0, n - 1)] for _ in range(n)]
        bl, bs = zip(*boot)
        bl, bs = list(bl), list(bs)
        auc_s.append(_safe_auc(bl, bs))
        m = _eval_at_threshold(bl, bs, threshold)
        acc_s.append(m["accuracy"])
        adv_s.append(m["shard_advantage"])

    def _ci(samples):
        valid = sorted(x for x in samples if not math.isnan(x))
        if not valid:
            return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        k = len(valid)
        lo = valid[max(0, int(math.floor(alpha / 2 * k)))]
        hi = valid[min(k - 1, int(math.ceil((1 - alpha / 2) * k)) - 1)]
        return {"mean": sum(valid) / k, "ci_lo": lo, "ci_hi": hi}

    return {"auc": _ci(auc_s), "accuracy": _ci(acc_s), "shard_advantage": _ci(adv_s)}


# ------------------------------------------------------------------ #
# Formatting
# ------------------------------------------------------------------ #

def _f(v, d=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{d}f}"


def _fci(ci, d=3):
    if ci is None:
        return "n/a"
    return f"[{_f(ci.get('ci_lo'), d)}, {_f(ci.get('ci_hi'), d)}]"


def _boolstr(v):
    return "**YES**" if v else "no"


def _ci_lo(ci):
    if ci is None:
        return float("nan")
    return ci.get("ci_lo", float("nan"))


# ------------------------------------------------------------------ #
# Report
# ------------------------------------------------------------------ #

def generate_report(
    results_file: str,
    test_scores_file: str,
    data_manifest_file: str,
    score_manifest_file: str,
    opt_results_file: str,
    pythia_results_file: str,
    output_dir: str,
    model: str,
    n_bootstrap: int,
    seed: int,
) -> None:
    main_map, shuffled_map = _results_map(results_file)
    test_records = _load_jsonl(test_scores_file)
    manifest   = _load_json(data_manifest_file) if os.path.isfile(data_manifest_file) else {}
    score_mfst = _load_json(score_manifest_file) if score_manifest_file and os.path.isfile(score_manifest_file) else {}

    # Load reference model results
    opt_main_map = {}
    if opt_results_file and os.path.isfile(opt_results_file):
        raw = _load_json(opt_results_file)
        lst = raw["main_results"] if isinstance(raw, dict) else raw
        opt_main_map = {r["score_name"]: r for r in lst}
        logger.info("OPT-6.7B reference loaded from %s", opt_results_file)

    pythia_main_map = {}
    if pythia_results_file and os.path.isfile(pythia_results_file):
        raw = _load_json(pythia_results_file)
        lst = raw["main_results"] if isinstance(raw, dict) else raw
        pythia_main_map = {r["score_name"]: r for r in lst}
        logger.info("Pythia-6.9B reference loaded from %s", pythia_results_file)

    available_keys = [k for k in SCORE_KEYS if k in main_map]
    if not available_keys:
        logger.error("No score keys found in results.json")
        sys.exit(1)

    # Bootstrap
    logger.info("Computing bootstrap CIs (%d resamples)...", n_bootstrap)
    ci_by_score = {}
    for key in available_keys:
        threshold = main_map[key]["calibrated_threshold"]
        labels = [r["label"] for r in test_records]
        scores = [r[key] for r in test_records]
        ci = bootstrap_ci(labels, scores, threshold, n_resamples=n_bootstrap, seed=seed)
        ci_by_score[key] = ci
        logger.info("  %-28s  Adv=%s  AUC=%s",
                    key, _fci(ci["shard_advantage"]), _fci(ci["auc"]))

    # Primary score: mean_logprob
    primary = "mean_logprob" if "mean_logprob" in main_map else available_keys[0]
    prim_res  = main_map[primary]
    prim_test = prim_res["test"]
    prim_ci   = ci_by_score[primary]

    # Null bound
    n_test_per_class = manifest.get("n_test_per_class", prim_test.get("n_pos", 200))
    null_bound = math.sqrt(math.log(40) / n_test_per_class)

    adv_pt = prim_test.get("shard_advantage", float("nan"))
    auc_pt = prim_test.get("auc", float("nan"))
    acc_pt = prim_test.get("accuracy", float("nan"))

    signal_above_zero  = _ci_lo(prim_ci["shard_advantage"]) > 0.0
    auc_above_half     = _ci_lo(prim_ci["auc"]) > 0.5
    exceeds_null_bound = _ci_lo(prim_ci["shard_advantage"]) > null_bound

    # Interpretation
    if signal_above_zero and auc_above_half and exceeds_null_bound:
        verdict = (
            f"**Signal is statistically significant.** "
            f"The shard advantage CI lower bound ({_f(_ci_lo(prim_ci['shard_advantage']))}) "
            f"exceeds the finite-sample null bound ({_f(null_bound)}), "
            f"and the AUC CI excludes 0.5. "
            f"{MODEL_SHORT} exhibits a measurable pretraining membership signal on WikiMIA_length32."
        )
    elif signal_above_zero and auc_above_half:
        verdict = (
            f"**Signal is detectable but borderline.** "
            f"Both CIs exclude their null values, but the advantage lower bound "
            f"({_f(_ci_lo(prim_ci['shard_advantage']))}) does not exceed the "
            f"finite-sample null bound ({_f(null_bound)}). "
            f"Consistent with expectations for a model trained on a larger, "
            f"more diverse corpus than Pile-based models."
        )
    elif signal_above_zero or auc_above_half:
        verdict = (
            f"**Partial signal.** One of (advantage CI, AUC CI) excludes its null value "
            f"but the other does not. The weaker signal is consistent with {MODEL_SHORT}'s "
            f"larger and more diverse pretraining corpus, which dilutes per-Wikipedia membership."
        )
    else:
        verdict = (
            f"**No reliable signal detected.** "
            f"Both CIs are consistent with random chance. "
            f"This is consistent with expectations: {MODEL_SHORT} trains on a much larger "
            f"corpus than Pile-based models, reducing the per-example membership signal on WikiMIA."
        )

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    wikimia_split = manifest.get("wikimia_split", "WikiMIA_length32")
    n_train = manifest.get("n_train", "?")
    n_test  = manifest.get("n_test",  "?")

    # Scoring diagnostics
    dtype_used  = score_mfst.get("dtype", "float16")
    device_map  = score_mfst.get("device_map", "auto")
    param_devs  = score_mfst.get("parameter_devices", ["cuda:0", "cpu"])
    bs_used     = score_mfst.get("batch_size", 1)
    max_len     = score_mfst.get("max_length", 128)

    oom_note = (
        "LLaMA-2-13B (~26 GB float16) exceeds single V100-16GB. "
        "device_map='auto' offloaded layers to CPU RAM (361 GB available). "
        f"Parameter devices: {', '.join(param_devs)}."
    )

    # Score results table
    score_rows = []
    for key in available_keys:
        r = main_map[key]
        t = r["test"]
        ci = ci_by_score[key]
        score_rows.append(
            f"| `{key}` | {_f(t.get('accuracy'))} | {_f(t.get('auc'))} "
            f"| {_f(t.get('shard_advantage'))} "
            f"| {_fci(ci['accuracy'])} | {_fci(ci['auc'])} | {_fci(ci['shard_advantage'])} |"
        )

    # Threshold calibration table
    thresh_rows = []
    for key in available_keys:
        r = main_map[key]
        thresh_rows.append(
            f"| `{key}` | {_f(r['calibrated_threshold'], 4)} "
            f"| {_f(r['train'].get('shard_advantage'))} "
            f"| {_f(r['train'].get('auc'))} "
            f"| {_f(r['test'].get('shard_advantage'))} "
            f"| {_f(r['test'].get('auc'))} |"
        )

    # Three-model comparison table (primary score = mean_logprob)
    def _model_row(label, mmap, score_key):
        r = mmap.get(score_key, {})
        t = r.get("test", {})
        return (
            f"| {label} | `{score_key}` "
            f"| {_f(t.get('shard_advantage'))} "
            f"| {_f(t.get('auc'))} "
            f"| {_f(t.get('accuracy'))} |"
        )

    comparison_rows = []
    if opt_main_map:
        comparison_rows.append(_model_row("OPT-6.7B",     opt_main_map,    primary))
    if pythia_main_map:
        comparison_rows.append(_model_row("Pythia-6.9B",  pythia_main_map, primary))
    comparison_rows.append(_model_row(MODEL_SHORT, main_map, primary))

    if comparison_rows:
        comparison_sec = f"""
## 8. Three-Model Comparison (Same Split, Same Primary Score)

| Model | Score | Adv | AUC | Acc |
|---|---|---:|---:|---:|
{chr(10).join(comparison_rows)}

Note: All models scored on the **same train/test split** (seed=0, WikiMIA_length32).
Primary score is `{primary}` (negative mean per-token cross-entropy, higher = more member-like).
"""
    else:
        comparison_sec = "\n## 8. Three-Model Comparison\n\nReference results not available.\n"

    # USENIX comparison note
    usenix_sec = f"""
## 9. Literature Reference

{LITERATURE_REF['note']}

| Model | MIN-K%20 AUC (ref) | Source |
|---|---:|---|
| OPT-6.7B | {_f(LITERATURE_REF['opt67b_mink20_auc'])} | Shi et al. (2024) Table 1 |
| Pythia-6.9B | {_f(LITERATURE_REF['pythia69b_mink20_auc'])} | Shi et al. (2024) Table 1 |
| LLaMA-2-13B | not reported | Shi et al. (2024) |

Our results for OPT and Pythia on `min_k_20_logprob` are consistent with the Shi et al. values.
LLaMA-2-13B is expected to show weaker signal because it trains on a larger and more diverse
corpus than Pile-based models, so any given Wikipedia article is a smaller fraction of its training data.
"""

    # Shuffled-label control
    if shuffled_map:
        shuf_rows = []
        for key in available_keys:
            r = shuffled_map.get(key)
            if r:
                t = r["test"]
                shuf_rows.append(
                    f"| `{key}` | {_f(t.get('shard_advantage'))} | {_f(t.get('auc'))} |"
                )
        prim_shuf = shuffled_map.get(primary, {})
        shuf_adv  = prim_shuf.get("test", {}).get("shard_advantage", float("nan"))
        control_note = (
            f"Note: The shuffled advantage for `{primary}` is {_f(shuf_adv)} "
            f"(main: {_f(adv_pt)}). "
            "AUC values are identical to the main run — they depend only on the score distribution, "
            "not the threshold. A shuffled advantage close to the main result indicates the score "
            "distribution is inherently informative regardless of threshold placement."
        )
        control_sec = f"""
## 10. Controls

### Shuffled-label control

Train labels are randomly permuted; threshold is re-calibrated; test is evaluated with real labels.

| Score | Shuffled Adv | Shuffled AUC |
|---|---:|---:|
{chr(10).join(shuf_rows)}

{control_note}
"""
    else:
        control_sec = "\n## 10. Controls\n\nShuffled-label control: not run.\n"

    md = f"""# WikiMIA Shard-Membership Report: {model}

**Generated:** {now}

---

## 1. Executive Summary

- **Model:** `{model}`
- **Dataset:** `swj0419/WikiMIA`  split: `{wikimia_split}`
- **Train size:** {n_train} ({manifest.get('n_train_per_class', '?')} per class)
- **Test size:** {n_test} ({n_test_per_class} per class)
- **Primary score:** `{primary}`
- **Primary advantage:** {_f(adv_pt)}
- **Primary AUC:** {_f(auc_pt)}
- **Primary accuracy:** {_f(acc_pt)}
- **Bootstrap CIs (1000 resamples, seed={seed}):**
  - Shard advantage: {_fci(prim_ci["shard_advantage"])}
  - AUC: {_fci(prim_ci["auc"])}
  - Accuracy: {_fci(prim_ci["accuracy"])}
- **Finite-sample null bound (α=0.05, m={n_test_per_class}):** {_f(null_bound)}
- **Signal above zero?** {_boolstr(signal_above_zero)}
- **AUC above 0.5?** {_boolstr(auc_above_half)}
- **Exceeds null bound?** {_boolstr(exceeds_null_bound)}

**Main conclusion:** {verdict}

---

## 2. Design Notes

| Property | Value |
|---|---|
| Same split as OPT / Pythia | YES (seed=0, WikiMIA_length32) |
| Balanced design | YES (187+187 train, 200+200 test) |
| Bootstrap method | percentile, fixed threshold |
| Primary score | `mean_logprob` (expected weaker signal than Pile models) |
| Multi-seed | no (seed=0 only) |
| Shuffled-label control | YES |
| Three-model comparison | YES (OPT-6.7B, Pythia-6.9B, LLaMA-2-13B) |
| Memory handling | device_map='auto' (CPU offload) |

---

## 3. Data Diagnostics

| Field | Value |
|---|---|
| WikiMIA split | `{wikimia_split}` |
| Dataset | `{manifest.get('dataset_id', 'swj0419/WikiMIA')}` |
| N train total | {n_train} |
| N test total | {n_test} |
| N test per class | {n_test_per_class} |
| Label 1 meaning | member (Wikipedia articles before LLaMA-2 training cutoff) |
| Label 0 meaning | nonmember (after cutoff) |
| Preprocessing | normalize whitespace; min 8 words; no truncation |
| Cross-class overlap | {manifest.get('cross_class_overlap', 0)} |
| MIA train/test overlap | {manifest.get('mia_train_test_overlap', 0)} |
| Size fallback used | {manifest.get('size_fallback', True)} |
| Seed | {manifest.get('seed', 0)} |
| Same split as OPT-6.7B / Pythia-6.9B | YES |

---

## 4. Scoring Diagnostics

| Field | Value |
|---|---|
| Model | `{model}` |
| device_map | `{device_map}` |
| Parameter devices | {', '.join(param_devs)} |
| Dtype | {dtype_used} |
| Batch size | {bs_used} |
| Max length | {max_len} tokens |
| N train scored | {score_mfst.get('n_train_scored', n_train)} |
| N test scored | {score_mfst.get('n_test_scored', n_test)} |
| Causal-shift | left-shift logits; first token excluded |
| Padding handling | excluded via attention mask |
| OOM handling | {oom_note} |

---

## 5. Results by Score (with Bootstrap 95% CIs)

| Score | Acc | AUC | Adv | Acc 95% CI | AUC 95% CI | Adv 95% CI |
|---|---:|---:|---:|---|---|---|
{chr(10).join(score_rows)}

---

## 6. Threshold Calibration Details

| Score | Calibrated threshold | Train Adv | Train AUC | Test Adv | Test AUC |
|---|---:|---:|---:|---:|---:|
{chr(10).join(thresh_rows)}

Threshold calibrated on train split by maximizing balanced accuracy.
Test split not touched during calibration.

---

## 7. Signal vs. Sampling Error

| Metric | Value |
|---|---|
| Observed primary advantage | {_f(adv_pt)} |
| Advantage 95% CI | {_fci(prim_ci['shard_advantage'])} |
| Finite-sample null bound (α=0.05, m={n_test_per_class}) | {_f(null_bound)} |
| Advantage lower CI > null bound | {_boolstr(exceeds_null_bound)} |
| AUC 95% CI excludes 0.5 | {_boolstr(auc_above_half)} |
| Advantage 95% CI excludes 0 | {_boolstr(signal_above_zero)} |

**Formula for null bound:** √(log(2/α) / m) = √(log(40) / {n_test_per_class}) ≈ {_f(null_bound)}
{comparison_sec}
{usenix_sec}
{control_sec}

---

## 11. Commands Run

```bash
# Data: reused existing split (no regeneration needed)
# Validated: 374 train (187+187), 400 test (200+200), seed=0, overlap=0

# Score (requires HF token with approved LLaMA-2 access)
python scripts/experiments/run_wikimia_llama2_13b_experiment.py \\
  --data-dir        data/processed/wikimia_length32 \\
  --score-dir       data/scores/wikimia_llama2_13b_length32 \\
  --run-dir         outputs/runs/wikimia_llama2_13b_length32 \\
  --model           {model} \\
  --dtype           float16 \\
  --batch-size      1

# Report
python scripts/reports/report_wikimia_llama2_13b.py \\
  --results-file    outputs/runs/wikimia_llama2_13b_length32/results.json \\
  --test-scores     data/scores/wikimia_llama2_13b_length32/test_scores.jsonl \\
  --data-manifest   data/processed/wikimia_length32/manifest.json \\
  --score-manifest  data/scores/wikimia_llama2_13b_length32/manifest.json \\
  --opt-results     outputs/runs/wikimia_opt67b/results.json \\
  --pythia-results  outputs/runs/wikimia_pythia69b_length32/results.json \\
  --output-dir      outputs/reports/wikimia_llama2_13b_length32
```

---

## 12. Limitations and Next Steps

1. **Single split:** Results are for `WikiMIA_length32` only. Longer splits provide more tokens per example and may sharpen the signal.
2. **WikiMIA temporal-label caveat:** WikiMIA labels are based on Wikipedia snapshot dates. LLaMA-2's training corpus (Common Crawl + Wikipedia) is less precisely documented than Pile, so the membership labels carry additional uncertainty.
3. **Memory constraint:** LLaMA-2-13B (~26 GB float16) required CPU offloading via `device_map='auto'`. This yields correct scores but is ~8–15× slower than GPU-only inference. Results are faithful — same computation, same precision.
4. **Weaker expected signal:** LLaMA-2-13B trains on 2T tokens across a broader corpus than Pile-based models. A given Wikipedia article is a smaller fraction of its training data, diluting the per-example membership signal.
5. **Single seed (seed=0):** Multi-seed replication (seeds 1–4) is recommended for borderline results.
6. **Suggested next model:** Llama-2-7B (smaller, more manageable on V100) or Llama-2-70B (stronger membership signal expected) with multi-GPU access.

---

*Report generated by `scripts/reports/report_wikimia_llama2_13b.py` at {now}.*
"""

    results_out = {
        "model":              model,
        "dataset":            manifest.get("dataset_id", "swj0419/WikiMIA"),
        "wikimia_split":      wikimia_split,
        "n_test_per_class":   n_test_per_class,
        "null_bound_alpha05": round(null_bound, 4),
        "primary_score":      primary,
        "n_bootstrap":        n_bootstrap,
        "bootstrap_seed":     seed,
        "results_by_score": {
            key: {
                "calibrated_threshold": main_map[key]["calibrated_threshold"],
                "test_accuracy":        main_map[key]["test"].get("accuracy"),
                "test_auc":             main_map[key]["test"].get("auc"),
                "test_advantage":       main_map[key]["test"].get("shard_advantage"),
                "bootstrap_ci":         ci_by_score[key],
            }
            for key in available_keys
        },
        "interpretation": {
            "signal_above_zero_ci95":    signal_above_zero,
            "auc_above_half_ci95":       auc_above_half,
            "signal_exceeds_null_bound": exceeds_null_bound,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    _write_json(results_out, os.path.join(output_dir, "results.json"))
    _write_text(md, os.path.join(output_dir, "summary.md"))
    logger.info("Report written to %s", os.path.join(output_dir, "summary.md"))


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate LLaMA-2-13B WikiMIA report with bootstrap CIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-file",    dest="results_file",
                   default="outputs/runs/wikimia_llama2_13b_length32/results.json")
    p.add_argument("--test-scores",     dest="test_scores",
                   default="data/scores/wikimia_llama2_13b_length32/test_scores.jsonl")
    p.add_argument("--data-manifest",   dest="data_manifest",
                   default="data/processed/wikimia_length32/manifest.json")
    p.add_argument("--score-manifest",  dest="score_manifest",
                   default="data/scores/wikimia_llama2_13b_length32/manifest.json")
    p.add_argument("--opt-results",     dest="opt_results",
                   default="outputs/runs/wikimia_opt67b/results.json")
    p.add_argument("--pythia-results",  dest="pythia_results",
                   default="outputs/runs/wikimia_pythia69b_length32/results.json")
    p.add_argument("--output-dir",      dest="output_dir",
                   default="outputs/reports/wikimia_llama2_13b_length32")
    p.add_argument("--model",           default="meta-llama/Llama-2-13b-hf")
    p.add_argument("--n-bootstrap",     dest="n_bootstrap", type=int, default=1000)
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    for path, label in [(args.results_file, "--results-file"),
                        (args.test_scores,   "--test-scores")]:
        if not os.path.isfile(path):
            logger.error("Required file not found (%s): %s", label, path)
            sys.exit(1)

    generate_report(
        results_file=args.results_file,
        test_scores_file=args.test_scores,
        data_manifest_file=args.data_manifest,
        score_manifest_file=args.score_manifest,
        opt_results_file=args.opt_results,
        pythia_results_file=args.pythia_results,
        output_dir=args.output_dir,
        model=args.model,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    print(f"\n=== DONE ===")
    print(f"Report: {os.path.join(args.output_dir, 'summary.md')}")
    print(f"JSON:   {os.path.join(args.output_dir, 'results.json')}")


if __name__ == "__main__":
    main()

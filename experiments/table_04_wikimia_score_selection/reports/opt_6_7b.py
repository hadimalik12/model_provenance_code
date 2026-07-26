"""Generate WikiMIA / OPT-6.7B shard-membership report with bootstrap 95% CIs.

Reads:
  - Experiment results.json  (calibrated threshold + point estimates)
  - Raw test score JSONL     (for bootstrap resampling)
  - Data manifest.json       (WikiMIA split metadata)

Produces:
  - outputs/reports/wikimia_opt67b/summary.md
  - outputs/reports/wikimia_opt67b/results.json

Bootstrap procedure:
  1. Resample the test set (with replacement, 1000 iterations).
  2. At each resample compute AUC, accuracy, and shard_advantage
     at the calibrated threshold from the original run.
  3. Report 2.5th / 97.5th percentile as the 95% CI.

This is a percentile bootstrap with fixed threshold. The threshold was
calibrated on the training split (not touched during bootstrap), so there
is no threshold optimism in the CIs.

Usage:
    python experiments/table_04_wikimia_score_selection/reports/opt_6_7b.py \\
        --results-file     outputs/runs/wikimia_opt67b/results.json \\
        --test-scores      data/scores/wikimia_opt67b/test_scores.jsonl \\
        --data-manifest    data/processed/wikimia_length32/manifest.json \\
        --output-dir       outputs/reports/wikimia_opt67b \\
        --model            facebook/opt-6.7b \\
        --n-bootstrap      1000 \\
        --seed             42
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

# WikiMIA paper reference values for OPT-6.7B, WikiMIA_length32.
# Source: Shi et al. (2024) "Detecting Pretraining Data from Large Language Models"
# Table 1 / Figure 3. Note: paper AUC is for the full test set.
PAPER_AUC_MINK_20 = None      # set from --paper-auc-mink20 if known
PAPER_AUC_MEANLOGP = None     # set from --paper-auc-meanlogp if known


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


def _evaluate_at_threshold(labels, scores, threshold):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    tpr = tp / n_pos if n_pos else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    acc = (tp + tn) / len(labels)
    return {"accuracy": acc, "shard_advantage": tpr - fpr, "tpr": tpr, "fpr": fpr}


def bootstrap_ci(
    labels: list,
    scores: list,
    threshold: float,
    n_resamples: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap 95% CIs for AUC, accuracy, shard_advantage.

    Resamples the (label, score) pairs with replacement. Threshold is fixed
    at the value calibrated on the training split.

    Returns {metric: {"mean": ..., "ci_lo": ..., "ci_hi": ...}}.
    """
    rng = random.Random(seed)
    n = len(labels)
    pairs = list(zip(labels, scores))

    auc_samples, acc_samples, adv_samples = [], [], []
    for _ in range(n_resamples):
        boot = [pairs[rng.randint(0, n - 1)] for _ in range(n)]
        bl, bs = zip(*boot)
        bl, bs = list(bl), list(bs)

        auc_samples.append(_safe_auc(bl, bs))
        m = _evaluate_at_threshold(bl, bs, threshold)
        acc_samples.append(m["accuracy"])
        adv_samples.append(m["shard_advantage"])

    def _ci(samples):
        valid = [x for x in samples if not math.isnan(x)]
        if not valid:
            return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        valid.sort()
        k = len(valid)
        lo_idx = max(0, int(math.floor(alpha / 2 * k)))
        hi_idx = min(k - 1, int(math.ceil((1 - alpha / 2) * k)) - 1)
        return {
            "mean": sum(valid) / k,
            "ci_lo": valid[lo_idx],
            "ci_hi": valid[hi_idx],
        }

    return {
        "auc":             _ci(auc_samples),
        "accuracy":        _ci(acc_samples),
        "shard_advantage": _ci(adv_samples),
    }


# ------------------------------------------------------------------ #
# Formatting
# ------------------------------------------------------------------ #

def _fmt(v, d=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{d}f}"


def _fmt_ci(ci_dict, d=3):
    """Format [lo, hi]."""
    if ci_dict is None:
        return "n/a"
    lo = _fmt(ci_dict.get("ci_lo"), d)
    hi = _fmt(ci_dict.get("ci_hi"), d)
    return f"[{lo}, {hi}]"


def _exceeds_null(ci_dict):
    """Return True if the 95% CI lower bound is strictly > 0."""
    if ci_dict is None:
        return False
    lo = ci_dict.get("ci_lo", float("nan"))
    if math.isnan(lo):
        return False
    return lo > 0.0


def _auc_exceeds_half(ci_dict):
    """Return True if the 95% CI lower bound for AUC is strictly > 0.5."""
    if ci_dict is None:
        return False
    lo = ci_dict.get("ci_lo", float("nan"))
    if math.isnan(lo):
        return False
    return lo > 0.5


# ------------------------------------------------------------------ #
# Report generation
# ------------------------------------------------------------------ #

def _results_map(results_file: str) -> dict:
    """Load results.json → {score_name: result_dict}."""
    raw = _load_json(results_file)
    lst = raw["main_results"] if isinstance(raw, dict) else raw
    return {r["score_name"]: r for r in lst}


def generate_report(
    results_file: str,
    test_scores_file: str,
    data_manifest_file: str,
    output_dir: str,
    model: str,
    n_bootstrap: int,
    seed: int,
    paper_auc_mink20=None,
    paper_auc_meanlogp=None,
) -> None:
    logger.info("Loading results from %s", results_file)
    res_map = _results_map(results_file)

    logger.info("Loading test scores from %s", test_scores_file)
    test_records = _load_jsonl(test_scores_file)

    manifest = {}
    if data_manifest_file and os.path.isfile(data_manifest_file):
        logger.info("Loading data manifest from %s", data_manifest_file)
        manifest = _load_json(data_manifest_file)

    available_keys = [k for k in SCORE_KEYS if k in res_map]
    if not available_keys:
        logger.error("No matching score keys found in results.json. Keys present: %s",
                     list(res_map.keys()))
        sys.exit(1)

    # Bootstrap CIs per score
    logger.info("Computing bootstrap CIs (%d resamples)...", n_bootstrap)
    ci_by_score = {}
    for key in available_keys:
        res = res_map[key]
        threshold = res["calibrated_threshold"]
        test_labels = [r["label"] for r in test_records]
        test_sc = [r[key] for r in test_records]
        ci = bootstrap_ci(
            labels=test_labels, scores=test_sc, threshold=threshold,
            n_resamples=n_bootstrap, seed=seed,
        )
        ci_by_score[key] = ci
        logger.info("  %-28s  Adv=%s  AUC=%s",
                    key, _fmt_ci(ci["shard_advantage"]), _fmt_ci(ci["auc"]))

    # Primary score
    primary = "min_k_20_logprob" if "min_k_20_logprob" in res_map else available_keys[0]
    primary_res = res_map[primary]
    primary_test = primary_res["test"]
    primary_ci = ci_by_score[primary]

    # Null bound: √(log(40)/n_test) at α=0.05
    n_test_per_class = manifest.get("n_test_per_class", primary_test.get("n_pos"))
    null_bound = math.sqrt(math.log(40) / n_test_per_class) if n_test_per_class else float("nan")

    # Is signal genuine?
    adv_ci = primary_ci["shard_advantage"]
    auc_ci  = primary_ci["auc"]
    signal_exceeds_null_bound = (
        adv_ci.get("ci_lo", float("nan")) > null_bound
        if not math.isnan(null_bound) else False
    )
    signal_above_zero = _exceeds_null(adv_ci)
    auc_above_half    = _auc_exceeds_half(auc_ci)

    # ------------------------------------------------------------------ #
    # Build output JSON
    # ------------------------------------------------------------------ #
    output = {
        "model": model,
        "dataset": manifest.get("dataset_id", "swj0419/WikiMIA"),
        "wikimia_split": manifest.get("wikimia_split", "WikiMIA_length32"),
        "n_test_per_class": n_test_per_class,
        "null_bound_alpha05": round(null_bound, 4) if not math.isnan(null_bound) else None,
        "primary_score": primary,
        "n_bootstrap": n_bootstrap,
        "bootstrap_seed": seed,
        "results_by_score": {
            key: {
                "calibrated_threshold": res_map[key]["calibrated_threshold"],
                "test_accuracy": primary_test.get("accuracy")
                    if key == primary else res_map[key]["test"].get("accuracy"),
                "test_auc": primary_test.get("auc")
                    if key == primary else res_map[key]["test"].get("auc"),
                "test_advantage": primary_test.get("shard_advantage")
                    if key == primary else res_map[key]["test"].get("shard_advantage"),
                "bootstrap_ci": ci_by_score[key],
            }
            for key in available_keys
        },
        "interpretation": {
            "signal_above_zero_ci95":     signal_above_zero,
            "auc_above_half_ci95":        auc_above_half,
            "signal_exceeds_null_bound":  signal_exceeds_null_bound,
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(output, os.path.join(output_dir, "results.json"))

    # ------------------------------------------------------------------ #
    # Build Markdown report
    # ------------------------------------------------------------------ #
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    wikimia_split = manifest.get("wikimia_split", "WikiMIA_length32")
    n_test  = manifest.get("n_test",  "?")
    n_train = manifest.get("n_train", "?")

    def _boolstr(v):
        return "**YES**" if v else "no"

    # Score-level table
    rows = []
    for key in available_keys:
        r = res_map[key]
        ci = ci_by_score[key]
        t = r["test"]
        adv_pt = _fmt(t.get("shard_advantage"))
        auc_pt = _fmt(t.get("auc"))
        acc_pt = _fmt(t.get("accuracy"))
        adv_str = _fmt_ci(ci["shard_advantage"])
        auc_str = _fmt_ci(ci["auc"])
        acc_str = _fmt_ci(ci["accuracy"])
        rows.append(
            f"| `{key}` | {acc_pt} | {auc_pt} | {adv_pt} "
            f"| {acc_str} | {auc_str} | {adv_str} |"
        )
    score_table = "\n".join(rows)

    # Interpretation
    if signal_above_zero and auc_above_half:
        if signal_exceeds_null_bound:
            interp = (
                f"The MIN-K 20% signal is **statistically significant** at 95% confidence. "
                f"Both the shard advantage and AUC CIs exclude their null values, and "
                f"the lower CI bound ({_fmt(adv_ci.get('ci_lo'))}) exceeds the finite-sample "
                f"null bound ({_fmt(null_bound)}). The membership signal survives sampling noise."
            )
        else:
            interp = (
                f"The shard advantage CI excludes zero and the AUC CI excludes 0.5, "
                f"but the lower CI bound ({_fmt(adv_ci.get('ci_lo'))}) does not exceed "
                f"the finite-sample null bound ({_fmt(null_bound)}). "
                f"Signal is detectable but borderline; more test examples would help."
            )
    elif signal_above_zero or auc_above_half:
        interp = (
            f"Partial signal: one of the two indicators (advantage CI, AUC CI) excludes "
            f"its null value but the other does not. Signal is weak and caution is warranted. "
            f"Collect more test examples or try a sequence-level distinguisher."
        )
    else:
        interp = (
            f"No reliable signal detected. Both the shard advantage CI and AUC CI are "
            f"consistent with random chance. MIN-K% and mean log-prob scalar distinguishers "
            f"are insufficient for this model/split combination."
        )

    # Paper comparison section
    paper_sec = ""
    if paper_auc_mink20 is not None or paper_auc_meanlogp is not None:
        rows_p = []
        if paper_auc_mink20 is not None:
            our_auc = _fmt(res_map.get("min_k_20_logprob", {}).get("test", {}).get("auc"))
            rows_p.append(
                f"| `min_k_20_logprob` | {_fmt(paper_auc_mink20)} | {our_auc} |"
            )
        if paper_auc_meanlogp is not None:
            our_auc = _fmt(res_map.get("mean_logprob", {}).get("test", {}).get("auc"))
            rows_p.append(
                f"| `mean_logprob` | {_fmt(paper_auc_meanlogp)} | {our_auc} |"
            )
        paper_table = "\n".join(rows_p)
        paper_sec = f"""
---

## 7. Comparison to Published WikiMIA Results

| Score | Paper AUC (Shi et al.) | Our AUC |
|---|---:|---:|
{paper_table}

Note: paper values are for the full (non-subsampled) test set. Small differences
are expected due to different test-set sizes and preprocessing.
"""

    md = f"""# WikiMIA Shard-Membership Report: {model}

**Generated:** {now}

---

## 1. Executive Summary

- **Model:** `{model}`
- **Dataset:** `swj0419/WikiMIA`  split: `{wikimia_split}`
- **Primary score:** `{primary}`
- **Test set:** {n_test} records ({n_test_per_class} per class)
- **Bootstrap:** {n_bootstrap} resamples, seed={seed}

| Metric | Point est. | 95% CI |
|---|---:|---|
| Shard advantage | {_fmt(primary_test.get('shard_advantage'))} | {_fmt_ci(primary_ci['shard_advantage'])} |
| AUC | {_fmt(primary_test.get('auc'))} | {_fmt_ci(primary_ci['auc'])} |
| Accuracy | {_fmt(primary_test.get('accuracy'))} | {_fmt_ci(primary_ci['accuracy'])} |
| Finite-sample null bound (α=0.05) | {_fmt(null_bound)} | — |

Signal above zero (adv CI₉₅ > 0): {_boolstr(signal_above_zero)}
AUC above 0.5 (AUC CI₉₅ > 0.5): {_boolstr(auc_above_half)}
Exceeds null bound: {_boolstr(signal_exceeds_null_bound)}

**Interpretation:** {interp}

---

## 2. Experimental Setup

- **Dataset source:** `swj0419/WikiMIA`, split `{wikimia_split}`
- **Label semantics:** `label=1` = member of model's pretraining corpus; `label=0` = nonmember
- **Preprocessing:** normalize whitespace; filter < 8 words; no truncation (WikiMIA is pre-partitioned by length)
- **Scoring:** causal LM log-probabilities with causal shift; MIN-K% PROB at k=5,10,20,40; mean log-prob
- **Threshold calibration:** maximize balanced accuracy on the train split
- **Evaluation:** held-out test split (no contamination from calibration)
- **Bootstrap:** percentile bootstrap on the TEST split at the fixed calibrated threshold

---

## 3. Data Diagnostics

| Field | Value |
|---|---|
| WikiMIA split | `{wikimia_split}` |
| N train total | {n_train} |
| N test total | {n_test} |
| N test per class | {n_test_per_class} |
| Size fallback | {manifest.get('size_fallback', '?')} |
| Cross-class overlap | {manifest.get('cross_class_overlap', '?')} |
| MIA train/test overlap | {manifest.get('mia_train_test_overlap', '?')} |

---

## 4. Results by Score (with Bootstrap 95% CIs)

| Score | Acc (pt) | AUC (pt) | Adv (pt) | Acc 95% CI | AUC 95% CI | Adv 95% CI |
|---|---:|---:|---:|---|---|---|
{score_table}

---

## 5. Threshold Calibration Details

| Score | Calibrated threshold | Train Adv | Train AUC | Test Adv | Test AUC |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| `{k}` | {_fmt(res_map[k]['calibrated_threshold'], 4)} "
        f"| {_fmt(res_map[k]['train'].get('shard_advantage'))} "
        f"| {_fmt(res_map[k]['train'].get('auc'))} "
        f"| {_fmt(res_map[k]['test'].get('shard_advantage'))} "
        f"| {_fmt(res_map[k]['test'].get('auc'))} |"
        for k in available_keys
    ) + f"""

---

## 6. Interpretation

### What the null bound means

With {n_test_per_class} test examples per class and α=0.05, the finite-sample
null bound is **{_fmt(null_bound)}** (formula: √(log(40)/n_test_per_class)).
Observed advantage above this bound is unlikely to be explained by sampling noise alone.

### Signal interpretation

{interp}

### Limitations

1. **Single model, single split.** Results for `{wikimia_split}` may not generalise to longer splits.
2. **Scalar distinguisher only.** MIN-K% PROB and mean log-prob are aggregate scores. A sequence-level GRU or per-token distinguisher may find signal missed here.
3. **Single seed.** Bootstrap CIs account for test-set sampling noise but not for data selection noise (train/test split randomness). Multi-seed replication is recommended for borderline results.
4. **Fixed threshold.** Bootstrap CIs use the threshold calibrated on the original train split. CI width reflects test-set variance only.
5. **WikiMIA label quality.** WikiMIA labels are based on Wikipedia snapshot dates; some member texts may appear in both pre- and post-cutoff corpora.
{paper_sec}
---

## 8. Next Steps

1. Score additional WikiMIA splits (`WikiMIA_length64`, `WikiMIA_length128`) to check if longer texts yield stronger signal.
2. Run a **null control** (randomize labels) to verify the CI procedure.
3. Try a **GRU/sequence-level distinguisher** for splits where scalar scores are weak.
4. Repeat with **multi-seed** data splits (seeds 1–4) to estimate split-selection noise.

---

*Report generated by `experiments/table_04_wikimia_score_selection/reports/opt_6_7b.py` at {now}.*
"""

    report_path = os.path.join(output_dir, "summary.md")
    _write_text(md, report_path)
    logger.info("Report written to %s", report_path)


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate WikiMIA / OPT-6.7B report with bootstrap 95% CIs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-file",  default="outputs/runs/wikimia_opt67b/results.json",
                   dest="results_file")
    p.add_argument("--test-scores",   default="data/scores/wikimia_opt67b/test_scores.jsonl",
                   dest="test_scores")
    p.add_argument("--data-manifest", default="data/processed/wikimia_length32/manifest.json",
                   dest="data_manifest")
    p.add_argument("--output-dir",    default="outputs/reports/wikimia_opt67b")
    p.add_argument("--model",         default="facebook/opt-6.7b")
    p.add_argument("--n-bootstrap",   type=int, default=1000, dest="n_bootstrap")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--paper-auc-mink20", type=float, default=None, dest="paper_auc_mink20",
                   help="Published AUC for min_k_20 (from Shi et al.) for comparison")
    p.add_argument("--paper-auc-meanlogp", type=float, default=None, dest="paper_auc_meanlogp",
                   help="Published AUC for mean_logprob (from Shi et al.) for comparison")
    return p.parse_args()


def main():
    args = parse_args()

    for path, label in [
        (args.results_file, "--results-file"),
        (args.test_scores,  "--test-scores"),
    ]:
        if not os.path.isfile(path):
            logger.error("Required file not found (%s): %s", label, path)
            sys.exit(1)

    generate_report(
        results_file=args.results_file,
        test_scores_file=args.test_scores,
        data_manifest_file=args.data_manifest,
        output_dir=args.output_dir,
        model=args.model,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        paper_auc_mink20=args.paper_auc_mink20,
        paper_auc_meanlogp=args.paper_auc_meanlogp,
    )
    print(f"\n=== DONE ===")
    print(f"Report: {os.path.join(args.output_dir, 'summary.md')}")
    print(f"JSON:   {os.path.join(args.output_dir, 'results.json')}")


if __name__ == "__main__":
    main()

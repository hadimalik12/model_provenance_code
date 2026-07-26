"""Nonmember-vs-nonmember null control for WikiMIA OPT-6.7B and Pythia-6.9B.

Both pseudo-classes are drawn exclusively from the WikiMIA nonmember pool
(label=0 examples). Labels are ARTIFICIAL.

  pseudo-label = 1  →  pseudo-shard A  (still nonmember w.r.t. pretraining)
  pseudo-label = 0  →  pseudo-shard B  (still nonmember w.r.t. pretraining)

Scientific purpose:
    If the distinguisher is finding genuine membership signal in the main
    experiment, it should show near-zero advantage here. Combining a strong
    main result with a near-zero control result supports that the signal is
    tied to actual pretraining membership and not a scoring artefact.

Protocol:
    1. Extract label=0 (nonmember) records from train_scores.jsonl (N_tr nonmembers).
    2. Extract label=0 (nonmember) records from test_scores.jsonl  (N_te nonmembers).
    3. Shuffle each pool deterministically (seed=7).
    4. Assign pseudo-labels: first half → 1, second half → 0.
    5. Calibrate threshold on pseudo-train by maximizing balanced accuracy.
    6. Evaluate on pseudo-test.
    7. Compute bootstrap 95% CI on pseudo-test.

Null bound: √(log(40) / m) where m = pseudo-test examples per pseudo-class.

Usage:
    python scripts/experiments/run_wikimia_null_control.py \\
        --opt-scores-dir       data/scores/wikimia_opt67b \\
        --pythia-scores-dir    data/scores/wikimia_pythia69b_length32 \\
        --opt-main-results     outputs/runs/wikimia_opt67b/results.json \\
        --pythia-main-results  outputs/runs/wikimia_pythia69b_length32/results.json \\
        --output-dir           outputs/reports/wikimia_null_control \\
        --seed                 7 \\
        --bootstrap-seed       42 \\
        --n-bootstrap          1000
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.auditing.distinguishers import run_distinguisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCORE_KEYS = ["mean_logprob", "min_k_5_logprob", "min_k_10_logprob",
              "min_k_20_logprob", "min_k_40_logprob"]


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
# Control construction
# ------------------------------------------------------------------ #

def _build_pseudo_split(records: list, seed: int) -> tuple:
    """Shuffle nonmember records and assign pseudo-labels.

    Returns (pseudo_labels, {score_key: scores}) for threshold experiment.
    First half gets pseudo-label=1, second half gets pseudo-label=0.
    """
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n = len(shuffled)
    half = n // 2
    pseudo_labels = [1] * half + [0] * (n - half)
    score_dict = {}
    avail = [k for k in SCORE_KEYS if k in shuffled[0]]
    for key in avail:
        score_dict[key] = [r[key] for r in shuffled]
    return pseudo_labels, score_dict, half, n - half


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
    return {"accuracy": (tp + tn) / len(labels), "shard_advantage": tpr - fpr}


def bootstrap_ci(labels, scores, threshold, n_resamples=1000, seed=42, alpha=0.05):
    rng = random.Random(seed)
    n = len(labels)
    pairs = list(zip(labels, scores))
    auc_s, adv_s = [], []
    for _ in range(n_resamples):
        boot = [pairs[rng.randint(0, n - 1)] for _ in range(n)]
        bl, bs = zip(*boot)
        auc_s.append(_safe_auc(list(bl), list(bs)))
        m = _eval_at_threshold(list(bl), list(bs), threshold)
        adv_s.append(m["shard_advantage"])

    def _ci(samples):
        valid = sorted(x for x in samples if not math.isnan(x))
        if not valid:
            return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
        k = len(valid)
        lo = valid[max(0, int(math.floor(alpha / 2 * k)))]
        hi = valid[min(k - 1, int(math.ceil((1 - alpha / 2) * k)) - 1)]
        return {"mean": sum(valid) / k, "ci_lo": lo, "ci_hi": hi}

    return {"auc": _ci(auc_s), "shard_advantage": _ci(adv_s)}


# ------------------------------------------------------------------ #
# Per-model control run
# ------------------------------------------------------------------ #

def run_model_control(
    scores_dir: str,
    model_label: str,
    primary_score: str,
    seed: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> dict:
    """Run nonmember-vs-nonmember control for one model. Returns results dict."""
    train_records = _load_jsonl(os.path.join(scores_dir, "train_scores.jsonl"))
    test_records  = _load_jsonl(os.path.join(scores_dir, "test_scores.jsonl"))

    # Filter to nonmembers only
    train_nm = [r for r in train_records if r["label"] == 0]
    test_nm  = [r for r in test_records  if r["label"] == 0]
    logger.info("[%s] Nonmember pool: %d train, %d test",
                model_label, len(train_nm), len(test_nm))

    # Build pseudo-splits
    tr_labels, tr_scores, tr_n1, tr_n0 = _build_pseudo_split(train_nm, seed=seed)
    te_labels, te_scores, te_n1, te_n0 = _build_pseudo_split(test_nm, seed=seed + 1)

    avail_keys = [k for k in SCORE_KEYS if k in tr_scores]
    results_by_score = {}

    for key in avail_keys:
        res = run_distinguisher(
            tr_labels, tr_scores[key],
            te_labels, te_scores[key],
            score_name=key,
        )
        threshold = res["calibrated_threshold"]
        ci = bootstrap_ci(
            te_labels, te_scores[key], threshold,
            n_resamples=n_bootstrap, seed=bootstrap_seed,
        )
        results_by_score[key] = {
            "calibrated_threshold":  threshold,
            "train_advantage":       res["train"].get("shard_advantage"),
            "train_auc":             res["train"].get("auc"),
            "test_advantage":        res["test"].get("shard_advantage"),
            "test_auc":              res["test"].get("auc"),
            "test_accuracy":         res["test"].get("accuracy"),
            "bootstrap_ci":          ci,
        }

    null_bound = math.sqrt(math.log(40) / te_n1)  # per pseudo-class
    logger.info("[%s] Control results for primary score (%s):", model_label, primary_score)
    pr = results_by_score.get(primary_score, {})
    logger.info("  Test Adv=%.3f  AUC=%.3f  null_bound=%.3f",
                pr.get("test_advantage", float("nan")),
                pr.get("test_auc", float("nan")),
                null_bound)

    return {
        "model":             model_label,
        "n_train_nonmember": len(train_nm),
        "n_test_nonmember":  len(test_nm),
        "n_pseudo_per_class_train": tr_n1,
        "n_pseudo_per_class_test":  te_n1,
        "null_bound_alpha05": round(null_bound, 4),
        "seed":              seed,
        "results_by_score":  results_by_score,
    }


# ------------------------------------------------------------------ #
# Report
# ------------------------------------------------------------------ #

def _f(v, d=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{d}f}"


def _fci(ci, d=3):
    if ci is None:
        return "n/a"
    return f"[{_f(ci.get('ci_lo'), d)}, {_f(ci.get('ci_hi'), d)}]"


def _main_adv(main_results_file: str, score_key: str) -> tuple:
    """Return (main_adv, main_auc) from a main results.json."""
    if not main_results_file or not os.path.isfile(main_results_file):
        return float("nan"), float("nan")
    raw = _load_json(main_results_file)
    lst = raw["main_results"] if isinstance(raw, dict) else raw
    m = {r["score_name"]: r for r in lst}
    r = m.get(score_key, {})
    return (r.get("test", {}).get("shard_advantage", float("nan")),
            r.get("test", {}).get("auc", float("nan")))


def build_report(
    opt_ctrl: dict,
    pythia_ctrl: dict,
    opt_main_file: str,
    pythia_main_file: str,
    primary_score: str,
    n_bootstrap: int,
    bootstrap_seed: int,
    output_dir: str,
) -> None:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Null bounds
    opt_nb    = opt_ctrl["null_bound_alpha05"]
    pythia_nb = pythia_ctrl["null_bound_alpha05"]
    opt_n_te  = opt_ctrl["n_pseudo_per_class_test"]
    py_n_te   = pythia_ctrl["n_pseudo_per_class_test"]

    # Main results for comparison
    opt_main_adv,    opt_main_auc    = _main_adv(opt_main_file,    primary_score)
    pythia_main_adv, pythia_main_auc = _main_adv(pythia_main_file, primary_score)

    def _row(ctrl: dict, key: str):
        r = ctrl["results_by_score"].get(key, {})
        ci = r.get("bootstrap_ci", {})
        return (f"| `{key}` | {_f(r.get('test_advantage'))} | {_f(r.get('test_auc'))} "
                f"| {_fci(ci.get('shard_advantage'))} | {_fci(ci.get('auc'))} |")

    avail_keys = [k for k in SCORE_KEYS if k in opt_ctrl["results_by_score"]]

    opt_rows    = "\n".join(_row(opt_ctrl,    k) for k in avail_keys)
    pythia_rows = "\n".join(_row(pythia_ctrl, k) for k in avail_keys)

    # Per-model primary verdict
    def _verdict(ctrl: dict, null_bound: float, main_adv: float) -> str:
        pr = ctrl["results_by_score"].get(primary_score, {})
        ci = pr.get("bootstrap_ci", {}).get("shard_advantage", {})
        adv = pr.get("test_advantage", float("nan"))
        ci_lo = ci.get("ci_lo", float("nan"))
        ci_hi = ci.get("ci_hi", float("nan"))
        if math.isnan(adv):
            return "n/a"
        above_null = ci_lo > null_bound
        ci_includes_zero = ci_lo <= 0.0
        if ci_includes_zero:
            return (f"Control advantage {_f(adv)} with CI [{_f(ci_lo)}, {_f(ci_hi)}] "
                    f"**includes zero** — null hypothesis not rejected. ✓")
        elif not above_null:
            return (f"Control advantage {_f(adv)} with CI [{_f(ci_lo)}, {_f(ci_hi)}] "
                    f"excludes zero but stays below the null bound ({_f(null_bound)}). "
                    f"Borderline — consistent with finite-sample noise.")
        else:
            return (f"**WARNING:** Control advantage {_f(adv)} CI [{_f(ci_lo)}, {_f(ci_hi)}] "
                    f"exceeds null bound ({_f(null_bound)}). "
                    f"Unexpected signal in the null control. Investigate.")

    opt_verdict    = _verdict(opt_ctrl,    opt_nb,    opt_main_adv)
    pythia_verdict = _verdict(pythia_ctrl, pythia_nb, pythia_main_adv)

    md = f"""# WikiMIA Nonmember-vs-Nonmember Null Control Report

**Generated:** {now}

---

## 1. Purpose

This report tests whether the membership distinguishers find spurious signal
when **both pseudo-classes are drawn from the nonmember pool**.

If the main-experiment signal is genuine (tied to pretraining membership),
the control advantage should be near zero and the control AUC near 0.5.

**Protocol:**
- Nonmember records are split into two pseudo-classes (pseudo-label 1 and 0).
- The threshold is calibrated on a pseudo-train set, evaluated on a pseudo-test set.
- Bootstrap 95% CI on pseudo-test.

**Models tested:** OPT-6.7B, Pythia-6.9B
**Dataset:** WikiMIA_length32 (nonmember pool only)
**No additional GPU scoring required** — reuses existing test score files.

---

## 2. Design

| Property | Value |
|---|---|
| Dataset | `swj0419/WikiMIA`, split `WikiMIA_length32` |
| Pseudo-class source | WikiMIA **nonmembers only** (label=0 in main experiment) |
| OPT nonmember pool | {opt_ctrl['n_train_nonmember']} train + {opt_ctrl['n_test_nonmember']} test |
| Pythia nonmember pool | {pythia_ctrl['n_train_nonmember']} train + {pythia_ctrl['n_test_nonmember']} test |
| Pseudo-class split | first half → pseudo-label=1; second half → pseudo-label=0 |
| Control pseudo-test per class | OPT: {opt_n_te}, Pythia: {py_n_te} |
| Finite-sample null bound (OPT) | {_f(opt_nb)} |
| Finite-sample null bound (Pythia) | {_f(pythia_nb)} |
| Shuffle seed | {opt_ctrl['seed']} |
| Bootstrap | {n_bootstrap} resamples, seed={bootstrap_seed} |

---

## 3. Main vs. Control Comparison

| Model | Setting | `{primary_score}` Adv | AUC |
|---|---|---:|---:|
| OPT-6.7B | **Main experiment** | {_f(opt_main_adv)} | {_f(opt_main_auc)} |
| OPT-6.7B | **Null control** | {_f(opt_ctrl['results_by_score'].get(primary_score, {}).get('test_advantage'))} | {_f(opt_ctrl['results_by_score'].get(primary_score, {}).get('test_auc'))} |
| Pythia-6.9B | **Main experiment** | {_f(pythia_main_adv)} | {_f(pythia_main_auc)} |
| Pythia-6.9B | **Null control** | {_f(pythia_ctrl['results_by_score'].get(primary_score, {}).get('test_advantage'))} | {_f(pythia_ctrl['results_by_score'].get(primary_score, {}).get('test_auc'))} |

---

## 4. OPT-6.7B Null Control Results

**Pseudo-test per class:** {opt_n_te}
**Null bound (α=0.05):** {_f(opt_nb)}

| Score | Control Adv | Control AUC | Adv 95% CI | AUC 95% CI |
|---|---:|---:|---|---|
{opt_rows}

**Verdict:** {opt_verdict}

---

## 5. Pythia-6.9B Null Control Results

**Pseudo-test per class:** {py_n_te}
**Null bound (α=0.05):** {_f(pythia_nb)}

| Score | Control Adv | Control AUC | Adv 95% CI | AUC 95% CI |
|---|---:|---:|---|---|
{pythia_rows}

**Verdict:** {pythia_verdict}

---

## 6. Interpretation

A valid null control satisfies at least one of:
- Control advantage 95% CI includes zero
- Control AUC 95% CI includes 0.5

If the control is well-behaved and the main experiment shows strong signal,
we can attribute the main advantage to genuine pretraining membership.

### Comparison of signal to null

| Model | Main Adv | Control Adv | Main AUC | Control AUC | Control CI ⊇ 0 |
|---|---:|---:|---:|---:|---|
| OPT-6.7B | {_f(opt_main_adv)} | {_f(opt_ctrl['results_by_score'].get(primary_score, {}).get('test_advantage'))} | {_f(opt_main_auc)} | {_f(opt_ctrl['results_by_score'].get(primary_score, {}).get('test_auc'))} | {'yes' if opt_ctrl['results_by_score'].get(primary_score, {}).get('bootstrap_ci', {}).get('shard_advantage', {}).get('ci_lo', 1.0) <= 0 else 'no'} |
| Pythia-6.9B | {_f(pythia_main_adv)} | {_f(pythia_ctrl['results_by_score'].get(primary_score, {}).get('test_advantage'))} | {_f(pythia_main_auc)} | {_f(pythia_ctrl['results_by_score'].get(primary_score, {}).get('test_auc'))} | {'yes' if pythia_ctrl['results_by_score'].get(primary_score, {}).get('bootstrap_ci', {}).get('shard_advantage', {}).get('ci_lo', 1.0) <= 0 else 'no'} |

---

## 7. Limitations

1. **Small pseudo-class pool.** WikiMIA_length32 has only 200 nonmember test examples total → 100 per pseudo-class. The null bound ({_f(opt_nb)}) is wide.
2. **Pool reuse.** The control uses the same nonmember examples that appear in the main experiment's test set. There is no independent held-out nonmember pool.
3. **Single pseudo-split seed.** Results may vary with different random seeds for the pseudo-class assignment.

---

*Report generated by `scripts/experiments/run_wikimia_null_control.py` at {now}.*
"""

    combined = {
        "primary_score":      primary_score,
        "n_bootstrap":        n_bootstrap,
        "bootstrap_seed":     bootstrap_seed,
        "opt_control":        opt_ctrl,
        "pythia_control":     pythia_ctrl,
        "main_comparison": {
            "opt":    {"adv": opt_main_adv,    "auc": opt_main_auc},
            "pythia": {"adv": pythia_main_adv, "auc": pythia_main_auc},
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    _write_json(combined, os.path.join(output_dir, "results.json"))
    _write_text(md, os.path.join(output_dir, "summary.md"))
    logger.info("Report written to %s", os.path.join(output_dir, "summary.md"))


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="WikiMIA nonmember-vs-nonmember null control for OPT and Pythia.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--opt-scores-dir",
                   default="data/scores/wikimia_opt67b",
                   dest="opt_scores_dir")
    p.add_argument("--pythia-scores-dir",
                   default="data/scores/wikimia_pythia69b_length32",
                   dest="pythia_scores_dir")
    p.add_argument("--opt-main-results",
                   default="outputs/runs/wikimia_opt67b/results.json",
                   dest="opt_main_results")
    p.add_argument("--pythia-main-results",
                   default="outputs/runs/wikimia_pythia69b_length32/results.json",
                   dest="pythia_main_results")
    p.add_argument("--output-dir",     default="outputs/reports/wikimia_null_control")
    p.add_argument("--primary-score",  default="min_k_20_logprob")
    p.add_argument("--seed",           type=int, default=7)
    p.add_argument("--bootstrap-seed", type=int, default=42, dest="bootstrap_seed")
    p.add_argument("--n-bootstrap",    type=int, default=1000, dest="n_bootstrap")
    return p.parse_args()


def main():
    args = parse_args()

    for path, label in [
        (os.path.join(args.opt_scores_dir,    "train_scores.jsonl"), "OPT train scores"),
        (os.path.join(args.opt_scores_dir,    "test_scores.jsonl"),  "OPT test scores"),
        (os.path.join(args.pythia_scores_dir, "train_scores.jsonl"), "Pythia train scores"),
        (os.path.join(args.pythia_scores_dir, "test_scores.jsonl"),  "Pythia test scores"),
    ]:
        if not os.path.isfile(path):
            logger.error("Required file not found (%s): %s", label, path)
            sys.exit(1)

    logger.info("=== WikiMIA Nonmember-vs-Nonmember Null Control ===")

    logger.info("\n--- OPT-6.7B ---")
    opt_ctrl = run_model_control(
        scores_dir=args.opt_scores_dir,
        model_label="OPT-6.7B",
        primary_score=args.primary_score,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    logger.info("\n--- Pythia-6.9B ---")
    pythia_ctrl = run_model_control(
        scores_dir=args.pythia_scores_dir,
        model_label="Pythia-6.9B",
        primary_score=args.primary_score,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )

    logger.info("\n--- Building report ---")
    build_report(
        opt_ctrl=opt_ctrl,
        pythia_ctrl=pythia_ctrl,
        opt_main_file=args.opt_main_results,
        pythia_main_file=args.pythia_main_results,
        primary_score=args.primary_score,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        output_dir=args.output_dir,
    )

    print(f"\n=== DONE ===")
    print(f"Report: {os.path.join(args.output_dir, 'summary.md')}")
    print(f"JSON:   {os.path.join(args.output_dir, 'results.json')}")


if __name__ == "__main__":
    main()

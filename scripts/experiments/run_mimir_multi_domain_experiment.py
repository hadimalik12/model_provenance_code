"""Score and evaluate both models across all prepared MIMIR domains.

Loads each model ONCE and scores all domains, then runs threshold experiments.
This avoids reloading 1.4B-parameter models for each domain.

Usage:
    python scripts/experiments/run_mimir_multi_domain_experiment.py \\
        --processed-root data/processed/mimir_domains \\
        --models EleutherAI/pythia-1.4b,nnheui/pythia-1.4b-sft-full \\
        --model-slugs parent_pythia_1_4b,target_nnheui_pythia_1_4b_sft_full \\
        --min-k-pcts 5,10,20,40 \\
        --batch-size 4 \\
        --score-root data/scores/mimir_domains \\
        --run-root outputs/runs/mimir_domains \\
        --primary-score min_k_20_logprob
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.logprobs import get_device, load_model_and_tokenizer, extract_token_logprobs
from src.shard_audit.mia_scores import compute_all_scores
from src.shard_audit.metrics import score_diagnostics
from src.shard_audit.distinguishers import run_distinguisher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCORE_KEYS = ["mean_logprob", "min_k_5_logprob", "min_k_10_logprob",
              "min_k_20_logprob", "min_k_40_logprob"]


# ------------------------------------------------------------------ #
# I/O helpers
# ------------------------------------------------------------------ #

def _load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _token_length_histogram(lengths: list) -> dict:
    from collections import Counter
    bins = [0, 30, 40, 64, 128, 256, 512, 999999]
    counts = Counter()
    for l in lengths:
        for lo, hi in zip(bins, bins[1:]):
            if lo <= l < hi:
                counts[f"{lo}-{hi-1}"] += 1
                break
    return dict(counts)


# ------------------------------------------------------------------ #
# Scoring (with pre-loaded model)
# ------------------------------------------------------------------ #

def score_domain(
    domain_dir: str,
    score_dir: str,
    model,
    tokenizer,
    device,
    model_name: str,
    k_pcts: list,
    batch_size: int,
    max_length: int = 512,
) -> bool:
    """Score one domain with an already-loaded model. Returns True if scored."""
    train_scores_path = os.path.join(score_dir, "train_scores.jsonl")
    test_scores_path  = os.path.join(score_dir, "test_scores.jsonl")

    if os.path.isfile(train_scores_path) and os.path.isfile(test_scores_path):
        logger.info("  Scores already exist at %s — skipping.", score_dir)
        return False

    train_path = os.path.join(domain_dir, "train.jsonl")
    test_path  = os.path.join(domain_dir, "test.jsonl")
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        logger.warning("  Missing data files in %s — skipping.", domain_dir)
        return False

    train_records = _load_jsonl(train_path)
    test_records  = _load_jsonl(test_path)

    def _score_split(records, split_name):
        results = []
        token_lengths = []
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            texts = [r["text"] for r in batch]
            batch_lp = extract_token_logprobs(texts, model, tokenizer, device,
                                              max_length=max_length)
            for rec, lp in zip(batch, batch_lp):
                scores = compute_all_scores(rec["text"], lp, k_pcts=tuple(k_pcts))
                score_rec = {
                    "id": rec["id"],
                    "label": rec["label"],
                    "phase_split": rec.get("phase_split", split_name),
                    "text_hash": rec["text_hash"],
                    "model": model_name,
                    "num_input_tokens": len(lp) + 1,
                    "num_scored_tokens": len(lp),
                    "mean_logprob": round(scores["mean_logprob"], 6),
                    "mean_loss": round(scores["mean_loss"], 6),
                    **{f"min_k_{k}_logprob": round(scores[f"min_k_{k}_logprob"], 6)
                       for k in k_pcts},
                    "zlib_norm_logprob": scores.get("zlib_norm_logprob"),
                }
                results.append(score_rec)
                token_lengths.append(len(lp))
            if (i // batch_size + 1) % 10 == 0:
                logger.info("    Scored %d / %d %s examples...",
                            min(i + batch_size, len(records)), len(records), split_name)
        return results, token_lengths

    logger.info("  Scoring %d train examples...", len(train_records))
    train_scores, train_lens = _score_split(train_records, "train")
    logger.info("  Scoring %d test examples...", len(test_records))
    test_scores, test_lens = _score_split(test_records, "test")

    # Diagnostics — score_diagnostics expects a list of score record dicts
    import torch
    dtype = next(model.parameters()).dtype
    diag_train = score_diagnostics(train_scores, score_keys=tuple(SCORE_KEYS))
    diag_test  = score_diagnostics(test_scores,  score_keys=tuple(SCORE_KEYS))
    for k, d in diag_train.items():
        dir_ok = "OK" if d.get("direction_ok") else "INVERTED"
        logger.info("  [train] %-28s AUC=%.4f dir=%s",
                    k, d.get("auc", float("nan")), dir_ok)

    manifest = {
        "model": model_name,
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": batch_size,
        "k_pcts": k_pcts,
        "n_train_scored": len(train_scores),
        "n_test_scored": len(test_scores),
        "token_length_histogram": _token_length_histogram(train_lens + test_lens),
        "score_diagnostics_train": diag_train,
        "score_diagnostics_test": diag_test,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    _write_jsonl(train_scores, train_scores_path)
    _write_jsonl(test_scores, test_scores_path)
    _write_json(manifest, os.path.join(score_dir, "manifest.json"))
    logger.info("  Scores written to %s", score_dir)
    return True


# ------------------------------------------------------------------ #
# Experiment (threshold distinguisher)
# ------------------------------------------------------------------ #

def run_experiment_for_domain(
    score_dir: str,
    run_dir: str,
    primary_score: str,
    criterion: str = "balanced_accuracy",
) -> dict:
    """Calibrate threshold and evaluate. Returns results dict."""
    results_path = os.path.join(run_dir, "results.json")
    if os.path.isfile(results_path):
        logger.info("  Experiment results already exist at %s — skipping.", run_dir)
        with open(results_path) as f:
            return json.load(f)

    train_path = os.path.join(score_dir, "train_scores.jsonl")
    test_path  = os.path.join(score_dir, "test_scores.jsonl")
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        logger.warning("  Missing score files at %s — cannot run experiment.", score_dir)
        return {}

    train_records = _load_jsonl(train_path)
    test_records  = _load_jsonl(test_path)

    train_labels = [r["label"] for r in train_records]
    test_labels  = [r["label"] for r in test_records]

    available_keys = [k for k in SCORE_KEYS if k in train_records[0]]
    main_results = []

    for key in available_keys:
        train_sc = [r[key] for r in train_records]
        test_sc  = [r[key] for r in test_records]
        res = run_distinguisher(
            train_labels, train_sc,
            test_labels, test_sc,
            score_name=key, criterion=criterion,
        )
        main_results.append(res)

    output = {
        "main_results": main_results,
        "shuffled_label_control": None,
        "parent_threshold_transfer": None,
    }
    os.makedirs(run_dir, exist_ok=True)
    _write_json(output, results_path)
    logger.info("  Experiment results written to %s", run_dir)

    # Print test-split summary
    logger.info("  %-28s %6s %6s %6s %6s", "Score", "Acc", "AUC", "Adv", "T@1%")
    for r in main_results:
        t = r["test"]
        logger.info("  %-28s %6.3f %6.3f %6.3f %6.3f",
                    r["score_name"],
                    t.get("accuracy", float("nan")),
                    t.get("auc") or float("nan"),
                    t.get("shard_advantage", float("nan")),
                    t.get("tpr_at_1_fpr") or float("nan"))
    return output


# ------------------------------------------------------------------ #
# Discovery
# ------------------------------------------------------------------ #

def discover_domains(processed_root: str) -> list:
    """Return list of (slug, domain_dir) for prepared domains."""
    domains = []
    if not os.path.isdir(processed_root):
        return domains
    for entry in sorted(os.listdir(processed_root)):
        d = os.path.join(processed_root, entry)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "train.jsonl")):
            domains.append((entry, d))
    return domains


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Score and evaluate both models on all prepared MIMIR domains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--processed-root", default="data/processed/mimir_domains")
    p.add_argument("--models",
                   default="EleutherAI/pythia-1.4b,nnheui/pythia-1.4b-sft-full")
    p.add_argument("--model-slugs",
                   default="parent_pythia_1_4b,target_nnheui_pythia_1_4b_sft_full",
                   dest="model_slugs")
    p.add_argument("--min-k-pcts", default="5,10,20,40")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--score-root", default="data/scores/mimir_domains")
    p.add_argument("--run-root",   default="outputs/runs/mimir_domains")
    p.add_argument("--primary-score", default="min_k_20_logprob")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--max-length", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    model_slugs = [s.strip() for s in args.model_slugs.split(",")]
    k_pcts = [int(k) for k in args.min_k_pcts.split(",")]

    assert len(model_names) == len(model_slugs), \
        "--models and --model-slugs must have same length"

    domains = discover_domains(args.processed_root)
    if not domains:
        logger.error("No prepared domains found in %s", args.processed_root)
        sys.exit(1)

    logger.info("=== MIMIR Multi-Domain Scoring + Experiment ===")
    logger.info("Domains found: %s", [d for d, _ in domains])
    logger.info("Models:        %s", model_names)

    device = get_device(args.device)

    # Score each model (load once, score all domains)
    for model_name, model_slug in zip(model_names, model_slugs):
        logger.info("\n" + "=" * 60)
        logger.info("Loading model: %s", model_name)
        logger.info("=" * 60)

        # Check if any domain still needs scoring
        needs_scoring = any(
            not (
                os.path.isfile(os.path.join(args.score_root, slug, model_slug, "train_scores.jsonl"))
                and os.path.isfile(os.path.join(args.score_root, slug, model_slug, "test_scores.jsonl"))
            )
            for slug, _ in domains
        )
        if not needs_scoring:
            logger.info("All domains already scored for %s — skipping model load.", model_slug)
        else:
            model, tokenizer = load_model_and_tokenizer(model_name, device, args.dtype)

            for slug, domain_dir in domains:
                score_dir = os.path.join(args.score_root, slug, model_slug)
                logger.info("\n[Scoring] domain=%s  model=%s", slug, model_slug)
                score_domain(
                    domain_dir=domain_dir,
                    score_dir=score_dir,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    model_name=model_name,
                    k_pcts=k_pcts,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )

            # Free GPU memory before loading next model
            import torch
            del model
            torch.cuda.empty_cache()
            logger.info("Model %s freed from GPU.", model_name)

    # Run experiments (CPU only, fast)
    logger.info("\n" + "=" * 60)
    logger.info("Running threshold experiments...")
    logger.info("=" * 60)

    for slug, domain_dir in domains:
        for model_slug in model_slugs:
            score_dir = os.path.join(args.score_root, slug, model_slug)
            run_dir   = os.path.join(args.run_root,   slug, model_slug)
            logger.info("\n[Experiment] domain=%s  model=%s", slug, model_slug)
            run_experiment_for_domain(
                score_dir=score_dir,
                run_dir=run_dir,
                primary_score=args.primary_score,
            )

    print("\n=== ALL DONE ===")
    print(f"Scores:  {args.score_root}")
    print(f"Results: {args.run_root}")


if __name__ == "__main__":
    main()

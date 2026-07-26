"""Pythia-6.9B WikiMIA length-32 membership experiment.

Orchestrates: data validation → scoring → threshold experiment → results.

Reuses existing utilities:
  src/shard_audit/logprobs.py
  src/shard_audit/mia_scores.py
  src/shard_audit/distinguishers.py

Usage:
    python scripts/experiments/run_wikimia_pythia69b_experiment.py \\
        --data-dir        data/processed/wikimia_length32 \\
        --score-dir       data/scores/wikimia_pythia69b_length32 \\
        --run-dir         outputs/runs/wikimia_pythia69b_length32 \\
        --model           EleutherAI/pythia-6.9b \\
        --dtype           float16 \\
        --batch-size      1 \\
        --primary-score   min_k_20_logprob
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

from src.shard_audit.scoring.logprobs import get_device, load_model_and_tokenizer, extract_token_logprobs
from src.shard_audit.scoring.mia_scores import compute_all_scores
from src.shard_audit.auditing.distinguishers import run_distinguisher
from src.shard_audit.auditing.metrics import score_diagnostics

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


def _write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------------------------------------------ #
# Step 1: Validate data
# ------------------------------------------------------------------ #

def validate_data(data_dir: str) -> dict:
    """Validate that a prepared WikiMIA directory is balanced and overlap-free."""
    train_path    = os.path.join(data_dir, "train.jsonl")
    test_path     = os.path.join(data_dir, "test.jsonl")
    manifest_path = os.path.join(data_dir, "manifest.json")

    for p, label in [(train_path, "train.jsonl"), (test_path, "test.jsonl"),
                     (manifest_path, "manifest.json")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing {label} in {data_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    train = _load_jsonl(train_path)
    test  = _load_jsonl(test_path)

    n1_tr = sum(1 for r in train if r["label"] == 1)
    n0_tr = sum(1 for r in train if r["label"] == 0)
    n1_te = sum(1 for r in test  if r["label"] == 1)
    n0_te = sum(1 for r in test  if r["label"] == 0)

    assert n1_tr == n0_tr, f"Train imbalance: {n1_tr} member vs {n0_tr} nonmember"
    assert n1_te == n0_te, f"Test imbalance: {n1_te} member vs {n0_te} nonmember"

    h_tr = {r["text_hash"] for r in train}
    h_te = {r["text_hash"] for r in test}
    overlap = len(h_tr & h_te)
    assert overlap == 0, f"Train/test overlap: {overlap} texts"

    logger.info("Data validated: train=%d (%d+%d), test=%d (%d+%d), overlap=%d",
                len(train), n1_tr, n0_tr, len(test), n1_te, n0_te, overlap)
    return manifest


# ------------------------------------------------------------------ #
# Step 2: Score
# ------------------------------------------------------------------ #

def score_model(
    data_dir: str,
    score_dir: str,
    model_name: str,
    device,
    dtype_str: str,
    k_pcts: list,
    batch_size: int,
    max_length: int,
) -> bool:
    """Score both splits. Returns True if scoring was done, False if skipped."""
    train_out = os.path.join(score_dir, "train_scores.jsonl")
    test_out  = os.path.join(score_dir, "test_scores.jsonl")

    if os.path.isfile(train_out) and os.path.isfile(test_out):
        logger.info("Scores already exist in %s — skipping scoring.", score_dir)
        return False

    logger.info("Loading model: %s", model_name)
    model, tokenizer = load_model_and_tokenizer(model_name, device, dtype_str)
    import torch
    dtype = next(model.parameters()).dtype
    logger.info("Model loaded: dtype=%s device=%s", dtype, device)

    def _score_split(records, split_name, out_path):
        results = _load_jsonl(out_path) if os.path.isfile(out_path) else []
        done_ids = {record["id"] for record in results}
        remaining = [record for record in records if record["id"] not in done_ids]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as handle:
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i + batch_size]
                texts = [r["text"] for r in batch]
                batch_lp = extract_token_logprobs(texts, model, tokenizer, device,
                                                  max_length=max_length)
                for rec, lp in zip(batch, batch_lp):
                    sc = compute_all_scores(rec["text"], lp, k_pcts=tuple(k_pcts))
                    score_rec = {
                        "id":                rec["id"],
                        "label":             rec["label"],
                        "phase_split":       rec.get("phase_split", split_name),
                        "text_hash":         rec["text_hash"],
                        "model":             model_name,
                        "num_input_tokens":  len(lp) + 1,
                        "num_scored_tokens": len(lp),
                        "mean_logprob":      round(sc["mean_logprob"], 6),
                        "mean_loss":         round(sc["mean_loss"], 6),
                        **{f"min_k_{k}_logprob": round(sc[f"min_k_{k}_logprob"], 6)
                           for k in k_pcts},
                        "zlib_norm_logprob": sc.get("zlib_norm_logprob"),
                    }
                    results.append(score_rec)
                    handle.write(json.dumps(score_rec, ensure_ascii=False) + "\n")
                handle.flush()
                if (i // batch_size + 1) % 20 == 0 or i + batch_size >= len(remaining):
                    logger.info("  Scored %d / %d %s examples...",
                                len(done_ids) + min(i + batch_size, len(remaining)), len(records), split_name)
        return results

    train_records = _load_jsonl(os.path.join(data_dir, "train.jsonl"))
    test_records  = _load_jsonl(os.path.join(data_dir, "test.jsonl"))

    logger.info("Scoring %d train examples...", len(train_records))
    train_scores = _score_split(train_records, "train", train_out)
    logger.info("Scoring %d test examples...", len(test_records))
    test_scores = _score_split(test_records, "test", test_out)

    # Diagnostics
    diag_train = score_diagnostics(train_scores, score_keys=tuple(SCORE_KEYS))
    diag_test  = score_diagnostics(test_scores,  score_keys=tuple(SCORE_KEYS))
    logger.info("=== Score Diagnostics (Train) ===")
    for k, d in diag_train.items():
        dir_ok = "OK" if d.get("direction_ok") else "INVERTED"
        logger.info("  %-28s AUC=%.4f dir=%s", k, d.get("auc", float("nan")), dir_ok)
    logger.info("=== Score Diagnostics (Test) ===")
    for k, d in diag_test.items():
        dir_ok = "OK" if d.get("direction_ok") else "INVERTED"
        logger.info("  %-28s AUC=%.4f dir=%s", k, d.get("auc", float("nan")), dir_ok)

    manifest = {
        "model":             model_name,
        "device":            str(device),
        "dtype":             str(dtype),
        "batch_size":        batch_size,
        "k_pcts":            k_pcts,
        "max_length":        max_length,
        "n_train_scored":    len(train_scores),
        "n_test_scored":     len(test_scores),
        "score_diagnostics_train": diag_train,
        "score_diagnostics_test":  diag_test,
        "timestamp":         datetime.utcnow().isoformat() + "Z",
    }
    _write_jsonl(train_scores, train_out)
    _write_jsonl(test_scores, test_out)
    _write_json(manifest, os.path.join(score_dir, "manifest.json"))
    logger.info("Scores written to %s", score_dir)

    del model
    torch.cuda.empty_cache()
    return True


# ------------------------------------------------------------------ #
# Step 3: Threshold experiment + shuffled-label control
# ------------------------------------------------------------------ #

def run_experiment(
    score_dir: str,
    run_dir: str,
    primary_score: str,
    criterion: str = "balanced_accuracy",
    shuffled_seed: int = 7,
) -> dict:
    """Run threshold distinguisher for each score. Returns results dict."""
    results_path = os.path.join(run_dir, "results.json")
    if os.path.isfile(results_path):
        logger.info("Experiment results already exist — skipping.")
        with open(results_path) as f:
            return json.load(f)

    train_records = _load_jsonl(os.path.join(score_dir, "train_scores.jsonl"))
    test_records  = _load_jsonl(os.path.join(score_dir, "test_scores.jsonl"))

    train_labels = [r["label"] for r in train_records]
    test_labels  = [r["label"] for r in test_records]

    available_keys = [k for k in SCORE_KEYS if k in train_records[0]]

    # Main results
    main_results = []
    for key in available_keys:
        train_sc = [r[key] for r in train_records]
        test_sc  = [r[key] for r in test_records]
        res = run_distinguisher(
            train_labels, train_sc, test_labels, test_sc,
            score_name=key, criterion=criterion,
        )
        main_results.append(res)

    # Shuffled-label control: shuffle train labels, re-calibrate, evaluate on real test
    logger.info("\n--- Shuffled-label control ---")
    rng = random.Random(shuffled_seed)
    shuffled_train_labels = list(train_labels)
    rng.shuffle(shuffled_train_labels)
    shuffled_results = []
    for key in available_keys:
        train_sc = [r[key] for r in train_records]
        test_sc  = [r[key] for r in test_records]
        res = run_distinguisher(
            shuffled_train_labels, train_sc, test_labels, test_sc,
            score_name=key, criterion=criterion,
        )
        shuffled_results.append(res)

    output = {
        "main_results":          main_results,
        "shuffled_label_control": shuffled_results,
        "primary_score":         primary_score,
        "timestamp":             datetime.utcnow().isoformat() + "Z",
    }

    os.makedirs(run_dir, exist_ok=True)
    _write_json(output, results_path)
    logger.info("Experiment results written to %s", results_path)

    logger.info("\n=== Test-Split Results ===")
    logger.info("%-28s %6s %6s %6s %6s %6s",
                "Score", "Acc", "BalAcc", "AUC", "T@1FP", "Adv")
    for r in main_results:
        t = r["test"]
        logger.info("%-28s %6.3f %6.3f %6.3f %6s %6.3f",
                    r["score_name"],
                    t.get("accuracy", float("nan")),
                    t.get("balanced_accuracy", float("nan")),
                    t.get("auc") or float("nan"),
                    f"{t.get('tpr_at_1_fpr') or float('nan'):.3f}",
                    t.get("shard_advantage", float("nan")))

    return output


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Pythia-6.9B WikiMIA shard-membership experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",       default="data/processed/wikimia_length32",
                   help="Directory with prepared WikiMIA train.jsonl / test.jsonl")
    p.add_argument("--score-dir",      default="data/scores/wikimia_pythia69b_length32")
    p.add_argument("--run-dir",        default="outputs/runs/wikimia_pythia69b_length32")
    p.add_argument("--model",          default="EleutherAI/pythia-6.9b")
    p.add_argument("--dtype",          default="float16")
    p.add_argument("--batch-size",     type=int, default=1)
    p.add_argument("--max-length",     type=int, default=128)
    p.add_argument("--min-k-pcts",     default="5,10,20,40")
    p.add_argument("--primary-score",  default="min_k_20_logprob")
    p.add_argument("--device",         default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    k_pcts = [int(k) for k in args.min_k_pcts.split(",")]
    device = get_device(args.device)

    logger.info("=== Pythia-6.9B WikiMIA Experiment ===")
    logger.info("Data dir:   %s", args.data_dir)
    logger.info("Score dir:  %s", args.score_dir)
    logger.info("Run dir:    %s", args.run_dir)
    logger.info("Model:      %s", args.model)
    logger.info("Device:     %s", device)
    logger.info("Dtype:      %s", args.dtype)

    # Step 1
    logger.info("\n[1/3] Validating data...")
    manifest = validate_data(args.data_dir)
    logger.info("WikiMIA split: %s  train=%d  test=%d  seed=%d",
                manifest.get("wikimia_split"),
                manifest.get("n_train"), manifest.get("n_test"),
                manifest.get("seed"))

    # Step 2
    logger.info("\n[2/3] Scoring %s...", args.model)
    score_model(
        data_dir=args.data_dir,
        score_dir=args.score_dir,
        model_name=args.model,
        device=device,
        dtype_str=args.dtype,
        k_pcts=k_pcts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Step 3
    logger.info("\n[3/3] Running threshold experiment...")
    run_experiment(
        score_dir=args.score_dir,
        run_dir=args.run_dir,
        primary_score=args.primary_score,
    )

    print(f"\n=== ALL DONE ===")
    print(f"Scores:  {args.score_dir}")
    print(f"Results: {args.run_dir}")
    print(f"\nNext step:")
    print(f"  python experiments/table_04_wikimia_score_selection/reports/pythia_6_9b.py \\")
    print(f"    --results-file   {args.run_dir}/results.json \\")
    print(f"    --test-scores    {args.score_dir}/test_scores.jsonl \\")
    print(f"    --data-manifest  {args.data_dir}/manifest.json \\")
    print(f"    --output-dir     outputs/reports/wikimia_pythia69b_length32")


if __name__ == "__main__":
    main()

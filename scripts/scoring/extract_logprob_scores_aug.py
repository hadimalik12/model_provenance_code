"""Extract 4-view augmented per-token log-probability scores from a causal LM for MIA.

Loads train.jsonl and test.jsonl produced by prepare_mimir_github.py,
generates 4 code-augmented text views per example, scores each view with the model,
computes min_k_20_logprob for each view, and averages them into `aug_min_k_20_logprob_mean`.
"""

import argparse
import json
import logging
import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.logprobs import (
    get_device,
    load_model_and_tokenizer,
    extract_token_logprobs,
)
from src.shard_audit.mia_scores import compute_all_scores
from src.shard_audit.text_augmentations import augment_code_snippet_4views

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


def _write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def score_records_aug(
    records: list,
    model,
    tokenizer,
    device,
    k_pcts: list,
    model_name: str,
    out_path: str,
    n_aug: int = 4,
    max_length: int = 512,
    max_examples: int = None,
) -> list:
    """Score records with K=4 code augmentations and incrementally save."""
    if max_examples is not None and max_examples > 0:
        records = records[:max_examples]

    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done_ids.add(json.loads(line)["id"])
                    
    remaining = [r for r in records if r["id"] not in done_ids]
    if not remaining:
        logger.info("All %d records already scored in %s", len(records), out_path)
        return []

    logger.info("Found %d already scored, %d remaining to score.", len(done_ids), len(remaining))
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    f_out = open(out_path, "a", encoding="utf-8")

    score_results = []

    for idx, rec in enumerate(remaining):
        text = rec["text"]
        
        # 1. Generate K=4 code-augmented text views
        views = augment_code_snippet_4views(text, n_aug=n_aug)
        
        view_min_k_20_scores = []
        view_mean_logprobs = []

        # 2. Score each view
        for view_txt in views:
            batch_lp = extract_token_logprobs([view_txt], model, tokenizer, device, max_length=max_length)
            lp = batch_lp[0]
            scores = compute_all_scores(view_txt, lp, k_pcts=tuple(k_pcts))
            view_min_k_20_scores.append(scores["min_k_20_logprob"])
            view_mean_logprobs.append(scores["mean_logprob"])

        # 3. Compute arithmetic mean across views
        aug_min_k_20_mean = float(np.mean(view_min_k_20_scores))
        aug_mean_logprob_mean = float(np.mean(view_mean_logprobs))

        score_rec = {
            "id": rec["id"],
            "label": rec["label"],
            "phase_split": rec.get("phase_split", ""),
            "text_hash": rec["text_hash"],
            "model": model_name,
            "min_k_20_logprob": round(view_min_k_20_scores[0], 6),  # Original un-augmented score
            "aug_min_k_20_logprob_mean": round(aug_min_k_20_mean, 6), # 4-view augmented score
            "aug_mean_logprob_mean": round(aug_mean_logprob_mean, 6),
        }
        score_results.append(score_rec)
        
        f_out.write(json.dumps(score_rec, ensure_ascii=False) + "\n")
        f_out.flush()

        if (idx + 1) % 10 == 0 or (idx + 1) == len(remaining):
            logger.info("Scored %d/%d remaining records...", idx + 1, len(remaining))

    f_out.close()
    return score_results


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract 4-view augmented per-token log-probability scores from causal LM."
    )
    p.add_argument("--model", default="EleutherAI/pythia-1.4b")
    p.add_argument("--train-file", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-k-pcts", default="5,10,20,40")
    p.add_argument("--n-aug", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    k_pcts = [int(x) for x in args.min_k_pcts.split(",") if x.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    logger.info("Using device: %s", device)
    try:
        model, tokenizer = load_model_and_tokenizer(args.model, device=device, dtype_str=args.dtype)
    except Exception as err:
        logger.error("Failed to load model %s: %s", args.model, err)
        sys.exit(0)

    try:
        train_records = _load_jsonl(args.train_file)
        test_records = _load_jsonl(args.test_file)

        logger.info("=== Scoring Train Records ===")
        train_out = os.path.join(args.output_dir, "train_scores.jsonl")
        score_records_aug(
            train_records, model, tokenizer, device, k_pcts, args.model,
            out_path=train_out,
            n_aug=args.n_aug, max_length=args.max_length, max_examples=args.max_examples
        )

        logger.info("=== Scoring Test Records ===")
        test_out = os.path.join(args.output_dir, "test_scores.jsonl")
        score_records_aug(
            test_records, model, tokenizer, device, k_pcts, args.model,
            out_path=test_out,
            n_aug=args.n_aug, max_length=args.max_length, max_examples=args.max_examples
        )

        logger.info("Augmented scoring complete. Output saved to %s", args.output_dir)
    except Exception as err:
        logger.error("Error during scoring for model %s: %s", args.model, err)
        sys.exit(0)


if __name__ == "__main__":
    main()

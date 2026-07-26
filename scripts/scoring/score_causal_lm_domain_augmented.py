"""Extract domain-augmented per-token log-probability scores from a causal LM.

Like ``score_causal_lm_code_augmented.py`` but uses domain-specific augmentations from
``text_augmentations_domains.py`` rather than the GitHub-only code augmentations.

Each example is scored in 4 augmented views and the MIN-K% 20 scores are averaged
into ``aug_min_k_20_logprob_mean``.  The vanilla (un-augmented) ``min_k_20_logprob``
is also recorded.

Usage:
    python scripts/scoring/score_causal_lm_domain_augmented.py \\
        --model EleutherAI/pythia-1.4b \\
        --train-file data/processed/mimir_domains/arxiv/train.jsonl \\
        --test-file  data/processed/mimir_domains/arxiv/test.jsonl \\
        --output-dir data/scores/mimir_domains_aug/arxiv/parent_pythia_1_4b \\
        --domain arxiv \\
        --min-k-pcts 5,10,20,40 \\
        --n-aug 4 \\
        --dtype float16
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

from src.shard_audit.scoring.logprobs import (
    extract_token_logprobs_with_output_head,
    get_device,
    load_model_and_tokenizer,
    load_output_head,
)
from src.shard_audit.scoring.mia_scores import compute_all_scores
from src.shard_audit.scoring.text_augmentations_domains import augment_by_domain

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


def score_records_domain_aug(
    records: list,
    model,
    tokenizer,
    device,
    k_pcts: list,
    parent_model_name: str,
    target_model_name: str,
    parent_output_head,
    domain: str,
    out_path: str,
    n_aug: int = 4,
    max_length: int = 512,
    max_examples: int = None,
) -> list:
    """Score records with domain-specific augmentations and incrementally save."""
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

        # 1. Generate K augmented text views using domain-specific strategy
        try:
            views = augment_by_domain(text, domain, n_aug=n_aug)
        except ValueError as e:
            logger.error("Augmentation error: %s", e)
            views = [text] * n_aug  # fallback to original

        view_min_k_scores = {k: [] for k in k_pcts}
        view_mean_logprobs = []

        # 2. Score each view
        for view_txt in views:
            batch_lp = extract_token_logprobs_with_output_head(
                [view_txt], model, parent_output_head, tokenizer, device, max_length=max_length
            )
            lp = batch_lp[0]
            scores = compute_all_scores(view_txt, lp, k_pcts=tuple(k_pcts))
            view_mean_logprobs.append(scores["mean_logprob"])
            for k in k_pcts:
                view_min_k_scores[k].append(scores[f"min_k_{k}_logprob"])

        # 3. Also score the original (un-augmented) text
        orig_lp = extract_token_logprobs_with_output_head(
            [text], model, parent_output_head, tokenizer, device, max_length=max_length
        )[0]
        orig_scores = compute_all_scores(text, orig_lp, k_pcts=tuple(k_pcts))

        # 4. Build output record
        score_rec = {
            "id": rec["id"],
            "label": rec["label"],
            "phase_split": rec.get("phase_split", ""),
            "text_hash": rec["text_hash"],
            "parent_model": parent_model_name,
            "target_model": target_model_name,
            "scoring_head": "parent_output_head",
            "domain": domain,
            # Vanilla (un-augmented) scores
            "mean_logprob": round(orig_scores["mean_logprob"], 6),
            "mean_loss": round(orig_scores["mean_loss"], 6),
        }
        for k in k_pcts:
            score_rec[f"min_k_{k}_logprob"] = round(
                orig_scores[f"min_k_{k}_logprob"], 6
            )

        # Augmented average scores
        score_rec["aug_mean_logprob_mean"] = round(
            float(np.mean(view_mean_logprobs)), 6
        )
        for k in k_pcts:
            score_rec[f"aug_min_k_{k}_logprob_mean"] = round(
                float(np.mean(view_min_k_scores[k])), 6
            )

        score_results.append(score_rec)

        f_out.write(json.dumps(score_rec, ensure_ascii=False) + "\n")
        f_out.flush()

        if (idx + 1) % 10 == 0 or (idx + 1) == len(remaining):
            logger.info("Scored %d/%d remaining records...", idx + 1, len(remaining))

    f_out.close()
    return score_results


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract domain-augmented log-probability scores from causal LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--parent-model",
                   help="Checkpoint supplying the causal-LM output head.")
    p.add_argument("--target-model",
                   help="Checkpoint supplying transformer hidden states.")
    p.add_argument("--model", help=argparse.SUPPRESS)
    p.add_argument("--train-file", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--domain", required=True,
                   help="MIMIR domain name (github, arxiv, dm_mathematics, "
                        "wikipedia_en, pile_cc)")
    p.add_argument("--min-k-pcts", default="5,10,20,40")
    p.add_argument("--n-aug", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Not used for augmented scoring (always batch=1)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.model:
        if args.parent_model or args.target_model:
            raise ValueError("Use either --model or --parent-model/--target-model, not both.")
        args.parent_model = args.model
        args.target_model = args.model
    if not args.parent_model or not args.target_model:
        raise ValueError("Both --parent-model and --target-model are required.")
    k_pcts = [int(x) for x in args.min_k_pcts.split(",") if x.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    logger.info("Using device: %s", device)
    logger.info("Domain: %s", args.domain)
    logger.info("Parent output head: %s", args.parent_model)
    logger.info("Target body: %s", args.target_model)

    try:
        model, tokenizer = load_model_and_tokenizer(
            args.target_model, device=device, dtype_str=args.dtype
        )
        parent_model, parent_output_head = load_output_head(
            args.parent_model, device=device, dtype_str=args.dtype
        )
    except Exception as err:
        logger.error("Failed to load parent/target models: %s", err)
        sys.exit(1)

    try:
        train_records = _load_jsonl(args.train_file)
        test_records = _load_jsonl(args.test_file)

        logger.info("=== Scoring Train Records (domain=%s) ===", args.domain)
        train_out = os.path.join(args.output_dir, "train_scores.jsonl")
        score_records_domain_aug(
            train_records, model, tokenizer, device, k_pcts,
            args.parent_model, args.target_model, parent_output_head,
            domain=args.domain, out_path=train_out, n_aug=args.n_aug, max_length=args.max_length,
            max_examples=args.max_examples,
        )

        logger.info("=== Scoring Test Records (domain=%s) ===", args.domain)
        test_out = os.path.join(args.output_dir, "test_scores.jsonl")
        score_records_domain_aug(
            test_records, model, tokenizer, device, k_pcts,
            args.parent_model, args.target_model, parent_output_head,
            domain=args.domain, out_path=test_out, n_aug=args.n_aug, max_length=args.max_length,
            max_examples=args.max_examples,
        )

        logger.info("Domain-augmented scoring complete. Output: %s", args.output_dir)
    except Exception as err:
        logger.error("Error during parent-head scoring: %s", err)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

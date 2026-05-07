"""Prepare MIMIR ArXiv member/nonmember dataset for shard-membership auditing.

Dataset: iamgroot42/mimir  (config=arxiv, split=ngram_13_0.2 by default)

MIMIR file layout:
  cache_100_200_1000_512/train/arxiv_<split>.jsonl  → MEMBER texts
  cache_100_200_1000_512/test/arxiv_<split>.jsonl   → NONMEMBER texts

Usage:
    python scripts/data/prepare_mimir_arxiv.py \\
        --num-train-per-class 500 --num-test-per-class 200 \\
        --output-dir data/processed/mimir_arxiv
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.datasets import (
    DATASET_ID,
    KNOWN_SPLITS,
    list_configs,
    load_mimir_github, # Note: this function name is generic to any config if we pass config='arxiv'
    texts_to_records,
    _member_filename,
    _nonmember_filename,
)
from src.shard_audit.preprocessing import preprocess_text
from src.shard_audit.splitting import stratified_train_test_split
from src.shard_audit.sanity_checks import (
    check_label_balance,
    check_word_counts,
    check_required_fields,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _word_count_histogram(texts: list, bins: list = None) -> dict:
    if bins is None:
        bins = [0, 8, 16, 24, 32, 48, 64, 128, 256, 999999]
    counts = Counter()
    for text in texts:
        wc = len(text.split())
        for lo, hi in zip(bins, bins[1:]):
            if lo <= wc < hi:
                counts[f"{lo}-{hi-1}"] += 1
                break
    return dict(counts)


def _exact_overlap(hashes_a: set, hashes_b: set) -> int:
    return len(hashes_a & hashes_b)


def _write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare MIMIR ArXiv member/nonmember dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-id", default=DATASET_ID)
    p.add_argument("--config", default="arxiv",
                   help="Dataset config/subset (e.g. 'arxiv')")
    p.add_argument("--split", default="ngram_13_0.2",
                   help="N-gram split name (e.g. 'ngram_13_0.2')")
    p.add_argument("--num-train-per-class", type=int, default=500)
    p.add_argument("--num-test-per-class", type=int, default=200)
    p.add_argument("--max-words", type=int, default=32)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="data/processed/mimir_arxiv")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace API token (default: $HF_TOKEN)")
    p.add_argument("--allow-overlap", action="store_true",
                   help="Allow exact text overlap between member and nonmember")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=== MIMIR ArXiv Dataset Preparation ===")
    logger.info("Dataset:       %s", args.dataset_id)
    logger.info("Config:        %s", args.config)
    logger.info("Split:         %s", args.split)
    logger.info("Train/class:   %d", args.num_train_per_class)
    logger.info("Test/class:    %d", args.num_test_per_class)
    logger.info("max_words:     %d", args.max_words)
    logger.info("min_words:     %d", args.min_words)
    logger.info("Seed:          %d", args.seed)
    logger.info("Output dir:    %s", args.output_dir)

    # 1. Load raw texts
    logger.info("\n[1/5] Loading MIMIR ArXiv texts via hub download...")
    member_texts_raw, nonmember_texts_raw = load_mimir_github(
        config=args.config,
        split=args.split,
        token=args.token,
    )
    logger.info("Raw loaded: %d members, %d nonmembers", len(member_texts_raw), len(nonmember_texts_raw))

    n_member_raw = len(member_texts_raw)
    n_nonmember_raw = len(nonmember_texts_raw)

    wc_member_before = _word_count_histogram(member_texts_raw)
    wc_nonmember_before = _word_count_histogram(nonmember_texts_raw)

    # 2. Preprocess + deduplicate
    logger.info("\n[2/5] Preprocessing (min=%d, max=%d words)...", args.min_words, args.max_words)

    def _preprocess_list(texts):
        import hashlib
        kept, filtered, deduped = [], 0, 0
        seen_hashes = set()
        for t in texts:
            result = preprocess_text(t, max_words=args.max_words, min_words=args.min_words)
            if result is None:
                filtered += 1
                continue
            h = hashlib.sha256(result.encode()).hexdigest()
            if h in seen_hashes:
                deduped += 1
                continue
            seen_hashes.add(h)
            kept.append(result)
        return kept, filtered, deduped

    member_texts_clean, n_member_filtered, n_member_deduped = _preprocess_list(member_texts_raw)
    nonmember_texts_clean, n_nonmember_filtered, n_nonmember_deduped = _preprocess_list(nonmember_texts_raw)
    
    logger.info("Members:    %d kept, %d too-short, %d intra-class duplicates removed",
                len(member_texts_clean), n_member_filtered, n_member_deduped)
    logger.info("Nonmembers: %d kept, %d too-short, %d intra-class duplicates removed",
                len(nonmember_texts_clean), n_nonmember_filtered, n_nonmember_deduped)

    wc_member_after = _word_count_histogram(member_texts_clean)
    wc_nonmember_after = _word_count_histogram(nonmember_texts_clean)

    # 3. Build records and check overlap
    logger.info("\n[3/5] Building records and checking overlaps...")
    member_records = texts_to_records(
        member_texts_clean, label=1,
        source="mimir_arxiv", split_origin="member",
        id_prefix="member-",
    )
    nonmember_records = texts_to_records(
        nonmember_texts_clean, label=0,
        source="mimir_arxiv", split_origin="nonmember",
        id_prefix="nonmember-",
    )

    member_hashes = {r["text_hash"] for r in member_records}
    nonmember_hashes = {r["text_hash"] for r in nonmember_records}
    overlap_count = _exact_overlap(member_hashes, nonmember_hashes)

    if overlap_count > 0:
        msg = f"Exact text overlap: {overlap_count} texts appear in both member and nonmember sets."
        if args.allow_overlap:
            logger.warning(msg + " Continuing because --allow-overlap is set.")
        else:
            logger.error(msg + " Use --allow-overlap to proceed anyway.")
            sys.exit(1)
    else:
        logger.info("No exact text overlap between member and nonmember sets.")

    # 4. Split
    logger.info("\n[4/5] Creating MIA train/test splits...")
    total_needed = args.num_train_per_class + args.num_test_per_class
    for cls_name, records in (("member", member_records), ("nonmember", nonmember_records)):
        if len(records) < total_needed:
            logger.error("Not enough %s records: need %d, have %d.", cls_name, total_needed, len(records))
            sys.exit(1)

    train_records, test_records = stratified_train_test_split(
        member_records=member_records,
        nonmember_records=nonmember_records,
        num_train_per_class=args.num_train_per_class,
        num_test_per_class=args.num_test_per_class,
        seed=args.seed,
    )

    logger.info("MIA train: %d total, MIA test: %d total", len(train_records), len(test_records))

    # 5. Write outputs
    logger.info("\n[5/5] Writing outputs to %s...", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    _write_jsonl(train_records, os.path.join(args.output_dir, "train.jsonl"))
    _write_jsonl(test_records, os.path.join(args.output_dir, "test.jsonl"))

    manifest = {
        "dataset_id": args.dataset_id,
        "config": args.config,
        "split": args.split,
        "max_words": args.max_words,
        "min_words": args.min_words,
        "num_train_per_class": args.num_train_per_class,
        "num_test_per_class": args.num_test_per_class,
        "seed": args.seed,
        "n_member_raw": n_member_raw,
        "n_nonmember_raw": n_nonmember_raw,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "output_dir": os.path.abspath(args.output_dir),
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    main()

"""Prepare WikiMIA dataset for shard-membership auditing.

Dataset: swj0419/WikiMIA (public)
Splits:  WikiMIA_length32, WikiMIA_length64, WikiMIA_length128, WikiMIA_length256

Label convention (WikiMIA):
  label = 1: seen/member during pretraining
  label = 0: unseen/nonmember during pretraining
  Text column: 'input'

Output layout:
    <output_dir>/train.jsonl
    <output_dir>/test.jsonl
    <output_dir>/manifest.json
    <output_dir>/diagnostics.json

Size selection rule (maximize held-out test size):
  Priority 1: preferred_train + preferred_test (e.g. 325 + 500)
  Priority 2: min_train    + preferred_test (e.g. 200 + 500)
  Priority 3: min_train    + min_test       (e.g. 200 + 200)
  Fallback:   (n_available - min_test) train + min_test test
              (train may dip below min_train when pool is small)

Usage:
    # Inspect all splits (counts, feasibility):
    python scripts/data/prepare_wikimia.py --inspect-all

    # Prepare length-32 split (recommended for OPT-6.7B):
    python scripts/data/prepare_wikimia.py \\
        --wikimia-split WikiMIA_length32 \\
        --preferred-train-per-class 325 \\
        --preferred-test-per-class 500 \\
        --min-train-per-class 200 \\
        --min-test-per-class 200 \\
        --seed 0 \\
        --output-dir data/processed/wikimia_length32
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.wikimia import (
    DATASET_ID,
    KNOWN_SPLITS,
    load_wikimia_split,
    inspect_all_splits,
)
from src.shard_audit.preprocessing import normalize_text, count_words
from src.shard_audit.sanity_checks import check_label_balance, check_required_fields

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_count_histogram(texts: list) -> dict:
    bins = [0, 8, 16, 24, 32, 48, 64, 128, 256, 512, 999999]
    counts = Counter()
    for text in texts:
        wc = len(text.split())
        for lo, hi in zip(bins, bins[1:]):
            if lo <= wc < hi:
                counts[f"{lo}-{hi-1}"] += 1
                break
    return dict(counts)


def _preprocess_records(
    raw_records: list,
    min_words: int,
) -> tuple:
    """Normalize whitespace, filter by min_words, deduplicate within class.

    No truncation — WikiMIA is already length-partitioned.
    Returns (kept_records, n_filtered, n_deduped).
    """
    kept, n_filtered, n_deduped = [], 0, 0
    seen = set()
    for r in raw_records:
        text = normalize_text(r["text"])
        if count_words(text) < min_words:
            n_filtered += 1
            continue
        h = _text_hash(text)
        if h in seen:
            n_deduped += 1
            continue
        seen.add(h)
        kept.append({**r, "text": text, "text_hash": h})
    return kept, n_filtered, n_deduped


def _select_sizes(
    n_available: int,
    preferred_train: int,
    preferred_test: int,
    min_train: int,
    min_test: int,
) -> tuple:
    """Return (n_train, n_test) using the priority-maximization rule.

    Returns (None, None) if n_available < min_test + 1.
    """
    # Priority 1: preferred sizes
    if n_available >= preferred_train + preferred_test:
        return preferred_train, preferred_test
    # Priority 2: reduce train to min, keep preferred test
    if n_available >= min_train + preferred_test:
        return min_train, preferred_test
    # Priority 3: both at min
    if n_available >= min_train + min_test:
        return min_train, min_test
    # Fallback: reserve min_test, give remainder to train
    n_train = n_available - min_test
    if n_train >= 1:
        return n_train, min_test
    return None, None


def _stratified_split(
    member_records: list,
    nonmember_records: list,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple:
    """Deterministic stratified train/test split. Returns (train, test)."""
    rng = random.Random(seed)

    def _split_one(records, n_tr, n_te):
        shuffled = list(records)
        rng.shuffle(shuffled)
        return shuffled[:n_tr], shuffled[n_tr:n_tr + n_te]

    m_train, m_test = _split_one(member_records, n_train, n_test)
    nm_train, nm_test = _split_one(nonmember_records, n_train, n_test)

    train_hashes = {r["text_hash"] for r in m_train + nm_train}
    test_hashes  = {r["text_hash"] for r in m_test + nm_test}
    overlap = train_hashes & test_hashes
    if overlap:
        raise ValueError(
            f"Hash overlap between MIA train and test: {len(overlap)} texts. "
            "Inspect source data for near-duplicates."
        )

    def _tag(records, phase):
        return [{**r, "phase_split": phase} for r in records]

    train_all = _tag(m_train, "train") + _tag(nm_train, "train")
    test_all  = _tag(m_test,  "test")  + _tag(nm_test,  "test")

    rng.shuffle(train_all)
    rng.shuffle(test_all)
    return train_all, test_all


def _write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def _enrich_records(records: list, split_name: str, source: str) -> list:
    """Add id, source, split_origin, wikimia_split fields."""
    enriched = []
    for i, r in enumerate(records):
        label = r["label"]
        split_origin = "member" if label == 1 else "nonmember"
        enriched.append({
            "id": f"{split_origin}-{i:06d}",
            "text": r["text"],
            "label": label,
            "source": source,
            "split_origin": split_origin,
            "wikimia_split": split_name,
            "text_hash": r["text_hash"],
        })
    return enriched


# ------------------------------------------------------------------ #
# Commands
# ------------------------------------------------------------------ #

def cmd_inspect_all(token):
    print(f"\nDataset: {DATASET_ID}")
    print(f"{'Split':<24} {'N':>6} {'Label0':>7} {'Label1':>7} {'MinCls':>7} "
          f"{'WC_min':>7} {'WC_max':>7} {'WC_mean':>8} {'500+500':>8} {'200+200':>8}")
    print("-" * 90)
    results = inspect_all_splits(token=token)
    for r in results:
        if "error" in r:
            print(f"{r['split']:<24}  ERROR: {r['error']}")
            continue
        f500 = "YES" if r["feasible_500_500_test"] else "no"
        f200 = "YES" if r["feasible_200_200_test"] else "no"
        print(f"{r['split']:<24} {r['n']:>6} {r['n_label0']:>7} {r['n_label1']:>7} "
              f"{r['min_per_class']:>7} {r['wc_min']:>7} {r['wc_max']:>7} "
              f"{r['wc_mean']:>8.1f} {f500:>8} {f200:>8}")
    print()


def cmd_prepare(args):
    logger.info("=== WikiMIA Data Preparation ===")
    logger.info("Dataset:               %s", DATASET_ID)
    logger.info("Split:                 %s", args.wikimia_split)
    logger.info("Preferred train/class: %d", args.preferred_train_per_class)
    logger.info("Preferred test/class:  %d", args.preferred_test_per_class)
    logger.info("Min train/class:       %d", args.min_train_per_class)
    logger.info("Min test/class:        %d", args.min_test_per_class)
    logger.info("Min words:             %d", args.min_words)
    logger.info("Seed:                  %d", args.seed)
    logger.info("Output dir:            %s", args.output_dir)

    # Skip if already done
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        logger.info("Output already exists at %s — skipping (delete to re-run).",
                    args.output_dir)
        return

    # 1. Load
    logger.info("\n[1/4] Loading WikiMIA split: %s ...", args.wikimia_split)
    raw = load_wikimia_split(args.wikimia_split, token=args.token)
    n_raw = len(raw)
    n_raw_member    = sum(1 for r in raw if r["label"] == 1)
    n_raw_nonmember = sum(1 for r in raw if r["label"] == 0)
    logger.info("Loaded %d raw records: %d member, %d nonmember",
                n_raw, n_raw_member, n_raw_nonmember)

    member_raw    = [r for r in raw if r["label"] == 1]
    nonmember_raw = [r for r in raw if r["label"] == 0]

    wc_member_before    = _word_count_histogram([r["text"] for r in member_raw])
    wc_nonmember_before = _word_count_histogram([r["text"] for r in nonmember_raw])

    # 2. Preprocess
    logger.info("\n[2/4] Preprocessing (normalize whitespace, filter min_words=%d)...",
                args.min_words)
    m_clean, m_filt, m_dedup = _preprocess_records(member_raw, args.min_words)
    nm_clean, nm_filt, nm_dedup = _preprocess_records(nonmember_raw, args.min_words)
    logger.info("Members:    %d kept, %d too-short, %d intra-class duplicates",
                len(m_clean), m_filt, m_dedup)
    logger.info("Nonmembers: %d kept, %d too-short, %d intra-class duplicates",
                len(nm_clean), nm_filt, nm_dedup)

    wc_member_after    = _word_count_histogram([r["text"] for r in m_clean])
    wc_nonmember_after = _word_count_histogram([r["text"] for r in nm_clean])

    # Cross-class overlap check
    m_hashes  = {r["text_hash"] for r in m_clean}
    nm_hashes = {r["text_hash"] for r in nm_clean}
    cross_overlap = len(m_hashes & nm_hashes)
    if cross_overlap:
        logger.warning("Cross-class overlap: %d texts appear in both member and nonmember.",
                       cross_overlap)
    else:
        logger.info("No cross-class text overlap.")

    # 3. Size selection
    n_pool = min(len(m_clean), len(nm_clean))
    n_train, n_test = _select_sizes(
        n_available=n_pool,
        preferred_train=args.preferred_train_per_class,
        preferred_test=args.preferred_test_per_class,
        min_train=args.min_train_per_class,
        min_test=args.min_test_per_class,
    )
    if n_train is None:
        logger.error(
            "Pool too small: %d per class, need at least min_test=%d + 1.",
            n_pool, args.min_test_per_class,
        )
        sys.exit(1)

    size_fallback = not (
        n_train == args.preferred_train_per_class
        and n_test == args.preferred_test_per_class
    )
    if size_fallback:
        logger.warning(
            "Using fallback sizes: %d train + %d test per class "
            "(preferred was %d + %d, pool=%d).",
            n_train, n_test,
            args.preferred_train_per_class, args.preferred_test_per_class,
            n_pool,
        )
    else:
        logger.info("Using preferred sizes: %d train + %d test per class.", n_train, n_test)

    # Enrich records with id, source, split_origin, wikimia_split
    source = f"wikimia_{args.wikimia_split}"
    m_records  = _enrich_records(m_clean,  args.wikimia_split, source)
    nm_records = _enrich_records(nm_clean, args.wikimia_split, source)

    # 4. Split
    logger.info("\n[3/4] Creating MIA train/test splits...")
    try:
        train_records, test_records = _stratified_split(
            member_records=m_records,
            nonmember_records=nm_records,
            n_train=n_train,
            n_test=n_test,
            seed=args.seed,
        )
    except ValueError as e:
        logger.error("Split failed: %s", e)
        sys.exit(1)

    n_train_m  = sum(1 for r in train_records if r["label"] == 1)
    n_train_nm = sum(1 for r in train_records if r["label"] == 0)
    n_test_m   = sum(1 for r in test_records  if r["label"] == 1)
    n_test_nm  = sum(1 for r in test_records  if r["label"] == 0)
    logger.info("MIA train: %d total (%d member, %d nonmember)", len(train_records), n_train_m, n_train_nm)
    logger.info("MIA test:  %d total (%d member, %d nonmember)", len(test_records), n_test_m, n_test_nm)

    check_label_balance(train_records, name="train")
    check_label_balance(test_records, name="test")
    check_required_fields(train_records)
    check_required_fields(test_records)
    logger.info("Sanity checks passed.")

    # 5. Write
    logger.info("\n[4/4] Writing outputs to %s ...", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    _write_jsonl(train_records, os.path.join(args.output_dir, "train.jsonl"))
    _write_jsonl(test_records, os.path.join(args.output_dir, "test.jsonl"))

    mia_train_hashes = {r["text_hash"] for r in train_records}
    mia_test_hashes  = {r["text_hash"] for r in test_records}
    tt_overlap = len(mia_train_hashes & mia_test_hashes)

    manifest = {
        "dataset_id": DATASET_ID,
        "wikimia_split": args.wikimia_split,
        "text_column": "input",
        "label_column": "label",
        "label_1_meaning": "member (seen during pretraining)",
        "label_0_meaning": "nonmember (not seen during pretraining)",
        "min_words": args.min_words,
        "no_truncation": True,
        "preferred_train_per_class": args.preferred_train_per_class,
        "preferred_test_per_class": args.preferred_test_per_class,
        "min_train_per_class": args.min_train_per_class,
        "min_test_per_class": args.min_test_per_class,
        "n_train_per_class": n_train,
        "n_test_per_class": n_test,
        "size_fallback": size_fallback,
        "seed": args.seed,
        "n_raw": n_raw,
        "n_raw_member": n_raw_member,
        "n_raw_nonmember": n_raw_nonmember,
        "n_member_after_preprocessing": len(m_clean),
        "n_nonmember_after_preprocessing": len(nm_clean),
        "n_train": len(train_records),
        "n_test": len(test_records),
        "n_train_member": n_train_m,
        "n_train_nonmember": n_train_nm,
        "n_test_member": n_test_m,
        "n_test_nonmember": n_test_nm,
        "cross_class_overlap": cross_overlap,
        "mia_train_test_overlap": tt_overlap,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "output_dir": os.path.abspath(args.output_dir),
    }

    diagnostics = {
        "word_count_histogram": {
            "member_before": wc_member_before,
            "nonmember_before": wc_nonmember_before,
            "member_after": wc_member_after,
            "nonmember_after": wc_nonmember_after,
        },
        "filtering": {
            "member_filtered_too_short": m_filt,
            "member_deduped": m_dedup,
            "nonmember_filtered_too_short": nm_filt,
            "nonmember_deduped": nm_dedup,
        },
        "overlap": {
            "cross_class_exact_text": cross_overlap,
            "mia_train_test_exact_text": tt_overlap,
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.output_dir, "diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    print("\n=== DONE ===")
    print(f"  Dataset:        {DATASET_ID}  split={args.wikimia_split}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  train.jsonl:    {len(train_records)} records ({n_train} per class)")
    print(f"  test.jsonl:     {len(test_records)} records ({n_test} per class)")
    print(f"  Size fallback:  {size_fallback}")
    print(f"  Cross-class overlap: {cross_overlap}")
    print(f"  MIA train/test overlap: {tt_overlap}")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare WikiMIA dataset for shard-membership auditing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--inspect-all", action="store_true",
                   help="Print split diagnostics for all WikiMIA splits and exit")
    p.add_argument("--wikimia-split", default="WikiMIA_length32",
                   choices=KNOWN_SPLITS,
                   help="WikiMIA split to prepare")
    p.add_argument("--preferred-train-per-class", type=int, default=325,
                   dest="preferred_train_per_class")
    p.add_argument("--preferred-test-per-class", type=int, default=500,
                   dest="preferred_test_per_class")
    p.add_argument("--min-train-per-class", type=int, default=200,
                   dest="min_train_per_class")
    p.add_argument("--min-test-per-class", type=int, default=200,
                   dest="min_test_per_class")
    p.add_argument("--min-words", type=int, default=8,
                   help="Minimum word count after whitespace normalization")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="data/processed/wikimia_length32")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token (WikiMIA is public, not required)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.inspect_all:
        cmd_inspect_all(token=args.token)
        sys.exit(0)
    cmd_prepare(args)


if __name__ == "__main__":
    main()

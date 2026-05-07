"""Prepare ArXiv nonmember-vs-nonmember control data for shard-membership auditing.

Both pseudo-classes are drawn exclusively from the MIMIR ArXiv nonmember/test
side. Labels are ARTIFICIAL — they do NOT indicate membership.

Usage:
    python scripts/data/prepare_mimir_arxiv_nonmember_control.py \\
        --num-train-per-class 170 \\
        --num-test-per-class 200 \\
        --seed 0 \\
        --output-dir data/processed/mimir_arxiv_nonmember_control_seed0
"""

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sys
from collections import Counter
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.datasets import (
    DATASET_ID,
    load_mimir_github_nonmembers, # Generic if we pass config='arxiv'
    _nonmember_filename,
)
from src.shard_audit.preprocessing import preprocess_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _word_count_histogram(texts: list) -> dict:
    bins = [0, 8, 16, 24, 32, 48, 64, 128, 999999]
    counts = Counter()
    for text in texts:
        wc = len(text.split())
        for lo, hi in zip(bins, bins[1:]):
            if lo <= wc < hi:
                counts[f"{lo}-{hi-1}"] += 1
                break
    return dict(counts)


def _preprocess_and_dedup(texts: list, max_words: int, min_words: int) -> tuple:
    kept, n_filtered, n_deduped = [], 0, 0
    seen = set()
    for t in texts:
        result = preprocess_text(t, max_words=max_words, min_words=min_words)
        if result is None:
            n_filtered += 1
            continue
        h = _text_hash(result)
        if h in seen:
            n_deduped += 1
            continue
        seen.add(h)
        kept.append(result)
    return kept, n_filtered, n_deduped


def _determine_sizes(n_available: int, requested_train: int, requested_test: int) -> tuple:
    per_class = requested_train + requested_test
    total_needed = per_class * 2
    if n_available >= total_needed:
        return requested_train, requested_test
    
    # Fallback logic
    for fb_train in [150, 100, 50]:
        if n_available >= (fb_train + requested_test) * 2:
            logger.warning("Not enough examples. Falling back to %d train per class.", fb_train)
            return fb_train, requested_test
            
    min_per = n_available // 4
    if min_per >= 10:
        logger.warning("Very small pool. Using %d train + %d test per class.", min_per, min_per)
        return min_per, min_per
        
    raise ValueError(f"Only {n_available} examples available. Cannot construct control.")


def _make_control_record(text: str, idx: int, label: int, control_label: str, phase_split: str) -> dict:
    return {
        "id": f"{control_label}-{idx:06d}",
        "text": text,
        "label": label,
        "control_label": control_label,
        "true_membership": "nonmember",
        "source": "mimir_arxiv_nonmember_control",
        "split_origin": "mimir_test_nonmember",
        "phase_split": phase_split,
        "text_hash": _text_hash(text),
    }


def _write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare MIMIR ArXiv nonmember-vs-nonmember control dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-id", default=DATASET_ID)
    p.add_argument("--config", default="arxiv")
    p.add_argument("--ngram-split", default="ngram_13_0.2")
    p.add_argument("--num-train-per-class", type=int, default=170)
    p.add_argument("--num-test-per-class", type=int, default=200)
    p.add_argument("--max-words", type=int, default=32)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="data/processed/mimir_arxiv_nonmember_control_seed0")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=== MIMIR ArXiv Nonmember-vs-Nonmember Control Data Preparation ===")
    logger.info("Config:        %s", args.config)
    logger.info("Requested:     %d train + %d test per pseudo-class", args.num_train_per_class, args.num_test_per_class)
    logger.info("Output dir:    %s", args.output_dir)

    # 1. Load nonmember texts
    logger.info("\n[1/5] Loading MIMIR ArXiv nonmember texts...")
    nonmember_texts_raw = load_mimir_github_nonmembers(
        config=args.config,
        split=args.ngram_split,
        token=args.token,
    )
    logger.info("Raw nonmember texts: %d", len(nonmember_texts_raw))

    # 2. Preprocess
    clean_texts, n_filtered, n_deduped = _preprocess_and_dedup(nonmember_texts_raw, args.max_words, args.min_words)
    n_available = len(clean_texts)
    logger.info("After preprocessing: %d kept", n_available)

    # 3. Sizes
    n_train, n_test = _determine_sizes(n_available, args.num_train_per_class, args.num_test_per_class)
    per_class = n_train + n_test
    
    # 4. Split
    rng = random.Random(args.seed)
    shuffled = list(clean_texts)
    rng.shuffle(shuffled)

    s0_pool = shuffled[:per_class]
    s1_pool = shuffled[per_class : per_class * 2]

    train_records = []
    for i, t in enumerate(s0_pool[:n_train]):
        train_records.append(_make_control_record(t, i, label=1, control_label="nonmember_a", phase_split="train"))
    for i, t in enumerate(s1_pool[:n_train]):
        train_records.append(_make_control_record(t, i, label=0, control_label="nonmember_b", phase_split="train"))

    test_records = []
    for i, t in enumerate(s0_pool[n_train:]):
        test_records.append(_make_control_record(t, i, label=1, control_label="nonmember_a", phase_split="test"))
    for i, t in enumerate(s1_pool[n_train:]):
        test_records.append(_make_control_record(t, i, label=0, control_label="nonmember_b", phase_split="test"))

    rng.shuffle(train_records)
    rng.shuffle(test_records)

    # 5. Write
    _write_jsonl(train_records, os.path.join(args.output_dir, "train.jsonl"))
    _write_jsonl(test_records,  os.path.join(args.output_dir, "test.jsonl"))

    manifest = {
        "experiment_type": "nonmember_vs_nonmember_control",
        "config": args.config,
        "seed": args.seed,
        "n_train_per_class": n_train,
        "n_test_per_class": n_test,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Done.")


if __name__ == "__main__":
    main()

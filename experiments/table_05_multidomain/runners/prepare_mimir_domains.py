"""Prepare MIMIR data for multiple domains for shard-membership auditing.

Loops over a list of MIMIR configs, tries splits in priority order, and
writes per-domain processed datasets to an output root directory.

Output layout:
    <output_root>/<domain_slug>/train.jsonl
    <output_root>/<domain_slug>/test.jsonl
    <output_root>/<domain_slug>/manifest.json
    <output_root>/<domain_slug>/diagnostics.json

Usage:
    python experiments/table_05_multidomain/runners/prepare_mimir_domains.py \\
        --configs github,dm_mathematics,arxiv,wikipedia_(en),pile_cc \\
        --ngram-split ngram_13_0.2 \\
        --fallback-splits ngram_7_0.2,ngram_13_0.8,none \\
        --num-train-per-class 500 \\
        --num-test-per-class 200 \\
        --max-words 32 --min-words 8 \\
        --seed 0 \\
        --output-root data/processed/mimir_domains
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.data.datasets import (
    DATASET_ID,
    load_mimir_github_via_hub_download,
    texts_to_records,
    _member_filename,
    _nonmember_filename,
)
from src.shard_audit.data.preprocessing import preprocess_text
from src.shard_audit.data.splitting import stratified_train_test_split
from src.shard_audit.data.sanity_checks import check_label_balance, check_required_fields

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

FALLBACK_SIZES = [
    (500, 200),
    (300, 100),
    (200, 100),
    (100, 100),
]


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def domain_slug(config: str) -> str:
    """Convert MIMIR config name to a filesystem-safe slug."""
    return re.sub(r'[^a-zA-Z0-9_]', '_', config).strip('_').replace('__', '_')


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


def _preprocess_list(texts: list, max_words: int, min_words: int) -> tuple:
    """Preprocess and deduplicate. Returns (clean_texts, n_filtered, n_deduped)."""
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


def _write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _determine_sizes(n_member: int, n_nonmember: int) -> tuple:
    """Return (n_train, n_test) using the smallest pool as the bottleneck."""
    pool = min(n_member, n_nonmember)
    for n_train, n_test in FALLBACK_SIZES:
        if pool >= n_train + n_test:
            return n_train, n_test
    return None, None


# ------------------------------------------------------------------ #
# Per-domain processing
# ------------------------------------------------------------------ #

def process_domain(
    config: str,
    splits_to_try: list,
    num_train: int,
    num_test: int,
    max_words: int,
    min_words: int,
    seed: int,
    output_dir: str,
    token: str,
) -> dict:
    """Process one MIMIR domain. Returns a status dict."""
    slug = domain_slug(config)
    os.makedirs(output_dir, exist_ok=True)

    # Skip if already done
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        logger.info("[%s] Already processed — skipping.", slug)
        with open(manifest_path) as f:
            return {"status": "skipped", "slug": slug, **json.load(f)}

    logger.info("\n" + "=" * 60)
    logger.info("Processing domain: %s (config=%s)", slug, config)
    logger.info("=" * 60)

    for split in splits_to_try:
        logger.info("[%s] Trying split: %s", slug, split)
        try:
            member_texts_raw, nonmember_texts_raw = load_mimir_github_via_hub_download(
                config=config, split=split, token=token,
            )
        except Exception as e:
            logger.warning("[%s] Could not load split %s: %s", slug, split, e)
            continue

        logger.info("[%s] Raw: %d members, %d nonmembers", slug,
                    len(member_texts_raw), len(nonmember_texts_raw))

        wc_m_before = _word_count_histogram(member_texts_raw)
        wc_nm_before = _word_count_histogram(nonmember_texts_raw)

        m_clean, m_filt, m_dedup = _preprocess_list(member_texts_raw, max_words, min_words)
        nm_clean, nm_filt, nm_dedup = _preprocess_list(nonmember_texts_raw, max_words, min_words)

        logger.info("[%s] After preprocessing: %d members, %d nonmembers",
                    slug, len(m_clean), len(nm_clean))

        # Determine sizes
        n_train, n_test = _determine_sizes(len(m_clean), len(nm_clean))
        if n_train is None:
            logger.warning("[%s] Split %s: too few examples (m=%d, nm=%d). Trying next split.",
                           slug, split, len(m_clean), len(nm_clean))
            continue

        # If requested sizes are achievable, use them; otherwise use fallback
        if len(m_clean) >= num_train + num_test and len(nm_clean) >= num_train + num_test:
            final_train, final_test = num_train, num_test
            size_fallback = False
        else:
            final_train, final_test = n_train, n_test
            size_fallback = True
            logger.warning("[%s] Using fallback sizes: %d train + %d test per class",
                           slug, final_train, final_test)

        # Build records
        m_records = texts_to_records(m_clean, label=1,
                                     source=f"mimir_{config}", split_origin="member",
                                     id_prefix="member-")
        nm_records = texts_to_records(nm_clean, label=0,
                                      source=f"mimir_{config}", split_origin="nonmember",
                                      id_prefix="nonmember-")

        # Cross-class overlap check
        m_hashes = {r["text_hash"] for r in m_records}
        nm_hashes = {r["text_hash"] for r in nm_records}
        cross_overlap = len(m_hashes & nm_hashes)
        if cross_overlap:
            logger.warning("[%s] Cross-class overlap: %d texts appear in both classes.",
                           slug, cross_overlap)

        # Split
        try:
            train_records, test_records = stratified_train_test_split(
                member_records=m_records,
                nonmember_records=nm_records,
                num_train_per_class=final_train,
                num_test_per_class=final_test,
                seed=seed,
            )
        except ValueError as e:
            logger.error("[%s] Split failed: %s", slug, e)
            continue

        # Write outputs
        _write_jsonl(train_records, os.path.join(output_dir, "train.jsonl"))
        _write_jsonl(test_records, os.path.join(output_dir, "test.jsonl"))
        logger.info("[%s] Wrote %d train, %d test records.",
                    slug, len(train_records), len(test_records))

        n_train_m  = sum(1 for r in train_records if r["label"] == 1)
        n_train_nm = sum(1 for r in train_records if r["label"] == 0)
        n_test_m   = sum(1 for r in test_records  if r["label"] == 1)
        n_test_nm  = sum(1 for r in test_records  if r["label"] == 0)

        manifest = {
            "domain": slug,
            "config": config,
            "selected_split": split,
            "split_fallback": split != splits_to_try[0],
            "size_fallback": size_fallback,
            "n_member_raw": len(member_texts_raw),
            "n_nonmember_raw": len(nonmember_texts_raw),
            "n_member_after_preprocessing": len(m_clean),
            "n_nonmember_after_preprocessing": len(nm_clean),
            "n_train_per_class": final_train,
            "n_test_per_class": final_test,
            "n_train": len(train_records),
            "n_test": len(test_records),
            "n_train_member": n_train_m,
            "n_train_nonmember": n_train_nm,
            "n_test_member": n_test_m,
            "n_test_nonmember": n_test_nm,
            "cross_class_overlap": cross_overlap,
            "max_words": max_words,
            "min_words": min_words,
            "seed": seed,
            "member_source_file": _member_filename(config, split),
            "nonmember_source_file": _nonmember_filename(config, split),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "output_dir": os.path.abspath(output_dir),
        }
        diagnostics = {
            "word_count_histogram": {
                "member_before": wc_m_before,
                "nonmember_before": wc_nm_before,
                "member_after": _word_count_histogram(m_clean),
                "nonmember_after": _word_count_histogram(nm_clean),
            },
            "filtering": {
                "member_filtered": m_filt,
                "member_deduped": m_dedup,
                "nonmember_filtered": nm_filt,
                "nonmember_deduped": nm_dedup,
            },
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        with open(os.path.join(output_dir, "diagnostics.json"), "w") as f:
            json.dump(diagnostics, f, indent=2)

        logger.info("[%s] Done. Split=%s, train=%d/class, test=%d/class",
                    slug, split, final_train, final_test)
        return {"status": "ok", "slug": slug, **manifest}

    logger.error("[%s] All splits exhausted. Domain skipped.", slug)
    failure = {
        "status": "failed",
        "slug": slug,
        "config": config,
        "reason": "all splits exhausted",
    }
    with open(os.path.join(output_dir, "FAILED.json"), "w") as f:
        json.dump(failure, f, indent=2)
    return failure


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare MIMIR data for multiple domains.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--configs",
                   default="github,dm_mathematics,arxiv,wikipedia_(en),pile_cc",
                   help="Comma-separated list of MIMIR config names")
    p.add_argument("--ngram-split", default="ngram_13_0.2", dest="ngram_split")
    p.add_argument("--fallback-splits", default="ngram_7_0.2,ngram_13_0.8,none",
                   dest="fallback_splits")
    p.add_argument("--num-train-per-class", type=int, default=500)
    p.add_argument("--num-test-per-class",  type=int, default=200)
    p.add_argument("--max-words", type=int, default=32)
    p.add_argument("--min-words", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-root", default="data/processed/mimir_domains")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return p.parse_args()


def main():
    args = parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    primary_split = args.ngram_split
    fallback_splits = [s.strip() for s in args.fallback_splits.split(",") if s.strip()]
    splits_to_try = [primary_split] + [s for s in fallback_splits if s != primary_split]

    logger.info("=== MIMIR Multi-Domain Data Preparation ===")
    logger.info("Configs:         %s", configs)
    logger.info("Splits to try:   %s", splits_to_try)
    logger.info("Train/class:     %d", args.num_train_per_class)
    logger.info("Test/class:      %d", args.num_test_per_class)
    logger.info("Output root:     %s", args.output_root)

    os.makedirs(args.output_root, exist_ok=True)

    results = []
    for config in configs:
        slug = domain_slug(config)
        output_dir = os.path.join(args.output_root, slug)
        result = process_domain(
            config=config,
            splits_to_try=splits_to_try,
            num_train=args.num_train_per_class,
            num_test=args.num_test_per_class,
            max_words=args.max_words,
            min_words=args.min_words,
            seed=args.seed,
            output_dir=output_dir,
            token=args.token,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("=== DONE ===")
    print(f"{'Domain':<22} {'Status':<10} {'Split':<16} {'Train/cls':>10} {'Test/cls':>9}")
    print("-" * 70)
    for r in results:
        if r["status"] in ("ok", "skipped"):
            print(f"{r['slug']:<22} {r['status']:<10} "
                  f"{r.get('selected_split','?'):<16} "
                  f"{r.get('n_train_per_class','?'):>10} "
                  f"{r.get('n_test_per_class','?'):>9}")
        else:
            print(f"{r['slug']:<22} {r['status']:<10} {'—':<16}")

    # Write domain index
    index_path = os.path.join(args.output_root, "domain_index.json")
    with open(index_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Domain index written to %s", index_path)


if __name__ == "__main__":
    main()

"""Re-split existing MIMIR domain score files to a new train/test ratio.

Merges train_scores.jsonl + test_scores.jsonl from an existing score root,
re-shuffles, and writes new files with the requested per-class sizes.
No GPU work required — pure data manipulation.

Usage:
    python scripts/data/resplit_mimir_domain_scores.py \\
        --old-score-root data/scores/mimir_domains \\
        --new-score-root data/scores/mimir_domains_300_400 \\
        --num-train-per-class 300 \\
        --num-test-per-class  400 \\
        --seed 0
"""

import argparse
import json
import logging
import os
import random
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def resplit_domain_model(domain, model_slug, old_root, new_root, n_train, n_test, seed):
    old_dir = os.path.join(old_root, domain, model_slug)
    new_dir = os.path.join(new_root, domain, model_slug)

    new_train_path = os.path.join(new_dir, "train_scores.jsonl")
    new_test_path  = os.path.join(new_dir, "test_scores.jsonl")

    if os.path.isfile(new_train_path) and os.path.isfile(new_test_path):
        logger.info("  [%s/%s] Already exists — skipping.", domain, model_slug)
        return True

    old_train = os.path.join(old_dir, "train_scores.jsonl")
    old_test  = os.path.join(old_dir, "test_scores.jsonl")
    if not os.path.isfile(old_train) or not os.path.isfile(old_test):
        logger.warning("  [%s/%s] Missing old score files — skipping.", domain, model_slug)
        return False

    all_records = _load_jsonl(old_train) + _load_jsonl(old_test)
    members    = [r for r in all_records if r["label"] == 1]
    nonmembers = [r for r in all_records if r["label"] == 0]

    needed = n_train + n_test
    if len(members) < needed or len(nonmembers) < needed:
        logger.error(
            "  [%s/%s] Insufficient records: members=%d, nonmembers=%d, need=%d per class.",
            domain, model_slug, len(members), len(nonmembers), needed,
        )
        return False

    rng = random.Random(seed)
    rng.shuffle(members)
    rng.shuffle(nonmembers)

    train_records = members[:n_train] + nonmembers[:n_train]
    test_records  = members[n_train:n_train + n_test] + nonmembers[n_train:n_train + n_test]

    _write_jsonl(train_records, new_train_path)
    _write_jsonl(test_records,  new_test_path)

    old_manifest = os.path.join(old_dir, "manifest.json")
    if os.path.isfile(old_manifest):
        with open(old_manifest) as f:
            manifest = json.load(f)
        manifest.update({
            "n_train_per_class": n_train,
            "n_test_per_class":  n_test,
            "n_train": len(train_records),
            "n_test":  len(test_records),
            "resplit_from": os.path.abspath(old_dir),
            "resplit_seed": seed,
            "resplit_timestamp": datetime.utcnow().isoformat() + "Z",
        })
        with open(os.path.join(new_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    logger.info(
        "  [%s/%s] Re-split: members=%d, nonmembers=%d → train=%d/class, test=%d/class.",
        domain, model_slug, len(members), len(nonmembers), n_train, n_test,
    )
    return True


def parse_args():
    p = argparse.ArgumentParser(
        description="Re-split MIMIR domain score files to a new train/test ratio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--old-score-root", default="data/scores/mimir_domains")
    p.add_argument("--new-score-root", default="data/scores/mimir_domains_300_400")
    p.add_argument("--num-train-per-class", type=int, default=300)
    p.add_argument("--num-test-per-class",  type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--model-slugs",
        default="parent_pythia_1_4b,target_hermaster_pythia1_4b_lamini_docs",
    )
    return p.parse_args()


def main():
    args = parse_args()
    model_slugs = [s.strip() for s in args.model_slugs.split(",")]

    if not os.path.isdir(args.old_score_root):
        logger.error("Old score root not found: %s", args.old_score_root)
        return

    domains = sorted(
        e for e in os.listdir(args.old_score_root)
        if os.path.isdir(os.path.join(args.old_score_root, e))
    )
    logger.info("Domains found: %s", domains)
    logger.info("Model slugs:   %s", model_slugs)
    logger.info("New split:     %d train + %d test per class", args.num_train_per_class, args.num_test_per_class)

    ok, failed = 0, 0
    for domain in domains:
        for slug in model_slugs:
            success = resplit_domain_model(
                domain, slug,
                args.old_score_root, args.new_score_root,
                args.num_train_per_class, args.num_test_per_class,
                args.seed,
            )
            if success:
                ok += 1
            else:
                failed += 1

    print(f"\nDone. {ok} split(s) written, {failed} skipped/failed.")
    print(f"New score root: {args.new_score_root}")


if __name__ == "__main__":
    main()

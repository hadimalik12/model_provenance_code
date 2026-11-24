#!/usr/bin/env python3
"""Diagnose measured shift between cached MIMIR GitHub member/control shards.

No Pythia inference is performed.  The frozen encoder output is cached, and
``--resume`` reuses it on subsequent analysis/report runs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shard_audit.data.frozen_embeddings import frozen_text_embeddings
from src.shard_audit.data.split_diagnostics import (
    classifier_two_sample_test,
    lexical_js_divergence,
    numeric_feature_diagnostics,
    text_duplicate_diagnostics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(
        "artifacts/public_pythia_derivatives/aistats27_matched/seed_0/prepared/"
        "mimir_github_matched_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "artifacts/split_diagnostics/aistats27_q3/mimir_github"))
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--encoder-revision", default="c9745ed1d9f207416be6d2e6f8de32d1f16199bf")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Stratified cap for fast smoke runs")
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Run only structural, lexical, and duplicate diagnostics")
    return parser.parse_args()


def read_records(data_dir: Path) -> list[dict]:
    records: list[dict] = []
    for name in ("train.jsonl", "test.jsonl"):
        with (data_dir / name).open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def stratified_subset(records: list[dict], maximum: int | None, seed: int) -> list[dict]:
    if maximum is None:
        return records
    rng = random.Random(seed)
    selected: list[dict] = []
    for label in (0, 1):
        group = [record for record in records if int(record["label"]) == label]
        selected.extend(rng.sample(group, min(maximum, len(group))))
    return sorted(selected, key=lambda record: str(record["id"]))


def main() -> None:
    args = parse_args()
    records = stratified_subset(read_records(args.data_dir), args.max_per_class, args.seed)
    labels = [int(record["label"]) for record in records]
    texts = [record["text"] for record in records]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    structural_names = ("word_count", "char_count", "line_count", "code_symbol_fraction")
    structural = {
        name: [float(record.get(name, 0.0)) for record in records]
        for name in structural_names
    }
    report: dict = {
        "status": "observed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "source_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
        "seed": args.seed,
        "sample": {"member": int(sum(labels)), "nonmember": int(len(labels) - sum(labels))},
        "structural": numeric_feature_diagnostics(structural, labels),
        "lexical": lexical_js_divergence(
            (text for text, label in zip(texts, labels) if label == 1),
            (text for text, label in zip(texts, labels) if label == 0),
        ),
        "duplicates": text_duplicate_diagnostics(texts, labels),
        "interpretation": (
            "These diagnostics can detect measured split differences. Failure to reject "
            "does not prove equality, exchangeability, or exact parentage."
        ),
    }

    if not args.skip_embeddings:
        cache = args.output_dir / "frozen_text_embeddings.npz"
        ids = np.asarray([str(record["id"]) for record in records])
        if args.resume and cache.exists():
            saved = np.load(cache)
            if not np.array_equal(saved["ids"], ids):
                raise RuntimeError("Cached embeddings do not match selected records")
            embeddings = saved["embeddings"]
        else:
            embeddings = frozen_text_embeddings(
                texts, model_id=args.encoder, revision=args.encoder_revision,
                batch_size=args.batch_size, max_length=args.max_length, device=args.device,
            )
            np.savez_compressed(cache, ids=ids, embeddings=embeddings)
        report["semantic_c2st"] = {
            "encoder": args.encoder,
            "encoder_revision": args.encoder_revision,
            **classifier_two_sample_test(
                embeddings, labels, seed=args.seed, permutations=args.permutations,
            ),
        }
    else:
        report["semantic_c2st"] = {"status": "not_run", "reason": "--skip-embeddings"}

    (args.output_dir / "diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "command": " ".join(sys.argv),
        "inputs": [str(args.data_dir / "train.jsonl"), str(args.data_dir / "test.jsonl")],
        "outputs": [str(args.output_dir / "diagnostics.json")],
        "resume_cache": str(args.output_dir / "frozen_text_embeddings.npz"),
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

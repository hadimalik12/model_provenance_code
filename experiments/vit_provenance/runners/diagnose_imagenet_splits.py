#!/usr/bin/env python3
"""Diagnose measured shift in prepared ImageNet train/validation samples."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shard_audit.data.frozen_embeddings import frozen_image_embeddings
from src.shard_audit.data.split_diagnostics import (
    classifier_two_sample_test,
    image_duplicate_diagnostics,
    numeric_feature_diagnostics,
    resolve_image_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(
        "artifacts/vit_provenance/public_derivatives_seed0/prepared/imagenet_shard"))
    parser.add_argument("--output-dir", type=Path, default=Path(
        "artifacts/split_diagnostics/aistats27_q3/imagenet"))
    parser.add_argument("--encoder", default="openai/clip-vit-base-patch32")
    parser.add_argument("--encoder-revision", default="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap each origin/class cell for smoke runs")
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    return parser.parse_args()


def read_records(data_dir: Path) -> list[dict]:
    records: list[dict] = []
    for name in ("train.jsonl", "test.jsonl"):
        with (data_dir / name).open() as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def subset(records: list[dict], max_classes: int | None,
           max_per_class: int | None, seed: int) -> list[dict]:
    rng = random.Random(seed)
    classes = sorted({int(record["imagenet_class"]) for record in records})
    if max_classes is not None:
        classes = sorted(rng.sample(classes, min(max_classes, len(classes))))
    selected: list[dict] = []
    for class_id in classes:
        for label in (0, 1):
            cell = [record for record in records
                    if int(record["imagenet_class"]) == class_id
                    and int(record["label"]) == label]
            if max_per_class is not None:
                cell = rng.sample(cell, min(max_per_class, len(cell)))
            selected.extend(cell)
    return sorted(selected, key=lambda record: str(record["id"]))


def image_features(paths: list[Path]) -> dict[str, list[float]]:
    output = {name: [] for name in (
        "width", "height", "brightness", "contrast", "red_mean", "green_mean", "blue_mean"
    )}
    for path in paths:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        gray = rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        output["width"].append(float(rgb.shape[1]))
        output["height"].append(float(rgb.shape[0]))
        output["brightness"].append(float(gray.mean()))
        output["contrast"].append(float(gray.std()))
        output["red_mean"].append(float(rgb[:, :, 0].mean()))
        output["green_mean"].append(float(rgb[:, :, 1].mean()))
        output["blue_mean"].append(float(rgb[:, :, 2].mean()))
    return output


def class_balance(records: list[dict]) -> dict:
    member = Counter(int(r["imagenet_class"]) for r in records if int(r["label"]) == 1)
    nonmember = Counter(int(r["imagenet_class"]) for r in records if int(r["label"]) == 0)
    classes = sorted(member.keys() | nonmember.keys())
    p = np.asarray([member[c] for c in classes], dtype=float); p /= p.sum()
    q = np.asarray([nonmember[c] for c in classes], dtype=float); q /= q.sum()
    return {
        "number_of_classes": len(classes),
        "total_variation_distance": float(0.5 * np.abs(p - q).sum()),
        "maximum_class_probability_difference": float(np.abs(p - q).max(initial=0.0)),
        "counts": {str(c): {"member": member[c], "nonmember": nonmember[c]} for c in classes},
    }


def main() -> None:
    args = parse_args()
    if not (args.data_dir / "manifest.json").exists():
        raise SystemExit(
            f"Prepared ImageNet data not found at {args.data_dir}. Run prepare_shards.py first; "
            "this diagnostic never silently substitutes another dataset."
        )
    records = subset(read_records(args.data_dir), args.max_classes,
                     args.max_per_class, args.seed)
    paths = [resolve_image_path(record, args.data_dir) for record in records]
    labels = [int(record["label"]) for record in records]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "status": "observed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "source_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
        "seed": args.seed,
        "sample": {"member": int(sum(labels)), "nonmember": int(len(labels) - sum(labels))},
        "class_balance": class_balance(records),
        "image_properties": numeric_feature_diagnostics(image_features(paths), labels),
        "duplicates": image_duplicate_diagnostics(paths, labels),
        "interpretation": (
            "These diagnostics can detect measured split differences. Failure to reject "
            "does not prove equality, exchangeability, or exact parentage."
        ),
    }
    if not args.skip_embeddings:
        cache = args.output_dir / "frozen_image_embeddings.npz"
        ids = np.asarray([str(record["id"]) for record in records])
        if args.resume and cache.exists():
            saved = np.load(cache)
            if not np.array_equal(saved["ids"], ids):
                raise RuntimeError("Cached embeddings do not match selected records")
            embeddings = saved["embeddings"]
        else:
            embeddings = frozen_image_embeddings(
                paths, model_id=args.encoder, revision=args.encoder_revision,
                batch_size=args.batch_size, device=args.device,
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
        "resume_cache": str(args.output_dir / "frozen_image_embeddings.npz"),
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Score vanilla, conservative augmented, and ablation variants for Table 5."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.scoring.logprobs import extract_token_logprobs_with_output_head, get_device, load_model_and_tokenizer, load_output_head
from src.shard_audit.scoring.mia_scores import compute_all_scores
from src.shard_audit.scoring.table5_safe_augmentations import table5_variant_views

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _score_text(text, model, head, tokenizer, device, k_pcts, max_length):
    logprobs = extract_token_logprobs_with_output_head([text], model, head, tokenizer, device, max_length=max_length)[0]
    return compute_all_scores(text, logprobs, k_pcts=tuple(k_pcts))


def _score_records(records, model, head, tokenizer, device, k_pcts, domain, output_path, max_length):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    completed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as existing:
            for line in existing:
                if line.strip():
                    completed_ids.add(json.loads(line)["id"])
    remaining = [record for record in records if record["id"] not in completed_ids]
    logger.info("%s: %d complete, %d remaining", domain, len(completed_ids), len(remaining))
    with open(output_path, "a", encoding="utf-8") as handle:
        for index, record in enumerate(remaining, start=1):
            text = record["text"]
            original = _score_text(text, model, head, tokenizer, device, k_pcts, max_length)
            result = {key: record.get(key) for key in ("id", "label", "phase_split", "text_hash") if key in record}
            for k in k_pcts:
                result[f"min_k_{k}_logprob"] = round(float(original[f"min_k_{k}_logprob"]), 6)

            for variant, views in table5_variant_views(text, domain).items():
                per_view = [_score_text(view, model, head, tokenizer, device, k_pcts, max_length) for view in views]
                for k in k_pcts:
                    values = [view[f"min_k_{k}_logprob"] for view in per_view]
                    suffix = "_mean" if variant == "safe_full" else ""
                    result[f"{variant}_min_k_{k}_logprob{suffix}"] = round(float(np.mean(values)), 6)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if index % 10 == 0 or index == len(remaining):
                logger.info("Scored %d/%d remaining records", index, len(remaining))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--min-k-pcts", default="20,40")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", default="auto")
    args = parser.parse_args()
    k_pcts = [int(value) for value in args.min_k_pcts.split(",") if value.strip()]
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(args.target_model, device=device, dtype_str=args.dtype)
    _, parent_head = load_output_head(args.parent_model, device=device, dtype_str=args.dtype)
    _score_records(_load(args.train_file), model, parent_head, tokenizer, device, k_pcts, args.domain, os.path.join(args.output_dir, "train_scores.jsonl"), args.max_length)
    _score_records(_load(args.test_file), model, parent_head, tokenizer, device, k_pcts, args.domain, os.path.join(args.output_dir, "test_scores.jsonl"), args.max_length)


if __name__ == "__main__":
    main()

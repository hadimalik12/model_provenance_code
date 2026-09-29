"""Prepare bounded, shared downstream datasets for matched target/control runs."""
from __future__ import annotations

import argparse
import hashlib
import heapq
from pathlib import Path

from .common import DEFAULT_CONFIG, artifacts, config, fingerprint, read_jsonl, require_manifest, write_json, write_jsonl

PROFILES = {
    "hh_sft": {"dataset": "Anthropic/hh-rlhf", "subset": "all", "split": "train"},
    "helpful_sft": {"dataset": "Anthropic/hh-rlhf", "subset": "helpful", "split": "train"},
    "tulu_sft": {"dataset": "allenai/tulu-v2-sft-mixture", "subset": None, "split": "train"},
    "lamini_sft": {"dataset": "MBZUAI/LaMini-instruction", "subset": None, "split": "train"},
    "finance_sft": {"dataset": "gbharti/finance-alpaca", "subset": None, "split": "train"},
}


def normalize(profile: str, raw: dict) -> dict | None:
    if profile in {"hh_sft", "helpful_sft"}:
        chosen = raw.get("chosen", "")
        if not isinstance(chosen, str) or "\n\nAssistant:" not in chosen:
            return None
        prefix, answer = chosen.rsplit("\n\nAssistant:", 1)
        prompt, response = prefix + "\n\nAssistant:", answer.strip()
    elif profile == "tulu_sft":
        messages = raw.get("messages", [])
        if not isinstance(messages, list) or len(messages) < 2 or messages[-1].get("role") != "assistant":
            return None
        response = messages[-1].get("content", "").strip()
        prompt = "\n".join(f"### {m.get('role', '').title()}: {m.get('content', '').strip()}"
                           for m in messages[:-1]) + "\n### Assistant:"
    elif profile == "lamini_sft":
        instruction = raw.get("instruction", "")
        response = raw.get("response", "")
        if not isinstance(instruction, str) or not isinstance(response, str):
            return None
        prompt = f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"
        response = response.strip()
    elif profile == "finance_sft":
        instruction = raw.get("instruction", "")
        context = raw.get("input", "")
        response = raw.get("output", "")
        if not all(isinstance(value, str) for value in (instruction, context, response)):
            return None
        prompt = f"### Instruction:\n{instruction.strip()}\n"
        if context.strip():
            prompt += f"\n### Input:\n{context.strip()}\n"
        prompt += "\n### Response:\n"
        response = response.strip()
    else:
        raise ValueError(profile)
    if not prompt.strip() or len(response) < 2:
        return None
    text = prompt + " " + response
    return {"id": hashlib.sha256(text.encode()).hexdigest(), "prompt": prompt,
            "response": response, "text": text}


def iter_source(profile: str):
    from datasets import load_dataset
    source = PROFILES[profile]
    if source["subset"] == "helpful":
        for subdir in ("helpful-base", "helpful-online", "helpful-rejection-sampled"):
            yield from load_dataset(source["dataset"], data_dir=subdir, split="train", streaming=True)
    else:
        yield from load_dataset(source["dataset"], split=source["split"], streaming=True)


def select(rows, profile: str, n: int, excluded: set[str]) -> list[dict]:
    heap: list[tuple[int, str, dict]] = []
    seen = set()
    for raw in rows:
        rec = normalize(profile, raw)
        if rec is None or rec["id"] in seen or rec["id"] in excluded:
            continue
        seen.add(rec["id"])
        rank = int(hashlib.sha256((profile + rec["id"]).encode()).hexdigest(), 16)
        item = (-rank, rec["id"], rec)
        if len(heap) < n:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) != n:
        raise ValueError(f"{profile}: found only {len(heap)} valid rows; need {n}")
    return [x[2] for x in sorted(heap, key=lambda x: -x[0])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--profile", choices=PROFILES, required=True)
    args = parser.parse_args()
    cfg = config(args.config)
    require_manifest(cfg)
    out = artifacts(cfg) / "downstream" / args.profile
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Downstream profile already prepared: {out}")
    prepared = artifacts(cfg) / "prepared"
    excluded = {r["id"] for arm in ("s_train.jsonl", "s_prime.jsonl")
                for r in read_jsonl(prepared / arm)}
    n_train = cfg["downstream"]["max_examples"]
    n_val = cfg["downstream"]["validation_examples"]
    selected = select(iter_source(args.profile), args.profile, n_train + n_val, excluded)
    write_jsonl(out / "train.jsonl", selected[:n_train])
    write_jsonl(out / "validation.jsonl", selected[n_train:])
    write_json(out / "manifest.json", {"profile": args.profile, "source": PROFILES[args.profile],
               "n_train": n_train, "n_validation": n_val,
               "selection_digest": fingerprint([r["id"] for r in selected]),
               "config_fingerprint": fingerprint(cfg),
               "note": "Reduced-budget subset of original downstream dataset"})
    print(f"Prepared {args.profile}: {n_train} train / {n_val} validation")


if __name__ == "__main__":
    main()

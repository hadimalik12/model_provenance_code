"""Stream paired examples and select equal disjoint S/S-prime arms by secret key."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import random
import secrets
from pathlib import Path

from .common import DEFAULT_CONFIG, artifacts, config, fingerprint, write_json, write_jsonl


def format_pair(prompt: str, response: str) -> str:
    return f"### Task\n{prompt.strip()}\n### Python\n{response.strip()}"


def canonical_row(raw: dict, limits: dict) -> dict | None:
    if limits.get("format") == "instructcoder":
        instruction = raw.get("instruction", "")
        before = raw.get("input", "")
        if not isinstance(instruction, str) or not isinstance(before, str):
            return None
        prompt = f"{instruction.strip()}\n### Existing code\n{before.strip()}"
        response = raw.get("output", "")
    else:
        prompt = raw.get("func_documentation_string", raw.get("prompt", ""))
        response = raw.get("func_code_string", raw.get("response", ""))
    if not isinstance(prompt, str) or not isinstance(response, str):
        return None
    prompt, response = prompt.strip(), response.strip()
    if not 16 <= len(prompt) <= limits["max_prompt_chars"]:
        return None
    if not 32 <= len(response) <= limits["max_response_chars"]:
        return None
    if raw.get("language", "python") != "python":
        return None
    formatted_prompt = f"### Task\n{prompt}\n### Answer\n"
    text = formatted_prompt + response
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"id": digest, "prompt": formatted_prompt, "response": response, "text": text,
            "text_hash": digest, "repository": raw.get("repository_name", ""),
            "url": raw.get("func_code_url", "")}


def choose(rows, key: bytes, limits: dict, n_per_arm: int, transform=None) -> tuple[list[dict], list[dict], dict]:
    """Choose the lowest 2n keyed ranks using bounded payload memory."""
    if n_per_arm < 1:
        raise ValueError("n_per_arm must be positive")
    capacity = 2 * n_per_arm
    heap: list[tuple[int, str, dict]] = []
    seen: set[str] = set()
    excluded = set(limits.get("exclude_ids", []))
    excluded_count = 0
    window_duplicates = 0
    raw_count = 0
    for raw in rows:
        raw_count += 1
        rec = canonical_row(raw, limits)
        if rec is None:
            continue
        if rec["id"] in excluded:
            excluded_count += 1
            continue
        if transform is not None:
            rec = transform(rec)
            if rec is None:
                continue
        if rec["text_hash"] in seen:
            window_duplicates += 1
            continue
        seen.add(rec["text_hash"])
        rank = int.from_bytes(
            hashlib.blake2b(rec["id"].encode(), key=key, digest_size=16).digest(), "big"
        )
        item = (-rank, rec["id"], rec)
        if len(heap) < capacity:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    if len(heap) != capacity:
        raise ValueError(f"Only {len(heap)} unique eligible rows; need {capacity}")
    chosen = [item[2] for item in sorted(heap, key=lambda x: -x[0])]
    selected, control = chosen[:n_per_arm], chosen[n_per_arm:]
    return selected, control, {"raw_seen": raw_count, "eligible_unique": len(seen),
                               "excluded_known_pile_github_overlap": excluded_count,
                               "duplicate_windows_removed": window_duplicates,
                               "s_count": len(selected), "s_prime_count": len(control)}


def choose_global_shuffle(rows, key: bytes, limits: dict, n_per_arm: int,
                          transform=None) -> tuple[list[dict], list[dict], dict]:
    """Shuffle the complete processed pool, then take two consecutive arms."""
    if n_per_arm < 1:
        raise ValueError("n_per_arm must be positive")
    eligible = []
    seen: set[str] = set()
    excluded = set(limits.get("exclude_ids", []))
    excluded_count = 0
    window_duplicates = 0
    raw_count = 0
    for raw in rows:
        raw_count += 1
        rec = canonical_row(raw, limits)
        if rec is None:
            continue
        if rec["id"] in excluded:
            excluded_count += 1
            continue
        if transform is not None:
            rec = transform(rec)
            if rec is None:
                continue
        if rec["text_hash"] in seen:
            window_duplicates += 1
            continue
        seen.add(rec["text_hash"])
        eligible.append(rec)
    if len(eligible) < 2 * n_per_arm:
        raise ValueError(f"Only {len(eligible)} unique eligible rows; need {2 * n_per_arm}")
    random.Random(int.from_bytes(key, "big")).shuffle(eligible)
    selected, control = eligible[:n_per_arm], eligible[n_per_arm:2 * n_per_arm]
    return selected, control, {"raw_seen": raw_count, "eligible_unique": len(seen),
                               "excluded_known_pile_github_overlap": excluded_count,
                               "duplicate_windows_removed": window_duplicates,
                               "s_count": len(selected), "s_prime_count": len(control)}


def fit_model_window(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    """Shorten responses deterministically so training and audit see the same text."""
    if max_length < 32:
        raise ValueError("max_length must be at least 32")
    result = []
    for original in rows:
        row = dict(original)
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        response_reserve = min(64, max_length // 2)
        if len(prompt_ids) > max_length - response_reserve:
            prefix, marker, _ = row["prompt"].rpartition("### Answer\n")
            if not marker:
                raise ValueError("Prepared prompt is missing the response marker")
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            marker_ids = tokenizer(marker, add_special_tokens=False)["input_ids"]
            prefix_ids = prefix_ids[:max_length - response_reserve - len(marker_ids)]
            row["prompt"] = tokenizer.decode(prefix_ids, clean_up_tokenization_spaces=False) + marker
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            while len(prompt_ids) > max_length - response_reserve:
                prefix_ids = prefix_ids[:-1]
                row["prompt"] = tokenizer.decode(prefix_ids, clean_up_tokenization_spaces=False) + marker
                prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        response_ids = response_ids[:max_length - len(prompt_ids)]
        row["response"] = tokenizer.decode(response_ids, clean_up_tokenization_spaces=False)
        row["text"] = row["prompt"] + row["response"]
        # Decode boundaries can change tokenization by one or two positions.
        while len(tokenizer(row["text"], add_special_tokens=False)["input_ids"]) > max_length:
            response_ids = response_ids[:-1]
            row["response"] = tokenizer.decode(response_ids, clean_up_tokenization_spaces=False)
            row["text"] = row["prompt"] + row["response"]
        if len(response_ids) < 8:
            raise ValueError("Selected example has fewer than eight response tokens")
        row["text_hash"] = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        result.append(row)
    if len({r["text_hash"] for r in result}) != len(result):
        raise ValueError("Truncation created duplicate selected texts")
    return result


def split_audit(selected: list[dict], control: list[dict], n_cal: int, n_test: int):
    if min(n_cal, n_test) < 1 or n_cal + n_test > min(len(selected), len(control)):
        raise ValueError("Audit split exceeds either arm")
    def label(rows: list[dict], y: int, phase: str):
        return [{"id": r["id"], "label": y, "phase_split": phase,
                 "text": r["text"], "text_hash": r["text_hash"]} for r in rows]
    calibration = label(selected[:n_cal], 1, "calibration") + label(control[:n_cal], 0, "calibration")
    test = label(selected[n_cal:n_cal+n_test], 1, "test") + label(control[n_cal:n_cal+n_test], 0, "test")
    return calibration, test


def source_rows(source: dict, input_jsonl: str | None):
    if source.get("local_jsonl") and not input_jsonl:
        from .local_source import validate
        input_jsonl = str(validate(source)[0])
    if input_jsonl:
        with open(input_jsonl, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        from datasets import load_dataset
        kwargs = {"split": source["split"], "revision": source["revision"], "streaming": True}
        if source.get("subset"):
            kwargs["name"] = source["subset"]
        yield from load_dataset(source["dataset"], **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input-jsonl", help="Local fixture or predownloaded source with the same schema")
    args = parser.parse_args()
    cfg = config(args.config)
    out = artifacts(cfg) / "prepared"
    if (out / "manifest.json").exists():
        raise FileExistsError(f"Prepared data already exists: {out}; use a new artifact root")
    out.mkdir(parents=True, exist_ok=True)
    secret_path = out / "selection_key.hex"
    if secret_path.exists():
        raise FileExistsError(secret_path)
    key = secrets.token_bytes(32)
    descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(key.hex() + "\n")
    source = dict(cfg["source"])
    filter_manifest = None
    if source.get("local_jsonl") and not args.input_jsonl:
        from .local_source import validate
        _, filter_manifest = validate(source)
    elif not args.input_jsonl:
        from huggingface_hub import HfApi
        source["revision"] = HfApi().dataset_info(source["dataset"], revision=source["revision"]).sha
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(next(iter(cfg["families"].values())))
    def window(row):
        try:
            return fit_model_window([row], tokenizer, cfg["training"]["max_length"])[0]
        except ValueError as exc:
            if "fewer than eight response tokens" in str(exc):
                return None
            raise
    # Define unique usable windows before sampling; truncation must not create
    # the same audit text in both S and S-prime.
    sampler = source.get("sampling_method", "keyed_rank")
    samplers = {"keyed_rank": choose, "global_shuffle": choose_global_shuffle}
    if sampler not in samplers:
        raise ValueError(f"Unknown sampling_method: {sampler}")
    selected, control, counts = samplers[sampler](source_rows(source, args.input_jsonl), key,
                                                  source, source["n_per_arm"], transform=window)
    counts["sampling_method"] = sampler
    calibration, test = split_audit(selected, control, source["calibration_per_arm"],
                                    source["test_per_arm"])
    if {r["id"] for r in selected} & {r["id"] for r in control}:
        raise AssertionError("S/S-prime overlap")
    write_jsonl(out / "s_train.jsonl", selected)
    write_jsonl(out / "s_prime.jsonl", control)
    write_jsonl(out / "calibration.jsonl", calibration)
    write_jsonl(out / "test.jsonl", test)
    write_json(out / "manifest.json", {"config_fingerprint": fingerprint(cfg),
               "source": source, "input_jsonl": args.input_jsonl, "counts": counts,
               "source_filter": filter_manifest,
               "s_digest": fingerprint([r["id"] for r in selected]),
               "s_prime_digest": fingerprint([r["id"] for r in control])})
    print(f"Prepared {counts['s_count']} S and {counts['s_prime_count']} S-prime rows in {out}")


if __name__ == "__main__":
    main()

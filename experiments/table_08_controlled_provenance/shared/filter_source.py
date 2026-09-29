"""Build a CodeSearchNet pool with normalized prompt/code matches removed."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from .common import write_json
from .overlap_audit import parquet_rows
from .prepare import canonical_row

LIMITS = {"max_prompt_chars": 300, "max_response_chars": 1100}
REVISION = "a4ee8053e119ef04ee3d491ffef05bcf2725ae82"


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split()).replace("\x00", "")


def anchor(text: str) -> str:
    width = min(120, len(text))
    start = (len(text) - width) // 2
    return text[start:start + width]


def matches(patterns: dict[str, list[tuple[int, str]]], pile_paths: list[Path], label: str) -> set[int]:
    import ahocorasick
    automaton = ahocorasick.Automaton()
    buckets = list(patterns.values())
    for index, value in enumerate(patterns):
        automaton.add_word(value, index)
    automaton.make_automaton()
    print(f"Checking {label}: {len(buckets)} distinct anchors", flush=True)
    found: set[int] = set()
    documents = 0
    for raw in parquet_rows(pile_paths, ["text"]):
        document = normalize(raw["text"] or "")
        for bucket in {value for _, value in automaton.iter(document)}:
            found.update(index for index, full in buckets[bucket] if full in document)
        documents += 1
        if documents % 50000 == 0:
            print(f"{label}: {documents} documents, {len(found)} matching rows", flush=True)
    if documents != 416000:
        raise ValueError(f"Unexpected Pile reference size: {documents}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codesearchnet-dir", required=True, type=Path)
    parser.add_argument("--pile-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    csn_paths = sorted(args.codesearchnet_dir.glob("python/train-*.parquet"))
    pile_paths = sorted(args.pile_dir.glob("github/partial/train/*.parquet"))
    if len(csn_paths) != 3 or len(pile_paths) != 10:
        raise FileNotFoundError("Need three CodeSearchNet and ten Pile GitHub shards")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite filtered pool: {args.output_dir}")

    candidates: list[tuple[str, str, str]] = []
    prompt_patterns: dict[str, list[tuple[int, str]]] = defaultdict(list)
    response_patterns: dict[str, list[tuple[int, str]]] = defaultdict(list)
    pair_hashes = set()
    source_rows = duplicates = 0
    columns = ["func_documentation_string", "func_code_string", "repository_name", "func_code_url"]
    for raw in parquet_rows(csn_paths, columns):
        source_rows += 1
        row = canonical_row(raw, LIMITS)
        if row is None:
            continue
        prompt = normalize(raw["func_documentation_string"])
        response = normalize(raw["func_code_string"])
        pair_hash = hashlib.sha256(json.dumps([prompt, response]).encode()).digest()
        if pair_hash in pair_hashes:
            duplicates += 1
            continue
        pair_hashes.add(pair_hash)
        index = len(candidates)
        candidates.append((row["id"], prompt, response))
        prompt_patterns[anchor(prompt)].append((index, prompt))
        response_patterns[anchor(response)].append((index, response))

    prompt_matches = matches(prompt_patterns, pile_paths, "prompts")
    del prompt_patterns
    response_matches = matches(response_patterns, pile_paths, "responses")
    excluded = prompt_matches | response_matches
    retained_ids = {row_id for index, (row_id, _, _) in enumerate(candidates) if index not in excluded}
    if len(retained_ids) < 40000:
        raise ValueError(f"Only {len(retained_ids)} survivors; need 40,000")

    args.output_dir.mkdir(parents=True)
    digest = hashlib.sha256()
    written = set()
    with (args.output_dir / "eligible.jsonl").open("wb") as handle:
        for raw in parquet_rows(csn_paths, columns):
            row = canonical_row(raw, LIMITS)
            if row is None or row["id"] not in retained_ids or row["id"] in written:
                continue
            payload = (json.dumps(raw, ensure_ascii=False) + "\n").encode()
            handle.write(payload)
            digest.update(payload)
            written.add(row["id"])
    if written != retained_ids:
        raise AssertionError("Filtered file differs from verified survivor IDs")
    with (args.output_dir / "excluded.jsonl").open("w", encoding="utf-8") as handle:
        for index in sorted(excluded):
            handle.write(json.dumps({"id": candidates[index][0],
                                     "prompt_match": index in prompt_matches,
                                     "response_match": index in response_matches}) + "\n")
    report = {
        "dataset": "claudios/code_search_net", "revision": REVISION,
        "subset": "python", "split": "train", "limits": LIMITS,
        "normalization": "NFKC + casefold + remove all whitespace and NUL",
        "exclusion_rule": "remove row if its complete normalized prompt or complete normalized response is a substring of any checked Pile document",
        "reference_scope": "416000 documents from official Pile GitHub partial Parquet shards; full Pile and Pile-CC not checked",
        "source_rows": source_rows, "eligible_unique_pairs": len(candidates),
        "normalized_duplicate_pairs_removed": duplicates,
        "prompt_matches": len(prompt_matches), "response_matches": len(response_matches),
        "excluded_either": len(excluded), "retained_pairs": len(retained_ids),
        "retained_detected_matches": 0, "eligible_sha256": digest.hexdigest(),
    }
    write_json(args.output_dir / "manifest.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

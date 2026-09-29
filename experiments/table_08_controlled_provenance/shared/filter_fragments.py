"""Filter CodeSearchNet using shared fragments, with a bounded matcher index."""
from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

from .common import write_json
from .overlap_audit import parquet_rows
from .prepare import canonical_row

LIMITS = {"max_prompt_chars": 300, "max_response_chars": 1100}
REVISION = "a4ee8053e119ef04ee3d491ffef05bcf2725ae82"


def normalize(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split()).replace("\x00", "")


def fragments(text: str, width: int = 80) -> set[str]:
    text = normalize(text)
    if not text:
        return set()
    if len(text) <= width:
        return {text}
    return {text[i:i + width] for i in range(0, len(text) - width + 1, width)} | {text[-width:]}


def matched_patterns(patterns: list[str], text: str) -> set[int]:
    """Small public helper for verifying normalization and overlapping matches."""
    target = normalize(text)
    try:
        from ahocorasick_rs import AhoCorasick, Implementation
    except ImportError:
        # Keep the verification helper usable in the training environment;
        # rebuilding the optional fragment index still requires ahocorasick-rs.
        return {index for index, pattern in enumerate(patterns)
                if normalize(pattern) in target}
    matcher = AhoCorasick(patterns, implementation=Implementation.ContiguousNFA, store_patterns=False)
    return {m[0] for m in matcher.find_matches_as_indexes(target, overlapping=True)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codesearchnet-dir", required=True, type=Path)
    parser.add_argument("--pile-compact-lines", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    from ahocorasick_rs import AhoCorasick, Implementation
    paths = sorted(args.codesearchnet_dir.glob("python/train-*.parquet"))
    if len(paths) != 3 or not args.pile_compact_lines.is_file():
        raise FileNotFoundError("Need all three source shards and checked Pile sample")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    patterns: dict[str, int] = {}
    buckets: list[list[int]] = []
    ids: list[str] = []
    pair_hashes = set()
    duplicates = source_count = 0
    columns = ["func_documentation_string", "func_code_string", "repository_name", "func_code_url"]
    for raw in parquet_rows(paths, columns):
        source_count += 1
        row = canonical_row(raw, LIMITS)
        if row is None:
            continue
        fields = (raw["func_documentation_string"], raw["func_code_string"])
        pair_hash = hashlib.sha256(json.dumps([normalize(x) for x in fields]).encode()).digest()
        if pair_hash in pair_hashes:
            duplicates += 1
            continue
        pair_hashes.add(pair_hash)
        index = len(ids)
        ids.append(row["id"])
        for field, text in enumerate(fields):
            for block in fragments(text):
                if block not in patterns:
                    patterns[block] = len(buckets)
                    buckets.append([])
                buckets[patterns[block]].append(index * 2 + field)
    pattern_list = list(patterns)
    del patterns, pair_hashes
    flags = bytearray(len(ids))
    seen_buckets = set()
    reference_hash = hashlib.sha256()
    batch_size = 150000
    print(f"{len(ids)} candidate pairs; {len(buckets)} distinct fragments in {(len(buckets) + batch_size - 1) // batch_size} bounded batches", flush=True)
    for offset in range(0, len(pattern_list), batch_size):
        matcher = AhoCorasick(pattern_list[offset:offset + batch_size],
                              implementation=Implementation.ContiguousNFA, store_patterns=False)
        documents = 0
        with args.pile_compact_lines.open(encoding="utf-8") as handle:
            for line in handle:
                documents += 1
                if offset == 0:
                    reference_hash.update(line.encode("utf-8"))
                hits = {offset + m[0] for m in matcher.find_matches_as_indexes(
                    normalize(line), overlapping=True)} - seen_buckets
                seen_buckets.update(hits)
                for bucket in hits:
                    for code in buckets[bucket]:
                        flags[code // 2] |= 1 << (code % 2)
                if documents % 100000 == 0:
                    print(f"Fragment batch {offset // batch_size + 1}: {documents} reference documents", flush=True)
        del matcher
        if documents != 416000:
            raise ValueError(f"Unexpected reference size: {documents}")
        print(f"Finished batch {offset // batch_size + 1}: {sum(bool(x) for x in flags)} rows excluded", flush=True)
    retained = {row_id for row_id, flag in zip(ids, flags) if not flag}
    if len(retained) < 40000:
        raise ValueError(f"Only {len(retained)} survivors; need 40,000")
    args.output_dir.mkdir(parents=True)
    digest = hashlib.sha256()
    written = set()
    with (args.output_dir / "eligible.jsonl").open("wb") as handle:
        for raw in parquet_rows(paths, columns):
            row = canonical_row(raw, LIMITS)
            if row is None or row["id"] not in retained or row["id"] in written:
                continue
            payload = (json.dumps(raw, ensure_ascii=False) + "\n").encode("utf-8")
            handle.write(payload)
            digest.update(payload)
            written.add(row["id"])
    if written != retained:
        raise AssertionError("Written pool differs from verified survivor IDs")
    with (args.output_dir / "excluded.jsonl").open("w", encoding="utf-8") as handle:
        for row_id, flag in zip(ids, flags):
            if flag:
                handle.write(json.dumps({"id": row_id, "prompt_match": bool(flag & 1),
                                         "response_match": bool(flag & 2)}) + "\n")
    report = {
        "dataset": "claudios/code_search_net", "revision": REVISION, "subset": "python", "split": "train",
        "limits": LIMITS, "normalization": "NFKC + casefold + remove whitespace and NUL",
        "rule": "Reject either field sharing an indexed 80-character block; shorter fields match in full; stride 80 plus suffix",
        "guaranteed_detected_shared_span_chars": 159,
        "reference_scope": "416000 documents from official Pile GitHub partial shards; full Pile and Pile-CC not checked",
        "reference_documents": documents, "reference_compact_sha256": reference_hash.hexdigest(),
        "source_rows": source_count, "eligible_unique_pairs": len(ids), "normalized_duplicate_pairs_removed": duplicates,
        "prompt_matches": sum(bool(x & 1) for x in flags), "response_matches": sum(bool(x & 2) for x in flags),
        "excluded_either": sum(bool(x) for x in flags), "retained_pairs": len(retained),
        "retained_matches_under_this_rule": 0, "local_sha256": digest.hexdigest(),
        # Compatibility with the existing full-field pool validator.
        "retained_detected_matches": 0, "eligible_sha256": digest.hexdigest(),
    }
    write_json(args.output_dir / "manifest.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

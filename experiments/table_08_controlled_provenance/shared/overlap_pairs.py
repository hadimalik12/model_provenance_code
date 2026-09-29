"""Count normalized CodeSearchNet prompt/response overlap with Pile GitHub.

Counts both fields anywhere in the sampled corpus (a liberal pair-overlap
upper bound for this corpus) and both fields in the same Pile document.
Repository metadata is intentionally ignored. This is not a full-Pile audit.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .overlap_audit import normalized, parquet_rows


def anchor_for(full: str) -> str:
    width = min(120, len(full))
    start = (len(full) - width) // 2
    return full[start:start + width]


def find_full_matches(patterns: dict[str, list[tuple[int, str]]], docs: Path,
                      label: str) -> set[int]:
    try:
        import ahocorasick
    except ImportError as exc:
        raise RuntimeError("This optional audit requires pyahocorasick") from exc

    automaton = ahocorasick.Automaton()
    buckets = list(patterns.values())
    for index, anchor in enumerate(patterns):
        automaton.add_word(anchor, index)
    automaton.make_automaton()
    print(f"Built {label} automaton with {len(buckets)} distinct anchors", flush=True)
    matched_ids: set[int] = set()
    with docs.open(encoding="utf-8") as handle:
        for document in handle:
            matched_anchors = {index for _, index in automaton.iter(document)}
            for index in matched_anchors:
                matched_ids.update(row_id for row_id, full in buckets[index] if full in document)
    print(f"Verified {len(matched_ids)} full {label} matches", flush=True)
    return matched_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codesearchnet-dir", type=Path, required=True)
    parser.add_argument("--pile-compact-lines", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    csn_files = sorted(args.codesearchnet_dir.glob("python/train-*.parquet"))
    if len(csn_files) != 3 or not args.pile_compact_lines.exists():
        raise FileNotFoundError("Expected all three CodeSearchNet train shards and compact Pile lines")

    prompts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    responses: dict[str, list[tuple[int, str]]] = defaultdict(list)
    eligible = 0
    total_rows = 0
    prompt_embedded_in_response = 0
    unique_pairs: set[tuple[str, str]] = set()
    for row in parquet_rows(csn_files, ["func_documentation_string", "func_code_string"]):
        total_rows += 1
        prompt = (row["func_documentation_string"] or "").strip()
        response = (row["func_code_string"] or "").strip()
        if not 16 <= len(prompt) <= 300 or not 32 <= len(response) <= 1100:
            continue
        compact_prompt, compact_response = normalized(prompt), normalized(response)
        if not compact_prompt or not compact_response:
            continue
        prompts[anchor_for(compact_prompt)].append((eligible, compact_prompt))
        responses[anchor_for(compact_response)].append((eligible, compact_response))
        prompt_embedded_in_response += compact_prompt in compact_response
        unique_pairs.add((compact_prompt, compact_response))
        eligible += 1
    print(f"Loaded {eligible} eligible pairs from {total_rows} train rows", flush=True)

    response_ids = find_full_matches(responses, args.pile_compact_lines, "responses")
    prompt_ids = find_full_matches(prompts, args.pile_compact_lines, "prompts")
    if prompt_embedded_in_response == eligible:
        # CodeSearchNet extracts the prompt from the function's docstring.
        # Thus a full response match necessarily contains its prompt in the
        # same Pile document. This is stronger than independent anchor hits.
        prompt_ids.update(response_ids)
        same_document_ids = response_ids
    else:
        raise AssertionError("Prompt is not embedded in every response; same-document logic must be revised")

    both_anywhere = prompt_ids & response_ids
    report = {
        "scope": "CodeSearchNet Python train rows eligible under old pilot filters versus 416,000 Pile GitHub partial documents; repository unrestricted; not full Pile or Pile-CC",
        "normalization": "strip field ends, then remove all whitespace; preserve case and punctuation",
        "codesearchnet_train_rows": total_rows,
        "eligible_pairs": eligible,
        "unique_normalized_pairs": len(unique_pairs),
        "pairs_whose_prompt_is_embedded_in_response": prompt_embedded_in_response,
        "prompt_found_anywhere": len(prompt_ids),
        "response_found_anywhere": len(response_ids),
        "both_fields_found_anywhere_maximum_pair_overlap_in_sample": len(both_anywhere),
        "both_fields_found_in_same_pile_document": len(same_document_ids),
        "prompt_only": len(prompt_ids - response_ids),
        "response_only": len(response_ids - prompt_ids),
        "neither": eligible - len(prompt_ids | response_ids),
        "matching_method": "Aho-Corasick enumerates overlapping anchors; each candidate is verified against the complete normalized field",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

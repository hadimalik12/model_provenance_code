"""Check full InstructCoder code fields against a downloaded Pile GitHub sample.

Ripgrep finds a distinctive interior anchor in whitespace-normalized Pile
documents. Each candidate is then verified using the *entire* normalized code
field, so anchor hits alone are never counted as full-code overlap.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def compact(text: str) -> str:
    return "".join(text.split()).replace("\x00", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pile-dir", type=Path, required=True)
    parser.add_argument("--instructcoder-json", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pile_files = sorted(args.pile_dir.glob("github/partial/train/*.parquet"))
    if not pile_files:
        raise FileNotFoundError("No downloaded Pile GitHub shards")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    docs_file = args.work_dir / "pile_compact_lines.txt"
    anchors_file = args.work_dir / "instructcoder_anchors.txt"
    hits_file = args.work_dir / "anchor_hits.txt"

    with args.instructcoder_json.open(encoding="utf-8") as handle:
        examples = json.load(handle)
    anchors: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for index, row in enumerate(examples):
        for field in ("input", "output"):
            full = compact(row.get(field) or "")
            if len(full) < 40:
                continue
            anchor_length = min(120, len(full))
            start = (len(full) - anchor_length) // 2
            anchors[full[start:start + anchor_length]].append((index, field, full))
    with anchors_file.open("w", encoding="utf-8") as handle:
        for anchor in anchors:
            if "\n" in anchor:
                raise AssertionError("Compact anchor contains newline")
            handle.write(anchor + "\n")

    document_count = 0
    with docs_file.open("w", encoding="utf-8") as handle:
        for path in pile_files:
            reader = pq.ParquetFile(path)
            for batch in reader.iter_batches(batch_size=256, columns=["text"]):
                for row in batch.to_pylist():
                    handle.write(compact(row["text"] or "") + "\n")
                    document_count += 1

    with hits_file.open("w", encoding="utf-8") as handle:
        result = subprocess.run(["rg", "--text", "--fixed-strings", "--only-matching",
                                 "--line-number", "--no-filename", "--file", str(anchors_file),
                                 str(docs_file)], stdout=handle, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(f"rg failed with code {result.returncode}")

    candidates_by_line: dict[int, set[str]] = defaultdict(set)
    with hits_file.open(encoding="utf-8") as handle:
        for line in handle:
            number, anchor = line.rstrip("\n").split(":", 1)
            candidates_by_line[int(number)].add(anchor)

    full_matches: set[tuple[int, str]] = set()
    examples_matched = []
    with docs_file.open(encoding="utf-8") as handle:
        for line_number, document in enumerate(handle, start=1):
            if line_number not in candidates_by_line:
                continue
            for anchor in candidates_by_line[line_number]:
                for index, field, full in anchors[anchor]:
                    if full in document:
                        full_matches.add((index, field))
                        if len(examples_matched) < 5:
                            examples_matched.append({"row": index, "field": field,
                                                     "code_prefix": full[:100]})

    input_matches = {index for index, field in full_matches if field == "input"}
    output_matches = {index for index, field in full_matches if field == "output"}
    report = {"scope": "All InstructCoder train rows versus the ten available converted Pile GitHub partial shards; exact full-code substring after removing whitespace, not full Pile-CC or full 95 GiB GitHub component",
              "instructcoder_train_rows": len(examples),
              "pile_github_documents": document_count,
              "anchors": len(anchors),
              "pile_documents_with_anchor_hit": len(candidates_by_line),
              "rows_with_full_input_code_match": len(input_matches),
              "rows_with_full_output_code_match": len(output_matches),
              "rows_with_either_full_code_match": len(input_matches | output_matches),
              "match_examples": examples_matched}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

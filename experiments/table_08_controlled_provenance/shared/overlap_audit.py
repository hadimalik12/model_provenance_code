"""Empirically compare CodeSearchNet functions with a downloaded Pile GitHub sample.

This is a source-selection audit, not a training job. It deliberately reports
the coverage of the Pile sample instead of extrapolating to the full corpus.
"""
from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def normalized(text: str) -> str:
    return "".join(text.split())


def repository(value: str) -> str:
    return value.strip().removesuffix(".git").lower()


def parquet_rows(paths: list[Path], columns: list[str]):
    for path in paths:
        reader = pq.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pile-dir", type=Path, required=True)
    parser.add_argument("--codesearchnet-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pile_files = sorted(args.pile_dir.glob("github/partial/train/*.parquet"))
    csn_files = sorted(args.codesearchnet_dir.glob("python/train-*.parquet"))
    if not pile_files or not csn_files:
        raise FileNotFoundError("Expected downloaded Pile GitHub and CodeSearchNet Python shards")

    by_repo: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    eligible_ids: set[int] = set()
    total_csn = 0
    for row in parquet_rows(csn_files, ["repository_name", "func_code_string", "func_documentation_string"]):
        code = row["func_code_string"] or ""
        prompt = row["func_documentation_string"] or ""
        repo = repository(row["repository_name"] or "")
        if not repo or not code.strip():
            total_csn += 1
            continue
        by_repo[repo].append((total_csn, code.strip(), normalized(code)))
        if 16 <= len(prompt.strip()) <= 300 and 32 <= len(code.strip()) <= 1100:
            eligible_ids.add(total_csn)
        total_csn += 1

    seen_repos: set[str] = set()
    exact_ids: set[int] = set()
    whitespace_ids: set[int] = set()
    pile_docs = 0
    pile_python_docs = 0
    examples = []
    for row in parquet_rows(pile_files, ["text", "meta"]):
        pile_docs += 1
        try:
            meta = ast.literal_eval(row["meta"] or "{}")
        except (ValueError, SyntaxError):
            continue
        repo = repository(str(meta.get("repo_name") or ""))
        if str(meta.get("repo_language") or "").lower() == "python":
            pile_python_docs += 1
        candidates = by_repo.get(repo)
        if not candidates:
            continue
        seen_repos.add(repo)
        text = row["text"] or ""
        compact_text = None
        for row_id, code, compact_code in candidates:
            if row_id in exact_ids:
                continue
            if code in text:
                exact_ids.add(row_id)
                whitespace_ids.add(row_id)
                if len(examples) < 5:
                    examples.append({"repository": repo, "code_prefix": code[:100]})
            elif row_id not in whitespace_ids and len(compact_code) >= 40:
                if compact_text is None:
                    compact_text = normalized(text)
                if compact_code in compact_text:
                    whitespace_ids.add(row_id)

    repo_ids = {row_id for repo in seen_repos for row_id, _, _ in by_repo[repo]}
    report = {
        "scope": "official EleutherAI/pile refs/convert/parquet github/partial/train shards, not full 95 GiB GitHub component or full Pile-CC",
        "pile_shards": [p.name for p in pile_files],
        "pile_documents": pile_docs,
        "pile_python_language_documents": pile_python_docs,
        "codesearchnet_shards": [p.name for p in csn_files],
        "codesearchnet_rows": total_csn,
        "codesearchnet_old_pilot_eligible_rows": len(eligible_ids),
        "codesearchnet_rows_with_repo_in_pile_sample": len(repo_ids),
        "codesearchnet_eligible_rows_with_repo_in_pile_sample": len(repo_ids & eligible_ids),
        "codesearchnet_exact_function_substring_matches": len(exact_ids),
        "codesearchnet_eligible_exact_function_substring_matches": len(exact_ids & eligible_ids),
        "codesearchnet_whitespace_insensitive_function_matches": len(whitespace_ids),
        "codesearchnet_eligible_whitespace_insensitive_matches": len(whitespace_ids & eligible_ids),
        "match_examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""Validate the immutable local filtered pool before preparation or submission."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import ROOT, artifacts, config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(source: dict) -> tuple[Path, dict]:
    path = ROOT / source["local_jsonl"]
    manifest_path = ROOT / source["filter_manifest"]
    manifest = json.loads(manifest_path.read_text())
    actual_sha256 = sha256_file(path)
    if actual_sha256 != manifest["eligible_sha256"] or source.get("local_sha256", actual_sha256) != actual_sha256:
        raise ValueError("Filtered source checksum mismatch; refusing to use changed data")
    for name in ("dataset", "revision", "subset", "split"):
        if source.get(name) != manifest.get(name):
            raise ValueError(f"Filtered source provenance mismatch: {name}")
    if manifest["limits"] != {name: source[name] for name in ("max_prompt_chars", "max_response_chars")}:
        raise ValueError("Eligibility filters differ from the checked source")
    if manifest["retained_detected_matches"] != 0 or "416000 documents" not in manifest["reference_scope"]:
        raise ValueError("Source filtering is incomplete")
    if manifest["retained_pairs"] < 2 * source["n_per_arm"]:
        raise ValueError("Insufficient filtered records for S and S-prime")
    return path, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--submission", action="store_true")
    args = parser.parse_args()
    cfg = config(args.config)
    _, manifest = validate(cfg["source"])
    if args.submission:
        root = artifacts(cfg)
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {root}; use a new artifact_root")
        expected = {"1b": ["hh_sft"], "1_4b": ["helpful_sft", "tulu_sft"]}
        if cfg["profiles"] != expected or set(cfg["families"]) != set(expected):
            raise ValueError("submit_all.sh supports the configured 1B HH and 1.4B helpful/Tulu branches only")
    print(f"Filtered source verified: {manifest['retained_pairs']} candidates; reference scope: {manifest['reference_scope']}")


if __name__ == "__main__":
    main()

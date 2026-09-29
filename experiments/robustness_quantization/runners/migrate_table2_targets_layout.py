"""Migrate completed Table 2 INT8 artifacts to the shared experiment layout.

The legacy layout stored one complete mini-experiment per target.  The shared
layout mirrors Table 5: prepared data are shared at the experiment root, while
target and representation are dimensions below ``scores/`` and ``results/``.

Run a dry run first, then pass ``--apply`` after it reports the expected six
targets. The migration validates that every legacy target uses identical shared
prepared splits before removing duplicate copies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default="artifacts/robustness_quantization/table2_targets_seed0",
    )
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_shared_prepared(source_roots: list[Path]) -> None:
    for dataset in ("mimir_github", "nonmember_control"):
        for filename in ("train.jsonl", "test.jsonl"):
            paths = [root / "prepared" / dataset / filename for root in source_roots]
            if not all(path.is_file() for path in paths):
                raise SystemExit(f"Missing prepared split: {dataset}/{filename}")
            hashes = {digest(path) for path in paths}
            if len(hashes) != 1:
                raise SystemExit(
                    f"Refusing migration: {dataset}/{filename} differs between targets."
                )


def move(source: Path, destination: Path) -> None:
    if destination.exists():
        raise SystemExit(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root).resolve()
    source_roots = sorted(
        path for path in root.iterdir() if path.is_dir() and (path / "run_spec.json").is_file()
    )
    if not source_roots:
        raise SystemExit("No legacy per-target artifact directories found.")

    verify_shared_prepared(source_roots)
    targets = []
    for source_root in source_roots:
        spec = json.loads((source_root / "run_spec.json").read_text())
        targets.append((source_root, spec.get("target_model"), source_root.name))
    print(f"Validated {len(targets)} targets with identical prepared splits:")
    for _, target, slug in targets:
        print(f"  {slug}: {target}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to migrate these artifacts.")
        return

    canonical = source_roots[0]
    move(canonical / "prepared", root / "prepared")
    allowed_children = {"config.lock.yaml", "prepared", "reports", "results", "run_spec.json", "scores"}
    for source_root, _, target_slug in targets:
        unexpected = {path.name for path in source_root.iterdir()} - allowed_children
        if unexpected:
            raise SystemExit(f"Unexpected files in {source_root}: {sorted(unexpected)}")
        for control in ("main", "nonmember_control"):
            source_scores = source_root / "scores" / control
            source_results = source_root / "results" / control
            if source_scores.exists():
                move(source_scores, root / "scores" / control / target_slug)
            if source_results.exists():
                move(source_results, root / "results" / control / target_slug)
        move(source_root / "config.lock.yaml", root / "config_locks" / f"{target_slug}.yaml")
        move(source_root / "run_spec.json", root / "run_specs" / f"{target_slug}.json")
        detail_root = root / "reports" / "details"
        for suffix in ("csv", "md"):
            report = source_root / "reports" / f"quantization_robustness.{suffix}"
            if report.exists():
                move(report, detail_root / f"{target_slug}.{suffix}")
        shutil.rmtree(source_root)

    print(f"Migrated artifacts into {root}")


if __name__ == "__main__":
    main()

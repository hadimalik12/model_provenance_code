"""Compile the submitted Appendix Table 8 public-ViT CLV audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TARGETS = (
    ("parent", "google/vit-base-patch16-224", "positive reference"),
    ("pets", "ISxOdin/vit-base-oxford-iiit-pets", "yes"),
    ("cifar10", "MatanBT/vit-base-patch16-224-cifar10", "yes"),
    ("cifar100", "MatanBT/vit-base-patch16-224-cifar100", "yes"),
    ("eurosat", "Abhiram4/vit-base-patch16-224-finetuned-eurosat", "yes"),
    ("celeba", "MatanBT/vit-base-patch16-224-celeba-smiling", "yes"),
    ("har", "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224", "yes"),
    ("random_init", "ViT-B/16-224 random initialization", "no"),
)
SCORE = "clv_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default="artifacts/vit_provenance/public_derivatives_seed0",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def metrics(blob: dict) -> dict:
    return next(
        (
            result.get("test", {})
            for result in blob.get("main_results", [])
            if result.get("score_name") == SCORE
        ),
        {},
    )


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root).resolve()
    rows = []
    for slug, model, expected in TARGETS:
        result = metrics(load(root / "results" / slug / "results.json"))
        advantage = result.get("shard_advantage")
        count = result.get("n_pos")
        threshold = math.sqrt(math.log(40) / count) if count else None
        detected = advantage is not None and threshold is not None and advantage > threshold
        rows.append(
            {
                "target_model": model,
                "signed_shard_advantage": advantage,
                "threshold": threshold,
                "decision": "reject" if detected else "fail_to_reject" if advantage is not None else "missing",
                "expected_witness": expected,
            }
        )

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "public_vit_derivatives.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Appendix Table 8: public ViT derivatives",
        "",
        "| Target model | Signed advantage | Threshold | Decision | Expected witness |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        advantage = "--" if row["signed_shard_advantage"] is None else f'{row["signed_shard_advantage"]:.4f}'
        threshold = "--" if row["threshold"] is None else f'{row["threshold"]:.4f}'
        lines.append(
            f'| {row["target_model"]} | {advantage} | {threshold} '
            f'| {row["decision"]} | {row["expected_witness"]} |'
        )
    markdown_path = report_dir / "public_vit_derivatives.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()

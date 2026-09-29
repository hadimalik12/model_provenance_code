"""Compile FP16-vs-INT8 Pythia provenance robustness results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCORES = ("min_k_20_logprob",)
VARIANTS = (
    ("parent_fp16", "Parent FP16"),
    ("target_fp16", "Target FP16"),
    ("target_int8", "Target INT8"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default="artifacts/robustness_quantization/int8_pythia1_4b_lamini_seed0",
    )
    parser.add_argument("--target-slug", required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open() as handle:
        return json.load(handle)


def entry(blob: dict, score: str) -> dict:
    return next(
        (item for item in blob.get("main_results", []) if item.get("score_name") == score),
        {},
    )


def shuffled(blob: dict, score: str) -> dict:
    return next(
        (
            item
            for item in blob.get("shuffled_label_control") or []
            if item.get("score_name") == score
        ),
        {},
    )


def threshold(metrics: dict) -> float | None:
    m = metrics.get("n_pos")
    return math.sqrt(math.log(40) / m) if m else None


def verdict(advantage, gamma) -> str:
    if advantage is None or gamma is None:
        return "missing"
    return "Reject H0" if advantage > gamma else "Fail to reject"


def fmt(value) -> str:
    return "--" if value is None else f"{value:.3f}"


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root).resolve()
    namespace = args.target_slug
    parent_blob = load(root / "results/main" / namespace / "parent_fp16/results.json")
    parent_metrics = {
        score: entry(parent_blob, score).get("test", {}) for score in SCORES
    }
    target_fp16_blob = load(root / "results/main" / namespace / "target_fp16/results.json")
    target_fp16_metrics = {
        score: entry(target_fp16_blob, score).get("test", {}) for score in SCORES
    }

    rows = []
    for variant, label in VARIANTS:
        main_blob = load(root / "results/main" / namespace / variant / "results.json")
        control_blob = load(root / "results/nonmember_control" / namespace / variant / "results.json")
        manifest = load(root / "scores/main" / namespace / variant / "manifest.json")
        for score in SCORES:
            metrics = entry(main_blob, score).get("test", {})
            control = entry(control_blob, score).get("test", {})
            shuf = shuffled(main_blob, score)
            advantage = metrics.get("shard_advantage")
            gamma = threshold(metrics)
            parent_advantage = parent_metrics[score].get("shard_advantage")
            fp16_advantage = target_fp16_metrics[score].get("shard_advantage")
            rows.append(
                {
                    "variant": variant,
                    "label": label,
                    "score": score,
                    "quantization": manifest.get("quantization"),
                    "is_loaded_in_8bit": manifest.get("is_loaded_in_8bit"),
                    "advantage": advantage,
                    "auc": metrics.get("auc"),
                    "gamma_0.05": gamma,
                    "verdict": verdict(advantage, gamma),
                    "retention_vs_parent": (
                        advantage / parent_advantage
                        if advantage is not None and parent_advantage
                        else None
                    ),
                    "delta_vs_target_fp16": (
                        advantage - fp16_advantage
                        if advantage is not None and fp16_advantage is not None
                        else None
                    ),
                    "nonmember_control_advantage": control.get("shard_advantage"),
                    "nonmember_control_gamma_0.05": threshold(control),
                    "nonmember_control_verdict": verdict(
                        control.get("shard_advantage"), threshold(control)
                    ),
                    "shuffled_advantage": shuf.get("test_advantage"),
                }
            )

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    details_dir = report_dir / "details"
    details_dir.mkdir(parents=True, exist_ok=True)
    csv_path = details_dir / f"{namespace}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Pythia provenance robustness to LLM.int8 quantization",
        "",
        "| Variant | Score | INT8 verified | Advantage | γ₀.₀₅ | Verdict | Retention vs parent | Δ vs target FP16 | NM control | NM verdict | Shuffled |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['score']} | {row['is_loaded_in_8bit']} "
            f"| {fmt(row['advantage'])} | {fmt(row['gamma_0.05'])} | {row['verdict']} "
            f"| {fmt(row['retention_vs_parent'])} | {fmt(row['delta_vs_target_fp16'])} "
            f"| {fmt(row['nonmember_control_advantage'])} "
            f"| {row['nonmember_control_verdict']} | {fmt(row['shuffled_advantage'])} |"
        )
    markdown_path = details_dir / f"{namespace}.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()

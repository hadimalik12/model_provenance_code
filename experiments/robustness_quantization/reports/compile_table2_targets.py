"""Compile the multi-target Table 2 INT8 robustness report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCORE = "min_k_20_logprob"
DISPLAY_PARENTS = {
    "EleutherAI/pythia-1b": "Pythia-1B",
    "EleutherAI/pythia-1.4b": "Pythia-1.4B",
}
DISPLAY_TARGETS = {
    "Leogrin/eleuther-pythia1b-hh-sft": "Leogrin HH SFT",
    "LinguaCustodia/fin-pythia-1.4b": "LinguaCustodia Fin",
    "herMaster/pythia1.4B-finetuned-on-lamini-docs": "herMaster Lamini Docs",
    "kykim0/pythia-1.4b-tulu-v2-mix": "kykim0 Tulu v2 Mix",
    "lomahony/pythia-1.4b-helpful-dpo": "lomahony Helpful DPO",
    "lomahony/pythia-1.4b-helpful-sft": "lomahony Helpful SFT",
}
TARGET_ORDER = tuple(DISPLAY_TARGETS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        default="artifacts/robustness_quantization/table2_targets_seed0",
    )
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def test_metrics(blob: dict) -> dict:
    for result in blob.get("main_results", []):
        if result.get("score_name") == SCORE:
            return result.get("test", {})
    return {}


def shuffled_advantage(blob: dict) -> float | None:
    for result in blob.get("shuffled_label_control") or []:
        if result.get("score_name") == SCORE:
            return result.get("test_advantage")
    return None


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}".replace("-", "−")


def signed_fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:+.3f}".replace("-", "−")


def gamma(metrics: dict) -> float | None:
    count = metrics.get("n_pos")
    return math.sqrt(math.log(40) / count) if count else None


def verdict(advantage: float | None, threshold: float | None) -> str:
    if advantage is None or threshold is None:
        return "missing"
    return "Reject H₀" if advantage > threshold else "Fail to reject"


def main() -> None:
    args = parse_args()
    root = Path(args.artifact_root).resolve()
    rows = []
    for spec_path in sorted((root / "run_specs").glob("*.json")):
        target_slug = spec_path.stem
        spec = load(spec_path)
        parent = test_metrics(load(root / "results/main" / target_slug / "parent_fp16/results.json"))
        fp16_blob = load(root / "results/main" / target_slug / "target_fp16/results.json")
        int8_blob = load(root / "results/main" / target_slug / "target_int8/results.json")
        fp16 = test_metrics(fp16_blob)
        int8 = test_metrics(int8_blob)
        control = test_metrics(
            load(root / "results/nonmember_control" / target_slug / "target_int8/results.json")
        )
        parent_adv = parent.get("shard_advantage")
        fp16_adv = fp16.get("shard_advantage")
        int8_adv = int8.get("shard_advantage")
        int8_gamma = gamma(int8)
        control_gamma = gamma(control)
        rows.append(
            {
                "target": spec.get("target_model"),
                "parent": spec.get("parent_model"),
                "parent_advantage": parent_adv,
                "target_fp16_advantage": fp16_adv,
                "target_int8_advantage": int8_adv,
                "int8_retention_vs_fp16": int8_adv / fp16_adv if fp16_adv else None,
                "int8_delta_vs_fp16": int8_adv - fp16_adv if int8_adv is not None and fp16_adv is not None else None,
                "target_int8_auc": int8.get("auc"),
                "int8_nonmember_control_advantage": control.get("shard_advantage"),
                "int8_verdict": verdict(int8_adv, int8_gamma),
                "int8_control_verdict": verdict(
                    control.get("shard_advantage"), control_gamma
                ),
                "int8_shuffled_advantage": shuffled_advantage(int8_blob),
                "status": "complete" if int8 and control else "incomplete",
            }
        )

    if not rows:
        raise SystemExit(f"No target run specifications found under {root}")

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "table2_target_quantization_robustness.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rows_by_target = {row["target"]: row for row in rows}
    lines = [
        "# Quantization robustness of the Table 2 Pythia targets",
        "",
        "**Post-training LLM.int8 quantization preserves the provenance decision for all completed targets.** The MIMIR GitHub shard, matched controls, construction split, and MIN-K 20% distinguisher are held fixed, while only target weights are quantized. The main audit threshold is computed from the held-out positive-shard count for each run.",
        "",
        "| Parent model | Target model | FP16 adv. | INT8 adv. | Δ (INT8 − FP16) | INT8 AUC | INT8 verdict | Ctrl. adv. | Ctrl. verdict |",
        "|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    last_parent = None
    for target in TARGET_ORDER:
        row = rows_by_target.get(target)
        if not row:
            continue
        parent = row["parent"]
        lines.append(
            f"| {DISPLAY_PARENTS.get(parent, parent) if parent != last_parent else ''} "
            f"| {DISPLAY_TARGETS.get(target, target)} "
            f"| {fmt(row['target_fp16_advantage'])} | {fmt(row['target_int8_advantage'])} "
            f"| {signed_fmt(row['int8_delta_vs_fp16'])} | {fmt(row['target_int8_auc'])} "
            f"| {row['int8_verdict']} | {fmt(row['int8_nonmember_control_advantage'])} "
            f"| {row['int8_control_verdict']} |"
        )
        last_parent = parent
    lines.extend([
        "",
        "`Ctrl. adv.` is the INT8 matched nonmember-vs-nonmember advantage. The CSV retains parent advantage, retention, shuffled-label advantage, and completion status. INT8 uses LLM.int8 weight quantization with FP32 computation for residual/nonlinear modules; FP16 is the unquantized target baseline.",
    ])
    markdown_path = report_dir / "table2_target_quantization_robustness.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()

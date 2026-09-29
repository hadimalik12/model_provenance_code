"""Compile the local vanilla Pythia GitHub audit into CSV and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCORES = ("min_k_20_logprob", "mean_logprob")
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
    parser.add_argument("--artifact-root",
                        default="artifacts/table_02_target_provenance/local_vanilla")
    return parser.parse_args()


def load(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open() as handle:
        return json.load(handle)


def score_entry(blob: dict, score: str) -> dict:
    return next((item for item in blob.get("main_results", [])
                 if item.get("score_name") == score), {})


def shuffled_entry(blob: dict, score: str) -> dict:
    return next((item for item in blob.get("shuffled_label_control") or []
                 if item.get("score_name") == score), {})


def gamma(metrics: dict) -> float | None:
    m = metrics.get("n_pos")
    return math.sqrt(math.log(40) / m) if m else None


def value(item: dict, key: str):
    return item.get(key) if item else None


def fmt(value, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}".replace("-", "−")


def pct(value) -> str:
    return "--" if value is None else f"{100 * value:.1f}%"


def paper_table(rows: list[dict], score: str) -> list[str]:
    """Render one scorer in the same compact layout used by the manuscript."""
    score_rows = {
        row["model"]: row
        for row in rows
        if row["role"] == "derived_target" and row["score"] == score
    }
    lines = [
        "| Parent model | Target model | Ctrl. Acc. | Ctrl. adv. | Main Acc. | Main adv. | γ₀.₀₅ | Provenance verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    last_parent = None
    for target in TARGET_ORDER:
        row = score_rows.get(target)
        if not row:
            continue
        parent = row["parent_model"]
        lines.append(
            f"| {DISPLAY_PARENTS.get(parent, parent) if parent != last_parent else ''} "
            f"| {DISPLAY_TARGETS.get(target, target)} "
            f"| {pct(row['nonmember_control_accuracy'])} "
            f"| {fmt(row['nonmember_control_advantage'])} "
            f"| {pct(row['main_accuracy'])} | {fmt(row['main_advantage'])} "
            f"| {fmt(row['gamma_0.05'])} | {row['main_verdict']} |"
        )
        last_parent = parent
    return lines


def main() -> None:
    args = parse_args()
    root = (REPO_ROOT / args.artifact_root).resolve()
    rows: list[dict] = []
    for seed_dir in sorted(root.glob("seed_*")):
        try:
            seed = int(seed_dir.name.removeprefix("seed_"))
        except ValueError:
            continue
        main_root = seed_dir / "results" / "main"
        control_root = seed_dir / "results" / "nonmember_control"
        for result_file in sorted(main_root.glob("*/results.json")):
            model = result_file.parent.name.replace("__", "/")
            blob = load(result_file)
            is_parent = model in {"EleutherAI/pythia-1b", "EleutherAI/pythia-1.4b"}
            parent_model = model if is_parent else (
                "EleutherAI/pythia-1b" if "pythia1b" in model.lower() or "pythia-1b" in model.lower()
                else "EleutherAI/pythia-1.4b"
            )
            parent_file = main_root / parent_model.replace("/", "__") / "results.json"
            parent_blob = load(parent_file)
            control_blob = load(control_root / result_file.parent.name / "results.json")
            for score in SCORES:
                main = score_entry(blob, score).get("test", {})
                parent = score_entry(parent_blob, score).get("test", {})
                control = score_entry(control_blob, score).get("test", {})
                shuffled = shuffled_entry(blob, score)
                threshold = gamma(main)
                control_threshold = gamma(control)
                advantage = value(main, "shard_advantage")
                control_advantage = value(control, "shard_advantage")
                shuffled_advantage = shuffled.get("test_advantage")
                rows.append({
                    "seed": seed,
                    "parent_model": parent_model,
                    "model": model,
                    "role": "parent" if is_parent else "derived_target",
                    "score": score,
                    "parent_advantage": value(parent, "shard_advantage"),
                    "main_advantage": advantage,
                    "main_auc": value(main, "auc"),
                    "main_accuracy": value(main, "accuracy"),
                    "gamma_0.05": threshold,
                    "main_verdict": ("Reject H₀" if advantage is not None and threshold is not None and advantage > threshold
                                     else "Fail to reject" if threshold is not None else "missing"),
                    "nonmember_control_advantage": control_advantage,
                    "nonmember_control_accuracy": value(control, "accuracy"),
                    "nonmember_control_gamma_0.05": control_threshold,
                    "nonmember_control_verdict": (
                        "Reject H₀" if control_advantage is not None and control_threshold is not None and control_advantage > control_threshold
                        else "Fail to reject" if control_threshold is not None else "not run"
                    ),
                    "shuffled_calibration_advantage": shuffled_advantage,
                    "shuffled_calibration_verdict": (
                        "Reject H₀" if shuffled_advantage is not None and threshold is not None and shuffled_advantage > threshold
                        else "Fail to reject" if threshold is not None else "not run"
                    ),
                })

    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "local_vanilla_audit.csv"
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    min_k_rows = [row for row in rows if row["score"] == "min_k_20_logprob"]
    parent_rows = {
        row["model"]: row
        for row in min_k_rows
        if row["role"] == "parent"
    }
    lines = [
        "# Table 2: Provenance auditing of fine-tuned Pythia targets on MIMIR GitHub",
        "",
        "MIN-K 20% is calibrated on the construction split and evaluated on 400 held-out member and 400 held-out nonmember examples (`γ₀.₀₅ = 0.096` for this run). `Ctrl.` is the matched nonmember-vs-nonmember audit. The report also records shuffled-label calibration checks in the CSV.",
        "",
        *paper_table(rows, "min_k_20_logprob"),
        "",
        "## Parent shard-signal sanity check",
        "",
        "The parent-only audit is run before evaluating its fine-tuned targets, using the same pre-specified MIN-K 20% scorer.",
        "",
        "| Parent model | Main Acc. | Main adv. | AUC | γ₀.₀₅ | Shard signal |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for parent in DISPLAY_PARENTS:
        row = parent_rows.get(parent)
        if row:
            lines.append(
                f"| {DISPLAY_PARENTS[parent]} | {pct(row['main_accuracy'])} "
                f"| {fmt(row['main_advantage'])} | {fmt(row['main_auc'])} "
                f"| {fmt(row['gamma_0.05'])} | "
                f"{'Yes' if row['main_verdict'] == 'Reject H₀' else 'No'} |"
            )
    lines.extend([
        "",
        "## Secondary membership scorer: mean log-probability",
        "",
        "This fixed audit is repeated with mean log-probability, using the same splits and thresholding procedure. Comparing it with MIN-K20 shows how the provenance test accommodates different membership scorers while their power may differ.",
        "",
        *paper_table(rows, "mean_logprob"),
    ])
    markdown_path = reports / "local_vanilla_audit.md"
    markdown_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()

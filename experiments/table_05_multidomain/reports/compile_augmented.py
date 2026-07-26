"""Compile augmented parent-head replications of paper Table 5.

The target body is scored with the parent Pythia output head. Each text receives
domain-specific augmented views, then Aug-MIN-K scores are averaged across
views. Separate tables are emitted for k=20 and k=40; neither mixes in vanilla
scores, so each table is an unambiguous rerun of the paper's Table 5 protocol.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.experiments.experiment_registry import MIMIR_DOMAINS, gamma
from src.shard_audit.reporting.reporting_tables import load_json, score_test_metrics, verdict, write_csv, write_text


PARENT_MODEL = "EleutherAI/pythia-1.4b"
TARGET_MODEL = "herMaster/pythia1.4B-finetuned-on-lamini-docs"
PARENT_SLUG = "parent_pythia_1_4b"
TARGET_SLUG = "target_hermaster_pythia1_4b_lamini_docs"
ARTIFACT_ROOT = "artifacts/table_05_multidomain/augmented_parent_head_seed0"
M_EVAL_PER_SHARD = 400
GAMMA_005 = gamma(0.05, M_EVAL_PER_SHARD)


def _metric(domain: str, model_slug: str, k: int) -> dict:
    path = os.path.join(
        ARTIFACT_ROOT, "results", f"aug_mink{k}", domain, model_slug, "results.json"
    )
    return score_test_metrics(load_json(path), f"aug_min_k_{k}_logprob_mean")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _compile(k: int) -> tuple[str, list[dict]]:
    lines = [
        f"Augmented parent-head Table 5: MIN-K {k}%",
        f"Parent head: {PARENT_MODEL}",
        f"Target body: {TARGET_MODEL}",
        f"Held-out examples per shard: {M_EVAL_PER_SHARD}; gamma_0.05 = {GAMMA_005:.3f}",
        "",
        f"{'Domain':<18} {'Delta P':>9} {'Parent verdict':<23} {'Delta T':>9} {'Target verdict':<23} {'Shard signal?'}",
        "-" * 110,
    ]
    rows: list[dict] = []

    for domain in MIMIR_DOMAINS:
        parent = _metric(domain, PARENT_SLUG, k)
        target = _metric(domain, TARGET_SLUG, k)
        parent_adv = parent.get("shard_advantage")
        target_adv = target.get("shard_advantage")
        parent_verdict = verdict(parent_adv, GAMMA_005)
        target_verdict = verdict(target_adv, GAMMA_005)
        if parent_adv is None or target_adv is None:
            signal = "n/a"
        elif parent_adv > GAMMA_005 and target_adv > GAMMA_005:
            signal = "Yes"
        else:
            signal = "No"

        lines.append(
            f"{domain:<18} {_fmt(parent_adv):>9} {parent_verdict:<23} "
            f"{_fmt(target_adv):>9} {target_verdict:<23} {signal}"
        )
        rows.append({
            "domain": domain,
            "score": f"aug_min_k_{k}_logprob_mean",
            "parent_advantage": parent_adv,
            "parent_verdict": parent_verdict,
            "target_advantage": target_adv,
            "target_verdict": target_verdict,
            "gamma_005": GAMMA_005,
            "shard_signal": signal,
        })

    return "\n".join(lines) + "\n", rows


def main() -> None:
    for k in (20, 40):
        text, rows = _compile(k)
        report_dir = os.path.join(ARTIFACT_ROOT, "reports")
        text_path = os.path.join(report_dir, f"table_5_aug_mink{k}.txt")
        csv_path = os.path.join(report_dir, f"table_5_aug_mink{k}.csv")
        print(text)
        write_text(text_path, text)
        write_csv(csv_path, rows)
        print(f"Saved {text_path}")
        print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()

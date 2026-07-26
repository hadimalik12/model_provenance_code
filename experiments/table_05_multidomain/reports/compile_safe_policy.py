"""Compile the conservative Table 5 comparison at MIN-K 20% and 40%."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.experiments.experiment_registry import MIMIR_DOMAINS, gamma
from src.shard_audit.reporting.reporting_tables import load_json, score_test_metrics, verdict, write_csv, write_text
from src.shard_audit.scoring.table5_safe_augmentations import table5_variant_labels, table5_variant_score_keys

PARENT_SLUG = "parent_pythia_1_4b"
TARGET_SLUG = "target_hermaster_pythia1_4b_lamini_docs"
ARTIFACT_ROOT = "artifacts/table_05_multidomain/conservative_policy_parent_head_seed0"
GAMMA_005 = gamma(0.05, 400)


def _metric(domain: str, model_slug: str, score: str) -> dict:
    path = os.path.join(ARTIFACT_ROOT, "results", domain, model_slug, "results.json")
    return score_test_metrics(load_json(path), score)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def compile_table(k: int) -> tuple[str, list[dict]]:
    lines = [
        f"Table 5 conservative policy comparison: MIN-K {k}%",
        f"Held-out examples per shard: 400; gamma_0.05 = {GAMMA_005:.3f}",
        "",
        f"{'Domain':<18} {'Variant':<38} {'Delta P':>9} {'Parent verdict':<23} {'Delta T':>9} {'Target verdict':<23} {'Shard signal?'}",
        "-" * 150,
    ]
    rows: list[dict] = []
    for domain in MIMIR_DOMAINS:
        labels = table5_variant_labels(domain)
        for variant, score in table5_variant_score_keys(k, domain).items():
            parent_adv = _metric(domain, PARENT_SLUG, score).get("shard_advantage")
            target_adv = _metric(domain, TARGET_SLUG, score).get("shard_advantage")
            parent_verdict = verdict(parent_adv, GAMMA_005)
            target_verdict = verdict(target_adv, GAMMA_005)
            signal = "Yes" if parent_adv is not None and target_adv is not None and parent_adv > GAMMA_005 and target_adv > GAMMA_005 else "No"
            lines.append(f"{domain:<18} {labels[variant]:<38} {_fmt(parent_adv):>9} {parent_verdict:<23} {_fmt(target_adv):>9} {target_verdict:<23} {signal}")
            rows.append({"domain": domain, "variant": variant, "variant_label": labels[variant], "score": score,
                         "parent_advantage": parent_adv, "parent_verdict": parent_verdict,
                         "target_advantage": target_adv, "target_verdict": target_verdict,
                         "gamma_005": GAMMA_005, "shard_signal": signal})
    return "\n".join(lines) + "\n", rows


def main() -> None:
    report_dir = os.path.join(ARTIFACT_ROOT, "reports")
    for k in (20, 40):
        text, rows = compile_table(k)
        write_text(os.path.join(report_dir, f"table_5_mink{k}.txt"), text)
        write_csv(os.path.join(report_dir, f"table_5_mink{k}.csv"), rows)
        print(text)


if __name__ == "__main__":
    main()

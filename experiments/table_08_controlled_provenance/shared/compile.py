"""Compile completed Table 8 result JSON files into one honest, partial-safe table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import DEFAULT_CONFIG, artifacts, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    cfg = config(args.config)
    out = artifacts(cfg)
    rows = []
    for path in sorted((out / "audits").glob("**/result.json")):
        result = json.loads(path.read_text())
        held = result["held_out"]
        rows.append((result["family"], result["profile"] or "", result["stage"],
                     held["n_pos"], held["tpr"], held["fpr"], held["shard_advantage"],
                     held["auc"], result["randomization_p_two_sided"], result["reject_at_0_05"]))
    lines = ["# Table 8: controlled filtered CodeSearchNet shard provenance", "",
             f"Source: {cfg['source']['dataset']}. Any overlap filtering covers only the reference data listed in prepared/manifest.json.", "",
             "Each row uses the paper's MIN-K20 scorer and balanced-accuracy threshold fitted on the calibration split. Significance uses an exact, two-sided label randomization test on the held-out split.", "",
             "| Family | Downstream | Checkpoint | m/arm | TPR | FPR | Advantage | AUC | p | Reject .05 |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for family, profile, stage, m, tpr, fpr, adv, auc, p, reject in rows:
        lines.append(f"| {family} | {profile or '—'} | {stage} | {m} | {tpr:.3f} | {fpr:.3f} | {adv:.3f} | {auc:.3f} | {p:.3g} | {'yes' if reject else 'no'} |")
    by_checkpoint = {(family, profile, stage): {"adv": adv, "p": p}
                     for family, profile, stage, _, _, _, adv, _, p, _ in rows}
    summary = []
    for family, profiles in cfg["profiles"].items():
        base = by_checkpoint.get((family, "", "base"))
        parent = by_checkpoint.get((family, "", "parent"))
        for profile in profiles:
            control = by_checkpoint.get((family, profile, "control"))
            target = by_checkpoint.get((family, profile, "target"))
            if all((base, parent, control, target)):
                delta = target["adv"] - control["adv"]
                directional = (parent["adv"] > 0 and target["adv"] > 0
                               and target["p"] < 0.05 and base["adv"] <= 0
                               and control["adv"] <= 0)
                summary.append((family, profile, base["adv"], parent["adv"],
                                control["adv"], target["adv"], delta, directional))
    if summary:
        lines += ["", "## Paper-ready provenance summary", "",
                  "The provenance criterion is a significant positive target advantage with positive parent signal and nonpositive base/control advantages. The target−control difference is a descriptive effect size; a paired significance test has not yet been applied to that contrast.", "",
                  "| Family | Downstream data | Base adv. | Parent adv. | Control adv. | Target adv. | Target − control | Provenance pattern |",
                  "|---|---|---:|---:|---:|---:|---:|---|"]
        for family, profile, base, parent, control, target, delta, directional in summary:
            lines.append(f"| {family} | {profile} | {base:+.3f} | {parent:+.3f} | {control:+.3f} | {target:+.3f} | {delta:+.3f} | {'yes' if directional else 'no'} |")
    if not rows:
        lines.append("\nNo completed audits yet.")
    report = out / "table_06_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

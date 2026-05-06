"""Generate multi-domain MIMIR parent/target shard-membership report.

Reads per-domain experiment results and data manifests and produces:
  outputs/reports/mimir_multi_domain_parent_target/summary.md
  outputs/reports/mimir_multi_domain_parent_target/results.csv
  outputs/reports/mimir_multi_domain_parent_target/results.json

Usage:
    python scripts/reports/report_mimir_multi_domain_parent_target.py \\
        --run-root outputs/runs/mimir_domains \\
        --processed-root data/processed/mimir_domains \\
        --output-dir outputs/reports/mimir_multi_domain_parent_target
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PARENT_SLUG = "parent_pythia_1_4b"
TARGET_SLUG = "target_nnheui_pythia_1_4b_sft_full"
PARENT_MODEL = "EleutherAI/pythia-1.4b"
TARGET_MODEL = "nnheui/pythia-1.4b-sft-full"

PRIMARY = "min_k_20_logprob"
SCORE_KEYS = ["mean_logprob", "min_k_5_logprob", "min_k_10_logprob",
              "min_k_20_logprob", "min_k_40_logprob"]

# Known GitHub result for comparison
GITHUB_PARENT_ADV = 0.380
GITHUB_TARGET_ADV = 0.375
GITHUB_PARENT_AUC = 0.736
GITHUB_TARGET_AUC = 0.741

UNSTABLE_THRESHOLD = 0.05


def _fmt(v, d=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.{d}f}"


def _ratio_str(target_adv, parent_adv):
    if parent_adv is None or parent_adv != parent_adv:
        return "n/a"
    if abs(parent_adv) < UNSTABLE_THRESHOLD:
        return "unstable"
    ratio = target_adv / parent_adv if parent_adv else float("nan")
    return _fmt(ratio)


def _load_results(path: str) -> dict:
    """Load results.json → {score_name: result_dict}."""
    with open(path) as f:
        raw = json.load(f)
    lst = raw["main_results"] if isinstance(raw, dict) else raw
    return {r["score_name"]: r for r in lst}


def _load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _classify_domain(parent_adv, target_adv):
    if parent_adv is None:
        return "no data"
    if parent_adv < UNSTABLE_THRESHOLD:
        return "parent weak"
    if parent_adv >= 0.25 and target_adv is not None and target_adv >= 0.20:
        return "strong"
    if parent_adv >= 0.10:
        if target_adv is not None and target_adv < 0.10:
            return "parent medium, target weak"
        return "medium"
    return "weak"


def discover_domains(run_root: str) -> list:
    """Return sorted list of domain slugs that have both parent and target results."""
    slugs = []
    if not os.path.isdir(run_root):
        return slugs
    for entry in sorted(os.listdir(run_root)):
        d = os.path.join(run_root, entry)
        if not os.path.isdir(d):
            continue
        p_res = os.path.join(d, PARENT_SLUG, "results.json")
        t_res = os.path.join(d, TARGET_SLUG, "results.json")
        if os.path.isfile(p_res) and os.path.isfile(t_res):
            slugs.append(entry)
        elif os.path.isfile(p_res):
            logger.warning("Domain %s: parent results found but no target results.", entry)
            slugs.append(entry)
    return slugs


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate multi-domain MIMIR parent/target report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-root",       default="outputs/runs/mimir_domains")
    p.add_argument("--processed-root", default="data/processed/mimir_domains")
    p.add_argument("--output-dir",
                   default="outputs/reports/mimir_multi_domain_parent_target")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    domains = discover_domains(args.run_root)
    if not domains:
        logger.error("No domain results found under %s", args.run_root)
        sys.exit(1)

    logger.info("Domains with results: %s", domains)

    # ------------------------------------------------------------------ #
    # Load all data
    # ------------------------------------------------------------------ #
    domain_data = {}
    for slug in domains:
        p_path = os.path.join(args.run_root, slug, PARENT_SLUG, "results.json")
        t_path = os.path.join(args.run_root, slug, TARGET_SLUG, "results.json")
        m_path = os.path.join(args.processed_root, slug, "manifest.json")

        parent_res = _load_results(p_path) if os.path.isfile(p_path) else {}
        target_res = _load_results(t_path) if os.path.isfile(t_path) else {}
        manifest   = _load_manifest(m_path) if os.path.isfile(m_path) else {}

        domain_data[slug] = {
            "manifest": manifest,
            "parent": parent_res,
            "target": target_res,
        }

    # ------------------------------------------------------------------ #
    # Build flat results table
    # ------------------------------------------------------------------ #
    rows = []
    for slug in domains:
        dd = domain_data[slug]
        mf = dd["manifest"]
        for key in SCORE_KEYS:
            pr = dd["parent"].get(key, {})
            tr = dd["target"].get(key, {})
            p_test = pr.get("test", {})
            t_test = tr.get("test", {})
            p_adv = p_test.get("shard_advantage")
            t_adv = t_test.get("shard_advantage")
            p_auc = p_test.get("auc")
            t_auc = t_test.get("auc")
            p_acc = p_test.get("accuracy")
            t_acc = t_test.get("accuracy")
            rows.append({
                "domain": slug,
                "config":  mf.get("config", slug),
                "split":   mf.get("selected_split", "?"),
                "score":   key,
                "parent_acc":   p_acc,
                "parent_auc":   p_auc,
                "parent_adv":   p_adv,
                "parent_tpr1":  p_test.get("tpr_at_1_fpr"),
                "parent_threshold": pr.get("calibrated_threshold"),
                "target_acc":   t_acc,
                "target_auc":   t_auc,
                "target_adv":   t_adv,
                "target_tpr1":  t_test.get("tpr_at_1_fpr"),
                "target_threshold": tr.get("calibrated_threshold"),
            })

    # ------------------------------------------------------------------ #
    # Write CSV
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(args.output_dir, "results.csv")
    fieldnames = [
        "domain", "config", "split", "score",
        "parent_acc", "parent_auc", "parent_adv", "parent_tpr1", "parent_threshold",
        "target_acc", "target_auc", "target_adv", "target_tpr1", "target_threshold",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV written to %s", csv_path)

    # ------------------------------------------------------------------ #
    # Write JSON
    # ------------------------------------------------------------------ #
    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({"domains": domains, "rows": rows,
                   "generated": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    logger.info("JSON written to %s", json_path)

    # ------------------------------------------------------------------ #
    # Write Markdown report
    # ------------------------------------------------------------------ #
    now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []

    lines += [
        "# Multi-Domain MIMIR Parent/Target Shard-Membership Report",
        "",
        f"**Generated:** {now_utc}",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Parent model:** `{PARENT_MODEL}`",
        f"- **Target model:** `{TARGET_MODEL}`",
        f"- **Domains:** {', '.join(domains)}",
        f"- **Primary score:** `{PRIMARY}`",
        f"- **Known GitHub result (reference):** parent adv={GITHUB_PARENT_ADV}, "
        f"target adv={GITHUB_TARGET_ADV}, ratio=0.987",
        "",
    ]

    # Quick summary table
    lines += [
        "| Domain | Parent Adv | Target Adv | Adv Ratio | Classification |",
        "|---|---:|---:|---:|---|",
    ]
    for slug in domains:
        dd = domain_data[slug]
        pr = dd["parent"].get(PRIMARY, {}).get("test", {})
        tr = dd["target"].get(PRIMARY, {}).get("test", {})
        p_adv = pr.get("shard_advantage")
        t_adv = tr.get("shard_advantage")
        cls = _classify_domain(p_adv, t_adv)
        lines.append(
            f"| {slug} | {_fmt(p_adv)} | {_fmt(t_adv)} "
            f"| {_ratio_str(t_adv, p_adv)} | {cls} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 2. Experimental Setup",
        "",
        f"- **Dataset source:** MIMIR (`iamgroot42/mimir`)",
        f"- **Parent model:** `{PARENT_MODEL}` (Pythia-1.4b, pretrained on The Pile)",
        f"- **Target model:** `{TARGET_MODEL}` (SFT on UltraChat 200k)",
        f"- **Label semantics:** `label=1` = member of Pythia's pretraining corpus; "
        f"`label=0` = nonmember. Labels do NOT reflect target fine-tuning data.",
        f"- **Preprocessing:** normalize whitespace; truncate to 32 words; min 8 words; no chat template",
        f"- **Scoring:** causal LM log-probabilities with causal shift; "
        f"MIN-K% PROB at k=5,10,20,40; mean log-prob",
        f"- **Threshold calibration:** maximize balanced accuracy on the construction/train split",
        f"- **Evaluation:** held-out test split",
        f"- **Primary score:** `{PRIMARY}`",
        f"- **Seed:** 0",
        "",
        "---",
        "",
        "## 3. Data Diagnostics by Domain",
        "",
        "| Domain | Config | Split | Train/class | Test/class | "
        "Member kept | Nonmember kept | Size fallback? |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]

    for slug in domains:
        mf = domain_data[slug]["manifest"]
        lines.append(
            f"| {slug} "
            f"| `{mf.get('config', slug)}` "
            f"| `{mf.get('selected_split', '?')}` "
            f"| {mf.get('n_train_per_class', '?')} "
            f"| {mf.get('n_test_per_class', '?')} "
            f"| {mf.get('n_member_after_preprocessing', '?')} "
            f"| {mf.get('n_nonmember_after_preprocessing', '?')} "
            f"| {'yes' if mf.get('size_fallback') else 'no'} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Primary Results: MIN-K 20%",
        "",
        "| Domain | Parent Acc | Parent AUC | Parent Adv | "
        "Target Acc | Target AUC | Target Adv | Adv Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for slug in domains:
        dd = domain_data[slug]
        pr = dd["parent"].get(PRIMARY, {}).get("test", {})
        tr = dd["target"].get(PRIMARY, {}).get("test", {})
        p_adv = pr.get("shard_advantage")
        t_adv = tr.get("shard_advantage")
        lines.append(
            f"| {slug} "
            f"| {_fmt(pr.get('accuracy'))} "
            f"| {_fmt(pr.get('auc'))} "
            f"| {_fmt(p_adv)} "
            f"| {_fmt(tr.get('accuracy'))} "
            f"| {_fmt(tr.get('auc'))} "
            f"| {_fmt(t_adv)} "
            f"| {_ratio_str(t_adv, p_adv)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 5. Score-Level Results",
        "",
    ]

    for slug in domains:
        dd = domain_data[slug]
        lines += [
            f"### {slug}",
            "",
            "| Score | Parent Adv | Parent AUC | Target Adv | Target AUC | Adv Ratio |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for key in SCORE_KEYS:
            pr = dd["parent"].get(key, {}).get("test", {})
            tr = dd["target"].get(key, {}).get("test", {})
            p_adv = pr.get("shard_advantage")
            t_adv = tr.get("shard_advantage")
            lines.append(
                f"| `{key}` "
                f"| {_fmt(p_adv)} "
                f"| {_fmt(pr.get('auc'))} "
                f"| {_fmt(t_adv)} "
                f"| {_fmt(tr.get('auc'))} "
                f"| {_ratio_str(t_adv, p_adv)} |"
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 6. Interpretation by Domain",
        "",
    ]

    for slug in domains:
        dd = domain_data[slug]
        mf = dd["manifest"]
        pr = dd["parent"].get(PRIMARY, {}).get("test", {})
        tr = dd["target"].get(PRIMARY, {}).get("test", {})
        p_adv = pr.get("shard_advantage")
        t_adv = tr.get("shard_advantage")
        p_auc = pr.get("auc")
        cls = _classify_domain(p_adv, t_adv)

        lines.append(f"### {slug}")
        lines.append("")
        lines.append(f"- **Classification:** {cls}")
        lines.append(f"- **Parent advantage (`{PRIMARY}`):** {_fmt(p_adv)}")
        lines.append(f"- **Parent AUC:** {_fmt(p_auc)}")
        lines.append(f"- **Target advantage:** {_fmt(t_adv)}")
        lines.append(f"- **Preservation ratio:** {_ratio_str(t_adv, p_adv)}")

        if p_adv is None:
            lines.append("- **Interpretation:** No results available.")
        elif abs(p_adv) < UNSTABLE_THRESHOLD:
            lines.append(
                "- **Interpretation:** Parent advantage is near zero. "
                "This domain does not yield a useful membership signal under the current "
                "scalar distinguisher. Not a reliable provenance shard."
            )
        elif p_adv < 0.10:
            lines.append(
                "- **Interpretation:** Weak parent signal. Below the strong-domain threshold. "
                "Caution in interpretation."
            )
        elif p_adv >= 0.25:
            if t_adv is not None and t_adv >= 0.20:
                ratio = t_adv / p_adv
                lines.append(
                    f"- **Interpretation:** Strong parent signal and strong target preservation "
                    f"(ratio {ratio:.3f}). The target inherits the parent's pretraining "
                    f"membership signal for this domain."
                )
            elif t_adv is not None and t_adv < 0.10:
                lines.append(
                    "- **Interpretation:** Parent signal is strong but target advantage is weak. "
                    "Fine-tuning may have attenuated this domain's signal, or the target was "
                    "not exposed to this domain via inheritance."
                )
            else:
                lines.append(
                    "- **Interpretation:** Strong parent signal with partial target preservation."
                )
        else:
            lines.append(
                "- **Interpretation:** Moderate signal. Parent shows some distinguishability; "
                "further experiments needed."
            )
        lines.append("")

    lines += [
        "---",
        "",
        "## 7. Comparison to Existing GitHub Result",
        "",
        "The GitHub domain was previously run with 500 train + 200 test per class "
        "(same settings):",
        "",
        "| Model | min_k_20 Adv | min_k_20 AUC |",
        "|---|---:|---:|",
        f"| Parent (`{PARENT_MODEL}`) | {GITHUB_PARENT_ADV} | {GITHUB_PARENT_AUC} |",
        f"| Target (`{TARGET_MODEL}`) | {GITHUB_TARGET_ADV} | {GITHUB_TARGET_AUC} |",
        "",
        "Preservation ratio on GitHub: **0.987** (target retains essentially 100% of parent signal).",
        "",
        "---",
        "",
        "## 8. Limitations",
        "",
        "1. **MIMIR labels are Pythia/Pile-specific.** "
        "Labels are valid only for models trained on The Pile. "
        "Results for other model families are not interpretable as membership tests.",
        "2. **No UltraChat overlap audit.** "
        "If MIMIR domain texts appear in UltraChat, the target may have *directly* "
        "seen shard texts during fine-tuning — this would confound the inherited-provenance claim.",
        "3. **Scalar threshold only.** "
        "MIN-K% PROB and mean log-prob are scalar aggregations. "
        "A sequence-level GRU distinguisher would be more sensitive.",
        "4. **Single seed.** Multi-seed replication is recommended for borderline domains.",
        "5. **Small nonmember pools.** Some domains may limit the test size below 200 per class.",
        "6. **No null controls for new domains.** "
        "Nonmember-vs-nonmember controls should be run for each domain "
        "showing a strong parent signal.",
        "",
        "---",
        "",
        "## 9. Next Steps",
        "",
        "1. Run **nonmember-vs-nonmember controls** for each domain with parent Adv ≥ 0.20.",
        "2. Run an **unrelated-model control** (e.g., `gpt2-xl`) to bound the floor.",
        "3. Run **UltraChat overlap audit** by domain.",
        "4. Try a **GRU/full-trace distinguisher** on domains with weak scalar signal.",
        "5. Repeat with **Pythia-6.9B** and a 6.9B-derived target.",
        "6. Run **multi-seed replication** (seeds 1–4) for borderline domains.",
        "",
        "---",
        "",
        f"*Report generated by `scripts/reports/report_mimir_multi_domain_parent_target.py` "
        f"at {now_utc}.*",
    ]

    md_path = os.path.join(args.output_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Markdown report written to %s", md_path)

    # Console summary
    print("\n=== Multi-Domain Summary (min_k_20_logprob) ===")
    print(f"{'Domain':<22} {'P_Adv':>7} {'P_AUC':>7} {'T_Adv':>7} {'T_AUC':>7} {'Ratio':>8} {'Class'}")
    print("-" * 70)
    for slug in domains:
        dd = domain_data[slug]
        pr = dd["parent"].get(PRIMARY, {}).get("test", {})
        tr = dd["target"].get(PRIMARY, {}).get("test", {})
        p_adv = pr.get("shard_advantage")
        t_adv = tr.get("shard_advantage")
        print(
            f"{slug:<22} {_fmt(p_adv):>7} {_fmt(pr.get('auc')):>7} "
            f"{_fmt(t_adv):>7} {_fmt(tr.get('auc')):>7} "
            f"{_ratio_str(t_adv, p_adv):>8}  {_classify_domain(p_adv, t_adv)}"
        )
    print(f"\nReport:  {md_path}")
    print(f"CSV:     {csv_path}")
    print(f"JSON:    {json_path}")


if __name__ == "__main__":
    main()

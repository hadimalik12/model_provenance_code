"""Compile Table 5: Cross-Domain Signal with Augmentation.

Reads per-domain experiment results for both parent and target models,
comparing vanilla vs. augmented scoring across 5 MIMIR domains.

Output:
    outputs/replicated_table_5_aug.txt
    outputs/replicated_table_5_aug.csv

Usage:
    python scripts/reports/report_table_5_aug.py
"""

import csv
import json
import math
import os
import sys


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

PARENT_MODEL = "EleutherAI/pythia-1.4b"
TARGET_MODEL = "lomahony/pythia-1.4b-helpful-sft"
PARENT_SLUG = "parent_pythia_1_4b"
TARGET_SLUG = "target_lomahony_pythia_1_4b_helpful_sft"

DOMAINS = ["arxiv", "dm_mathematics", "github", "pile_cc", "wikipedia_en"]
# Map filesystem slugs back to display names
DOMAIN_DISPLAY = {
    "arxiv": "arxiv",
    "dm_mathematics": "dm_mathematics",
    "github": "github",
    "pile_cc": "pile_cc",
    "wikipedia_en": "wikipedia_en",
    "wikipedia__en_": "wikipedia_en",
}

VANILLA_SCORE = "min_k_20_logprob"
AUG_SCORE = "aug_min_k_20_logprob_mean"

M_EVAL = 200  # test size per class (Table 5 uses smaller splits)
GAMMA_005 = math.sqrt(math.log(2 / 0.05) / M_EVAL)

VANILLA_RUN_ROOT = "outputs/runs/mimir_domains"
AUG_RUN_ROOT = "outputs/runs/mimir_domains_aug"


def _extract_score_test_metrics(data: dict, score_name: str) -> dict:
    """Extract test metrics for a given score from results.json."""
    if not isinstance(data, dict):
        return {}
    main_results = data.get("main_results", [])
    if isinstance(main_results, list):
        for entry in main_results:
            if isinstance(entry, dict) and entry.get("score_name") == score_name:
                return entry.get("test", {})
    return {}


def _load_result(path: str, score_name: str) -> dict:
    """Load results.json and extract test metrics for a specific score."""
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return _extract_score_test_metrics(data, score_name)


def _verdict(adv: float) -> str:
    """Return verdict string based on shard advantage vs gamma."""
    if adv > GAMMA_005:
        return "Reject H0 / Yes"
    return "Fail to reject H0 / No"


def _fmt(v, d=3):
    """Format a float with d decimal places, or return 'n/a'."""
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return f"{v:+.{d}f}"


def main():
    lines = []
    csv_rows = []

    # Header
    lines.append(
        f"{'Domain':<18} "
        f"{'ΔP(van)':>9} {'ΔP(aug)':>9} {'Parent verdict':<22} "
        f"{'ΔT(van)':>9} {'ΔT(aug)':>9} {'Target verdict':<22} "
        f"{'Signal?'}"
    )
    lines.append("-" * 120)

    for domain in DOMAINS:
        display = DOMAIN_DISPLAY.get(domain, domain)

        # --- Vanilla results ---
        van_parent_path = os.path.join(
            VANILLA_RUN_ROOT, domain, PARENT_SLUG, "results.json"
        )
        van_target_path = os.path.join(
            VANILLA_RUN_ROOT, domain, TARGET_SLUG, "results.json"
        )

        van_p = _load_result(van_parent_path, VANILLA_SCORE)
        van_t = _load_result(van_target_path, VANILLA_SCORE)

        van_p_adv = van_p.get("shard_advantage")
        van_t_adv = van_t.get("shard_advantage")

        # --- Augmented results ---
        aug_parent_path = os.path.join(
            AUG_RUN_ROOT, domain, PARENT_SLUG, "results.json"
        )
        aug_target_path = os.path.join(
            AUG_RUN_ROOT, domain, TARGET_SLUG, "results.json"
        )

        aug_p = _load_result(aug_parent_path, AUG_SCORE)
        aug_t = _load_result(aug_target_path, AUG_SCORE)

        aug_p_adv = aug_p.get("shard_advantage")
        aug_t_adv = aug_t.get("shard_advantage")

        # Verdict: use augmented if available, else vanilla
        best_p_adv = aug_p_adv if aug_p_adv is not None else van_p_adv
        best_t_adv = aug_t_adv if aug_t_adv is not None else van_t_adv

        p_verdict_str = _verdict(best_p_adv) if best_p_adv is not None else "n/a"
        t_verdict_str = _verdict(best_t_adv) if best_t_adv is not None else "n/a"

        # Signal: both parent and target reject H0
        if (best_p_adv is not None and best_p_adv > GAMMA_005 and
                best_t_adv is not None and best_t_adv > GAMMA_005):
            signal = "Yes"
        elif best_p_adv is None or best_t_adv is None:
            signal = "n/a"
        else:
            signal = "No"

        lines.append(
            f"{display:<18} "
            f"{_fmt(van_p_adv):>9} {_fmt(aug_p_adv):>9} {p_verdict_str:<22} "
            f"{_fmt(van_t_adv):>9} {_fmt(aug_t_adv):>9} {t_verdict_str:<22} "
            f"{signal}"
        )

        csv_rows.append({
            "domain": display,
            "delta_p_vanilla": van_p_adv,
            "delta_p_aug": aug_p_adv,
            "parent_verdict": p_verdict_str,
            "delta_t_vanilla": van_t_adv,
            "delta_t_aug": aug_t_adv,
            "target_verdict": t_verdict_str,
            "shard_signal": signal,
        })

    lines.append("-" * 120)
    lines.append(f"\nγ₀.₀₅ = {GAMMA_005:.3f}  (m_eval = {M_EVAL})")
    lines.append(f"Parent: {PARENT_MODEL}")
    lines.append(f"Target: {TARGET_MODEL}")
    lines.append(f"Vanilla score: {VANILLA_SCORE}")
    lines.append(f"Augmented score: {AUG_SCORE}")

    output_str = "\n".join(lines)
    print(output_str)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/replicated_table_5_aug.txt", "w") as f:
        f.write(output_str + "\n")
    print("\nSaved to outputs/replicated_table_5_aug.txt")

    if csv_rows:
        csv_path = "outputs/replicated_table_5_aug.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()

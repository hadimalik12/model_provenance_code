"""Generate the master summary table for the Membership Inference Audit.

This script crawls 'outputs/runs/', matches main experiments with their non-member 
controls, and prints a formatted, fixed-width Markdown table.
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = REPO_ROOT / "outputs" / "runs"

# The primary score we are reporting
PRIMARY_SCORE = "min_k_20_logprob"

def get_metrics(results_path, score_name):
    if not results_path.exists():
        return None
    
    with open(results_path, "r") as f:
        data = json.load(f)
    
    # Main results are in a list
    for res in data.get("main_results", []):
        if res["score_name"] == score_name:
            # We use the 'test' metrics for reporting
            return {
                "acc": res["test"]["accuracy"],
                "adv": res["test"]["shard_advantage"]
            }
    return None

def main():
    # Identify all slugs by looking for mimir_github_* folders
    slugs = []
    if RUNS_DIR.exists():
        for d in RUNS_DIR.iterdir():
            if d.is_dir() and d.name.startswith("mimir_github_") and not d.name.endswith("_mink"):
                slugs.append(d.name.replace("mimir_github_", ""))
    
    # Define parents manually for grouping
    parents = [
        ("1B", "pythia_1b_mink", "EleutherAI/pythia-1b"),
        ("1.4B", "pythia_1_4b_mink", "EleutherAI/pythia-1.4b"),
        ("6.9B", "pythia_6_9b_mink", "EleutherAI/pythia-6.9b"),
        ("12B", "pythia_12b_mink", "EleutherAI/pythia-12b"),
    ]

    # Prepare rows
    rows = []
    for scale, p_slug, p_name in parents:
        # 1. Get Parent Row
        p_main_path = RUNS_DIR / f"mimir_github_{p_slug}" / "results.json"
        p_ctrl_path = RUNS_DIR / f"nonmember_control_seed0_{p_slug}" / "results.json"
        
        m_main = get_metrics(p_main_path, PRIMARY_SCORE)
        m_ctrl = get_metrics(p_ctrl_path, PRIMARY_SCORE)

        if m_main and m_ctrl:
            rows.append([scale, f"**{p_name}** (Parent)", f"{m_ctrl['acc']:.1%}", f"{m_ctrl['adv']:.3f}", f"{m_main['acc']:.1%}", f"{m_main['adv']:.3f}"])
        
        # 2. Get Children Targets
        for slug in sorted(slugs):
            is_match = False
            if scale == "1B":
                if "1b" in slug and "1_4b" not in slug: is_match = True
            elif scale == "1.4B":
                if "1_4b" in slug: is_match = True
            elif scale == "6.9B":
                if "6_9b" in slug or "ultrachat" in slug: is_match = True
            elif scale == "12B":
                if "12b" in slug: is_match = True
            
            if is_match:
                c_main_path = RUNS_DIR / f"mimir_github_{slug}" / "results.json"
                c_ctrl_path = RUNS_DIR / f"nonmember_control_seed0_{slug}" / "results.json"
                m_main = get_metrics(c_main_path, PRIMARY_SCORE)
                m_ctrl = get_metrics(c_ctrl_path, PRIMARY_SCORE)
                
                if m_main and m_ctrl:
                    name = slug.replace("_pythia", "").replace("_1_4b", "").replace("_6_9b", "").replace("_12b", "").replace("1b", "")
                    display_name = name.replace("_", " ").strip().title()
                    rows.append(["", display_name, f"{m_ctrl['acc']:.1%}", f"{m_ctrl['adv']:.3f}", f"{m_main['acc']:.1%}", f"{m_main['adv']:.3f}"])
        
        rows.append(["---", "---", "---", "---", "---", "---"]) # Divider

    if not rows:
        print("No results found in outputs/runs/")
        return

    # Calculate column widths
    headers = ["Scale", "Model Name", "Ctrl Acc", "Ctrl Adv", "Main Acc", "Main Adv"]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    # Print Table
    def print_row(items):
        formatted = " | ".join(str(item).ljust(widths[i]) for i, item in enumerate(items))
        print(f"| {formatted} |")

    print_row(headers)
    print_row(["-" * w for w in widths])
    for row in rows:
        if row[0] == "---":
            continue
        print_row(row)

if __name__ == "__main__":
    main()

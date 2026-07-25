"""Collect sweep & MIA experiment results into an Augmented Table 2 replica."""
import json
import math
import os

TABLE_2_MODELS = [
    # (Parent Name, Target Model HF ID, Target Short Name)
    ("Pythia-1B", "Leogrin/eleuther-pythia1b-hh-sft", "Leogrin Hh Sft"),
    ("Pythia-1.4B", "herMaster/pythia1.4B-finetuned-on-lamini-docs", "Hermaster1 4B Lamini Docs"),
    ("Pythia-1.4B", "kykim0/pythia-1.4b-tulu-v2-mix", "Kykim0 Tulu V2 Mix"),
    ("Pythia-1.4B", "LinguaCustodia/fin-pythia-1.4b", "Linguacustodia Fin"),
    ("Pythia-1.4B", "lomahony/pythia-1.4b-helpful-dpo", "Lomahony Helpful Dpo"),
    ("Pythia-1.4B", "lomahony/pythia-1.4b-helpful-sft", "Lomahony Helpful Sft"),
    ("Pythia-1.4B", "nnheui/pythia-1.4b-sft-full", "Nnheui Sft Full"),
    ("Pythia-6.9B", "allenai/open-instruct-pythia-6.9b-tulu", "Allenai Tulu"),
    ("Pythia-6.9B", "lomahony/eleuther-pythia6.9b-hh-dpo", "Lomahony Hh Dpo"),
    ("Pythia-6.9B", "lomahony/eleuther-pythia6.9b-hh-sft", "Lomahony Hh Sft"),
    ("Pythia-6.9B", "pkarypis/pythia-ultrachat", "Pkarypis Ultrachat"),
    ("Pythia-6.9B", "usvsnsp/pythia-6.9b-ppo", "Usvsnsp Ppo"),
    ("Pythia-12B", "lomahony/eleuther-pythia12b-hh-dpo", "Lomahony Hh Dpo"),
    ("Pythia-12B", "lomahony/eleuther-pythia12b-hh-sft", "Lomahony Hh Sft"),
]

m_eval = 400  # held-out evaluation size per shard
gamma_0_05 = math.sqrt(math.log(2 / 0.05) / m_eval)  # 0.096


PARENT_HF_MAP = {
    "Pythia-1B": "EleutherAI/pythia-1b",
    "Pythia-1.4B": "EleutherAI/pythia-1.4b",
    "Pythia-6.9B": "EleutherAI/pythia-6.9b",
    "Pythia-12B": "EleutherAI/pythia-12b",
}


def _extract_score_test_metrics(data: dict, target_score: str) -> dict:
    """Extract the test metrics dict for a given score name from results.json."""
    if not isinstance(data, dict):
        return {}
    # Search main_results list
    main_results = data.get("main_results", [])
    if isinstance(main_results, list):
        for entry in main_results:
            if isinstance(entry, dict) and entry.get("score_name") == target_score:
                return entry.get("test", {})
    # Direct dictionary fallback
    if target_score in data and isinstance(data[target_score], dict):
        return data[target_score].get("test", {})
    return {}


def main():
    lines = []
    lines.append(f"{'Parent model':<15} {'Target model':<32} {'Ctrl Acc':>9} {'Δ_ctrl':>8} {'Main Acc':>9} {'Δ_main':>8} {'γ_0.05':>8} {'Verdict'}")
    lines.append("-" * 105)

    runs_dir = "outputs/runs"

    for parent, target_hf, target_name in TABLE_2_MODELS:
        clean_target = target_hf.replace("/", "__")
        result_file = os.path.join(runs_dir, f"mimir_github_{clean_target}", "results.json")

        # The parent model's performance on the main dataset should NOT be used as the control,
        # because the parent model was pre-trained on this dataset and thus memorized it.
        # We will attempt to load the target model's nonmember-vs-nonmember control run.
        acc_ctrl = 0.50
        adv_ctrl = 0.00
        
        nm_result_file = os.path.join(runs_dir, f"mimir_github_nonmember_control_aug_{clean_target}", "results.json")

        if os.path.exists(nm_result_file):
            with open(nm_result_file) as f:
                nm_data = json.load(f)
            nm_aug_res = _extract_score_test_metrics(nm_data, "min_k_20_logprob")
            acc_ctrl = nm_aug_res.get("accuracy", acc_ctrl)
            adv_ctrl = nm_aug_res.get("shard_advantage", adv_ctrl)
        elif os.path.exists(result_file):
            # Fall back to shuffled label control if nonmember control is missing
            with open(result_file) as f:
                data = json.load(f)
            if "shuffled_label_control" in data and isinstance(data["shuffled_label_control"], list):
                for ctrl in data["shuffled_label_control"]:
                    if ctrl.get("score_name") == "min_k_20_logprob":
                        acc_ctrl = ctrl.get("test_accuracy", acc_ctrl)
                        adv_ctrl = ctrl.get("test_advantage", adv_ctrl)
                        break

        if os.path.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            
            aug_res = _extract_score_test_metrics(data, "min_k_20_logprob")
            acc_main = aug_res.get("accuracy", 0.5)
            adv_main = aug_res.get("shard_advantage", 0.0)

            verdict = "Reject H0 / Yes" if adv_main > gamma_0_05 else "Fail to reject / No"
            lines.append(f"{parent:<15} {target_name:<32} {acc_ctrl:>8.1%} {adv_ctrl:>+8.3f} {acc_main:>8.1%} {adv_main:>+8.3f} {gamma_0_05:>8.3f} {verdict}")
        else:
            lines.append(f"{parent:<15} {target_name:<32} {'-- results not found --':>45}")

    lines.append("-" * 105)
    output_str = "\n".join(lines)
    print(output_str)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/replicated_table_2.txt", "w") as f:
        f.write(output_str)
    print("\nSaved compiled table to outputs/replicated_table_2.txt")


if __name__ == "__main__":
    main()

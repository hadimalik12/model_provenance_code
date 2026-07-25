"""Collect sweep results into a Table 1 replica."""
import json
import math
import os

targets = ["self", "pets", "cifar10", "cifar100", "eurosat", "celeba", "har"]
ctrl = "random_init_logprob"
m = 3850  # full evaluation split
gamma = math.sqrt(math.log(2 / 0.05) / (m // 2))  # m//2 = 1925 per-class

lines = []
lines.append(f"{'Target Model':<20} {'Ctrl Acc':>9} {'Δ_ctrl':>8} {'Main Acc':>9} {'Δ_main':>8} {'γ_0.05':>8} {'Verdict'}")
lines.append("-" * 80)

# 1. Print positive targets
for name in targets:
    path = f"outputs/runs/replicated_sweep_aug_logprob_{name}_vs_{ctrl}/results.json"
    if not os.path.exists(path):
        lines.append(f"{name:<20} -- results not found --")
        continue
    with open(path) as f:
        data = json.load(f)
    pos = data["results"]["positive"]
    neg = data["results"]["negative"]
    pos_entry = max(pos, key=lambda e: e["m_total"])
    neg_entry = max(neg, key=lambda e: e["m_total"])

    adv_main = pos_entry["advantage_mean"]
    acc_main = pos_entry["accuracy_mean"]
    adv_ctrl = neg_entry["advantage_mean"]
    acc_ctrl = neg_entry["accuracy_mean"]
    verdict = "Reject H0" if abs(adv_main) > gamma else "Fail to reject"

    lines.append(f"{name:<20} {acc_ctrl:>8.1%} {adv_ctrl:>+8.3f} {acc_main:>8.1%} {adv_main:>+8.3f} {gamma:>8.4f} {verdict}")

lines.append("-" * 80)

# 2. Print Negative Controls
# For random-init, we can extract from any positive target's negative run
path_self = f"outputs/runs/replicated_sweep_aug_logprob_self_vs_{ctrl}/results.json"
if os.path.exists(path_self):
    with open(path_self) as f:
        data = json.load(f)
    neg_entry = max(data["results"]["negative"], key=lambda e: e["m_total"])
    adv = neg_entry["advantage_mean"]
    acc = neg_entry["accuracy_mean"]
    verdict = "Reject H0" if abs(adv) > gamma else "Fail to reject"
    lines.append(f"{'random-init':<20} {'—':>9} {'—':>8} {acc:>8.1%} {adv:>+8.3f} {gamma:>8.4f} {verdict}")
else:
    lines.append(f"{'random-init':<20} -- results not found --")

# For deit, we can extract from self_vs_deit_logprob sweep's negative run
path_deit = "outputs/runs/replicated_sweep_aug_logprob_self_vs_deit_logprob/results.json"
if os.path.exists(path_deit):
    with open(path_deit) as f:
        data = json.load(f)
    neg_entry = max(data["results"]["negative"], key=lambda e: e["m_total"])
    adv = neg_entry["advantage_mean"]
    acc = neg_entry["accuracy_mean"]
    verdict = "Reject H0" if abs(adv) > gamma else "Fail to reject"
    lines.append(f"{'deit':<20} {'—':>9} {'—':>8} {acc:>8.1%} {adv:>+8.3f} {gamma:>8.4f} {verdict}")
else:
    lines.append(f"{'deit':<20} -- results not found --")

# Print to stdout and save to outputs
output_str = "\n".join(lines)
print(output_str)

os.makedirs("outputs", exist_ok=True)
with open("outputs/replicated_table_1.txt", "w") as f:
    f.write(output_str)
print("\nSaved compiled table to outputs/replicated_table_1.txt")

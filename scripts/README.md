# Shared Commands

These are reusable pipeline components, not places to look for a particular
paper table. Start in `experiments/` and follow the links from that experiment
README to the commands it uses.

| Directory | Responsibility |
|---|---|
| `data/` | Create validated member/control dataset splits |
| `scoring/` | Score causal language models and write per-example JSONL |
| `audit/` | Calibrate threshold distinguishers and evaluate held-out advantage |
| `cluster/` | Provision the PACE environment |
| `setup/` | Validate Python environment and experiment YAML |

The experiment-specific ImageNet scorers are intentionally under
`experiments/table_01_imagenet/scoring/`, because their parent-head and image
augmentation assumptions are not shared by language-model experiments.

# Revised Next-Step Plan: Scaffold Repo and Run MIMIR GitHub Parent-Membership Sanity Check

## Objective

The immediate goal is to scaffold the repository and run a **clean parent-model membership sanity check** using the most efficient validated membership-inference baseline from the LLM membership-auditing paper.

The sanity check should answer:

> For a base Pythia parent model \(P\), can we distinguish examples from a known member shard \(S\) from examples from a matched nonmember/control shard \(S'\), using the same dataset and preprocessing setup as the membership-inference paper?

This is not yet the full model-provenance experiment. It is the first validation step needed before the provenance paper’s target-model experiment.

---

## Key Change from the Previous Plan

The previous plan centered on full per-token loss traces and a GRU sequence classifier.

For this next step, change the first implementation to follow the efficient, validated baseline setup from the paper:

```text
Primary distinguisher input:
  scalar or small-vector membership scores derived from per-token log-probabilities

Primary method:
  MIN-K% PROB, with k = 20 by default

Secondary baselines:
  mean log-probability / perplexity
  zlib-normalized score, if easy to implement
  logistic regression over a small set of scalar membership features, optional

Not first priority:
  GRU over full loss traces
  PETAL label-only approximation
```

Rationale:

- The provenance paper’s GRU-style trace classifier is useful later, but the first sanity check should use a simple, established MIA baseline.
- If MIN-K% / PPL cannot detect membership on official MIMIR GitHub for base Pythia, then the pipeline or data semantics are likely wrong.
- Once the efficient baseline works, we can generalize to full-trace or label-only distinguishers.

---

## Core Experimental Question

Let:

```text
P  = base Pythia parent model
S  = MIMIR GitHub member shard
S' = MIMIR GitHub nonmember/control shard
```

For each candidate record \(x\), compute a member-likeness score:

\[
s_P(x)
\]

using the parent model’s per-token log-probabilities.

Train or calibrate a distinguisher \(A\) on construction splits:

```text
S_train and S'_train
```

Then report held-out test accuracy on:

```text
S_test and S'_test
```

The final question for this sanity check is:

> Does \(A\), constructed from Pythia scores, distinguish unseen MIMIR GitHub members from unseen MIMIR GitHub nonmembers?

---

## What Is the Input to the Distinguisher?

The distinguisher should conceptually take a raw candidate text \(x\), but internally it receives scores computed from \(P\).

### Step 1: Token-level scoring

Given text:

\[
x = (t_1,\ldots,t_n),
\]

compute causal-LM log-probabilities:

\[
\log p_P(t_i \mid t_{<i})
\]

for each scored token.

Implementation detail:

```text
logits at position i-1 predict token i
```

Do not score padding tokens.

### Step 2: Convert token log-probs into a membership score

Primary score:

```text
MIN-K% PROB, k = 20
```

For each record:

1. compute all token log-probabilities;
2. select the bottom 20% tokens by probability, i.e. lowest log-probability tokens;
3. average their log-probabilities;
4. use this value as the member-likeness score.

Definition:

\[
s_{\mathrm{minK20}}(x)
=
\frac{1}{|\mathrm{MinK}(x)|}
\sum_{i \in \mathrm{MinK}(x)}
\log p_P(t_i \mid t_{<i})
\]

where \(\mathrm{MinK}(x)\) is the set of the lowest-probability 20% of tokens in \(x\).

Higher score means more member-like because the model assigns less surprisingly low probability to the difficult tokens.

Secondary scores:

```text
s_mean_logprob(x) = average token log-probability
s_ppl(x)          = negative perplexity or -mean loss, with direction clearly defined
s_zlib(x)         = zlib-normalized log-probability, optional
```

### Step 3: Define the binary distinguisher

The simplest distinguisher is a threshold rule:

\[
A_\tau(x) = 1\{s_P(x) \ge \tau\}.
\]

Output convention:

```text
A(x) = 1  means predict x came from the member shard S
A(x) = 0  means predict x came from the control shard S'
```

The threshold \(\tau\) must be chosen using only the construction/training split:

```text
S_train and S'_train
```

Then evaluate once on held-out:

```text
S_test and S'_test
```

Optional learned distinguisher:

```text
logistic regression over [mean_logprob, min_k_5, min_k_10, min_k_20, min_k_40, zlib_score]
```

This is still efficient and auditable. It should not replace the pure MIN-K threshold baseline; it is only an additional diagnostic.

---

## Dataset and Preprocessing Should Follow the Membership-Inference Paper

Use the paper’s setup as closely as possible for the first run.

### Dataset

Use official MIMIR GitHub member/nonmember data.

Expected semantics:

```text
S  = GitHub examples from the Pile training/member side
S' = GitHub examples from the Pile held-out/test/nonmember side
```

Both sides should be GitHub-domain examples. The goal is not to distinguish domains; the goal is to distinguish training membership.

### Target/Parent model for sanity check

Use base Pythia first:

```text
EleutherAI/pythia-1.4b
```

If compute allows, repeat with:

```text
EleutherAI/pythia-6.9b
```

Do not use ShareGPT-fine-tuned Pythia models in this first sanity check.

### Text length

Follow the paper’s controlled main setup first:

```text
truncate to 32 words
```

Do not start with 1024-token traces. That is for the provenance paper’s sequence-model direction, not the first MIMIR sanity check.

Later variants:

```text
64 words
128 words
128 model tokens
256 model tokens
1024 model tokens
```

But the first result should be:

```text
MIMIR GitHub + base Pythia + 32-word truncation + MIN-K% PROB
```

### MIN-K setting

Use:

```text
k = 20
```

as the default, matching the membership-inference paper’s baseline configuration.

Also compute:

```text
k = 5, 10, 40
```

for diagnostics, but the primary reported MIN-K result should be k = 20 unless the report explicitly says otherwise.

---

## Metrics: Change Toward the Provenance Paper’s Goal

The membership-inference paper reports AUC, balanced accuracy, and TPR@1%FPR.

For our provenance paper, the first metric should be:

```text
held-out test accuracy
```

because the current goal is:

> Train/calibrate a distinguisher on construction parts of \(S\) and \(S'\), then test whether it distinguishes unseen held-out examples.

Still report diagnostic metrics:

```text
test accuracy
balanced test accuracy
AUC
TPR@1%FPR
empirical shard advantage
```

### Empirical shard advantage

Let:

```text
a  = fraction of S_test examples predicted as member
a' = fraction of S'_test examples predicted as member
```

Then:

\[
\widehat{\mathrm{Adv}}(A;P,S,S') = |a-a'|.
\]

For balanced test sets and a correctly oriented classifier:

\[
\mathrm{accuracy} = \frac{a + (1-a')}{2}
\]

and therefore:

\[
\widehat{\mathrm{Adv}} \approx 2(\mathrm{accuracy}-0.5).
\]

Report both accuracy and advantage because the provenance paper’s theory is stated in terms of shard distinguishing advantage.

---

## Desired Repository Structure

Scaffold the repo to support future generalization from this sanity check to provenance experiments.

```text
repo_root/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── paper/
├── Notebook/
│
├── configs/
│   ├── env/
│   │   └── local.yaml
│   └── experiments/
│       └── mimir_github_pythia_min_k.yaml
│
├── scripts/
│   ├── setup/
│   │   ├── setup_local_env.sh
│   │   └── check_env.py
│   ├── data/
│   │   └── prepare_mimir_github.py
│   ├── scoring/
│   │   └── extract_logprob_scores.py
│   ├── experiments/
│   │   └── run_mimir_github_pythia_mink_sanity.sh
│   └── reports/
│       └── make_mia_score_report.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   ├── metric.py
│   ├── pre_comp_emb.py
│   │
│   ├── apis/
│   ├── dpsda/
│   ├── utility_eval/
│   │
│   └── shard_audit/
│       ├── __init__.py
│       ├── datasets.py              # MIMIR loading and member/nonmember construction
│       ├── preprocessing.py         # paper-matched 32-word truncation
│       ├── logprobs.py              # causal LM token log-prob extraction
│       ├── mia_scores.py            # PPL, MIN-K, zlib-normalized scores
│       ├── distinguishers.py        # threshold and logistic scalar distinguishers
│       ├── metrics.py               # accuracy, AUC, TPR@FPR, shard advantage
│       ├── splitting.py             # construction/test split logic
│       ├── reporting.py             # Markdown/JSON reports
│       └── sanity_checks.py         # causal-shift and score-direction tests
│
├── data/
│   ├── raw/
│   │   └── mimir_github/
│   ├── processed/
│   │   └── mimir_github/
│   └── scores/
│       └── mimir_github_pythia_mink/
│
└── outputs/
    ├── runs/
    ├── reports/
    └── figures/
```

Important: this first scaffold should emphasize `logprobs.py` and `mia_scores.py`, not GRU training.

---

## Phase 1: Environment Setup

Create:

```text
scripts/setup/setup_local_env.sh
scripts/setup/check_env.py
```

The setup script should wrap or call the existing local setup script.

It should verify:

```text
python
torch
cuda
transformers
datasets
scikit-learn
numpy
zlib availability
HF cache path
```

Success command:

```bash
bash scripts/setup/setup_local_env.sh
python scripts/setup/check_env.py
```

---

## Phase 2: Load Official MIMIR GitHub

Create:

```text
src/shard_audit/datasets.py
scripts/data/prepare_mimir_github.py
```

Output files:

```text
data/processed/mimir_github/train.jsonl
data/processed/mimir_github/test.jsonl
data/processed/mimir_github/manifest.json
```

The JSONL format should be:

```json
{
  "id": "...",
  "text": "...",
  "label": 1,
  "source": "mimir_github",
  "split_origin": "member"
}
```

and:

```json
{
  "id": "...",
  "text": "...",
  "label": 0,
  "source": "mimir_github",
  "split_origin": "nonmember"
}
```

The split should be:

```text
train.jsonl = S_train ∪ S'_train
test.jsonl  = S_test ∪ S'_test
```

Initial sizes:

```text
smoke: 100 per class train, 100 per class test
small: 1,000 per class train, 1,000 per class test
main sanity: 5,000 per class train, 1,000 or 5,000 per class test
```

The manifest must record:

```json
{
  "dataset": "MIMIR GitHub",
  "member_definition": "MIMIR GitHub member/train side",
  "nonmember_definition": "MIMIR GitHub held-out/test side",
  "model_intended_for": "base Pythia trained on the Pile",
  "preprocessing": "32-word truncation",
  "seed": 0
}
```

---

## Phase 3: Paper-Matched Preprocessing

Create:

```text
src/shard_audit/preprocessing.py
```

Required behavior:

```text
normalize whitespace
truncate to exactly up to 32 words
drop examples below a minimum word count, e.g. 8 words
use identical preprocessing for member and nonmember
no prompt template
no instruction formatting
raw text only
```

Expose:

```python
def normalize_text(text: str) -> str: ...
def truncate_words(text: str, max_words: int = 32) -> str: ...
def is_valid_text(text: str, min_words: int = 8) -> bool: ...
```

The report must include:

```text
word length distribution before truncation
word length distribution after truncation
token length distribution under Pythia tokenizer
number filtered per class
```

---

## Phase 4: Extract Token Log-Probabilities

Create:

```text
src/shard_audit/logprobs.py
scripts/scoring/extract_logprob_scores.py
```

For each example, save:

```json
{
  "id": "...",
  "label": 1,
  "text_hash": "...",
  "model": "EleutherAI/pythia-1.4b",
  "num_scored_tokens": 37,
  "mean_logprob": -2.81,
  "mean_loss": 2.81,
  "min_k_5_logprob": -5.10,
  "min_k_10_logprob": -4.70,
  "min_k_20_logprob": -4.20,
  "min_k_40_logprob": -3.80,
  "zlib_norm_score": null,
  "logprobs_debug": [ ... optional for first N only ... ],
  "tokens_debug": [ ... optional for first N only ... ]
}
```

Save compact arrays if needed, but always save debug examples.

### Causal shift check

For token ids:

```text
[t_0, t_1, ..., t_n]
```

the score for `t_i` must come from logits at position `i-1`.

Do not use generated tokens. Do not score the wrong shifted position.

### Direction convention

Use member-likeness scores where higher means more likely member.

Therefore:

```text
mean_logprob: higher is more member-like
min_k_logprob: higher is more member-like
mean_loss: lower is more member-like; use -mean_loss if using as score
```

Add explicit assertions or comments in code.

---

## Phase 5: Build the Efficient Distinguisher

Create:

```text
src/shard_audit/distinguishers.py
```

### Primary distinguisher

Use a threshold on `min_k_20_logprob`:

\[
A_\tau(x) = 1\{s_{\mathrm{minK20}}(x) \ge \tau\}.
\]

Choose \(\tau\) on the training/construction split only.

Threshold selection options:

```text
default: maximize balanced accuracy on train
optional: maximize standard accuracy on train when classes are exactly balanced
optional: choose low-FPR threshold for TPR@1%FPR diagnostic
```

The primary reported test result should be from this thresholded MIN-K20 distinguisher.

### Secondary distinguishers

Implement but do not make them primary:

```text
mean_logprob_threshold
min_k_5_threshold
min_k_10_threshold
min_k_40_threshold
logistic_regression_scalar_features
```

Feature vector for logistic regression:

```text
[
  mean_logprob,
  min_k_5_logprob,
  min_k_10_logprob,
  min_k_20_logprob,
  min_k_40_logprob,
  zlib_norm_score if available,
  num_scored_tokens
]
```

Do not include features that create obvious preprocessing leakage unless explicitly reported.

---

## Phase 6: Metrics and Gold Labels

The gold labels are:

```text
label = 1 for S, the MIMIR GitHub member shard
label = 0 for S', the MIMIR GitHub nonmember/control shard
```

Primary metric for this project step:

```text
held-out test accuracy
```

Also report:

```text
balanced test accuracy
AUC
TPR@1%FPR
empirical shard advantage
threshold selected on train
train accuracy
test accuracy
```

Compute shard advantage on test:

```python
a = mean(A(x) == 1 for x in S_test)
a_prime = mean(A(x_prime) == 1 for x_prime in S_prime_test)
adv = abs(a - a_prime)
```

For balanced test sets:

```python
test_acc = 0.5 * (a + (1 - a_prime))
```

Report both.

---

## Phase 7: One-Command Sanity Run

Create:

```text
scripts/experiments/run_mimir_github_pythia_mink_sanity.sh
```

Suggested content:

```bash
#!/usr/bin/env bash
set -euo pipefail

bash scripts/setup/setup_local_env.sh

python scripts/data/prepare_mimir_github.py \
  --num-train-per-class 5000 \
  --num-test-per-class 1000 \
  --max-words 32 \
  --min-words 8 \
  --seed 0 \
  --output-dir data/processed/mimir_github

python scripts/scoring/extract_logprob_scores.py \
  --model EleutherAI/pythia-1.4b \
  --train-file data/processed/mimir_github/train.jsonl \
  --test-file data/processed/mimir_github/test.jsonl \
  --output-dir data/scores/mimir_github_pythia_mink \
  --min-k-pcts 5,10,20,40 \
  --batch-size 4 \
  --debug-examples 6

python scripts/reports/make_mia_score_report.py \
  --score-dir data/scores/mimir_github_pythia_mink \
  --output-dir outputs/reports/mimir_github_pythia_mink \
  --primary-score min_k_20_logprob \
  --methods mean_logprob,min_k_5,min_k_10,min_k_20,min_k_40,logistic_features
```

---

## Phase 8: Report Format

Each run should write:

```text
outputs/reports/mimir_github_pythia_mink/summary.md
outputs/reports/mimir_github_pythia_mink/metrics.json
outputs/reports/mimir_github_pythia_mink/score_histograms.png
outputs/reports/mimir_github_pythia_mink/manifest.json
```

Required `summary.md`:

```markdown
# MIMIR GitHub Pythia Parent Membership Sanity Check

## 1. Setup
- Parent model:
- Dataset:
- Member definition:
- Nonmember definition:
- Preprocessing:
- Score:
- Train/test sizes:
- Seed:

## 2. Score Extraction Sanity Checks
- Number of examples scored:
- Token length distribution:
- Mean logprob by class:
- Min-K20 logprob by class:
- Filtering:

## 3. Distinguisher Results
| Method | Test Accuracy | Balanced Accuracy | AUC | TPR@1%FPR | Shard Advantage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mean logprob threshold | ... | ... | ... | ... | ... |
| Min-K 5 threshold | ... | ... | ... | ... | ... |
| Min-K 10 threshold | ... | ... | ... | ... | ... |
| Min-K 20 threshold | ... | ... | ... | ... | ... |
| Min-K 40 threshold | ... | ... | ... | ... | ... |
| Logistic scalar features | ... | ... | ... | ... | ... |

## 4. Interpretation
- Does base Pythia reveal membership signal on MIMIR GitHub?
- Is the signal visible using the paper-matched efficient baseline?
- Is the pipeline ready for target/provenance experiments?

## 5. Next Steps
- Repeat with Pythia 6.9B.
- Repeat with MIMIR Pile-CC as a harder benchmark.
- Then switch model from parent P to target T for inherited shard-membership testing.
```

---

## Acceptance Criteria

This milestone is successful if:

1. The repo is scaffolded into a reusable structure.
2. The environment setup script works locally.
3. The dataset manifest clearly states official MIMIR GitHub member/nonmember semantics.
4. Preprocessing follows the paper’s 32-word truncation setup.
5. Token log-probability extraction passes causal-shift sanity checks.
6. The primary MIN-K20 threshold distinguisher is trained/calibrated only on the construction split.
7. The final report gives held-out test accuracy and shard advantage on \(S_{\text{test}}\) vs \(S'_{\text{test}}\).
8. The result is interpretable as a parent-model membership sanity check.

If the result is near random:

```text
1. verify MIMIR split loading;
2. verify label direction;
3. verify causal shift;
4. verify score direction;
5. try Pythia-6.9B;
6. compare with the paper’s reported MIMIR GitHub setup;
7. only then consider using a GRU/full trace model.
```

---

## Connection to the Final Model-Provenance Paper

After this parent sanity check works, the final provenance version replaces:

```text
P = base Pythia parent model
```

with:

```text
T = candidate derived target model
```

The same shard \(S\), control \(S'\), preprocessing, and score extraction logic can be reused.

The interpretation changes:

- Parent sanity check:
  - Does \(P\) reveal membership signal for \(S\)?
- Target provenance check:
  - Does \(T\) still reveal the same shard signal?
- Provenance claim:
  - Under the causal assumption that \(T\) could not access \(S\) except through \(P\), detectable dependence on \(S\) supports the claim that \(T\) was derived from \(P\).

Do not make the provenance claim from the parent sanity check alone. The parent sanity check only validates that the shard signal is detectable in the parent.

---

## Immediate Prompt for Coding Agent

```text
Please edit/scaffold the repository to implement the MIMIR GitHub Pythia parent-membership sanity check using the efficient MIA baseline setup from the paper "Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models."

Use the paper's dataset and preprocessing setup as closely as possible:
- official MIMIR GitHub member/nonmember pair;
- base Pythia model trained on the Pile;
- 32-word truncation for the first run;
- MIN-K% PROB with k=20 as the primary distinguisher score;
- also compute mean logprob/PPL and k in {5,10,40} for diagnostics.

The primary distinguisher should be:
A_tau(x) = 1{min_k_20_logprob_P(x) >= tau}
where tau is chosen only on the construction/train split.

Report the final held-out test accuracy on S_test versus S'_test, where:
- S is the MIMIR GitHub member shard;
- S' is the MIMIR GitHub nonmember/control shard.

Also report balanced accuracy, AUC, TPR@1%FPR, and empirical shard advantage.

Do not implement the GRU as the first priority. Do not run the full model-provenance experiment yet. The first milestone is a clean parent-model sanity check that confirms the membership signal exists in base Pythia.
```

---

## Expected Outcome

The expected deliverable is:

```text
outputs/reports/mimir_github_pythia_mink/summary.md
```

showing whether the efficient MIN-K/PPL-style membership baseline can distinguish official MIMIR GitHub members from nonmembers for base Pythia.

If this works, the pipeline is ready to be generalized toward the model-provenance experiment.

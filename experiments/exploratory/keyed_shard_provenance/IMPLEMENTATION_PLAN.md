# Implementation plan: keyed dataset shard provenance

Status: initial Table 8 implementation is in
`experiments/table_08_controlled_provenance/`. This file remains the broader
research checklist; the implementation README states the currently supported
profiles and limitations.

## Objective and scope

Use a secret random selection of existing code prompt–response examples to
construct a parent, derive downstream targets, and evaluate them with the
paper's existing MIN-K threshold distinguisher. Start with Pythia-1B and
Pythia-1.4B. Make model size configurable for a later 6.9B run, but do not add
that run to the initial scope.

Notation: B is the original base, P is B fine-tuned on S, T_D is P fine-tuned
on downstream dataset D, and C_D is B fine-tuned on the same D.
The user's description called the constructed parent P-zero; P here denotes
that same checkpoint, avoiding confusion with the attachment's notation.

This follows the user's corrected construction: select existing examples.
The attached plan's teacher-sampling construction is superseded. No teacher
generation or watermark is required.

## 1. Inspect and freeze existing audit behavior

- Read repository instructions, inspect the worktree, and preserve unrelated work.
- Trace the current paper runner, preprocessing, MIN-K20 calculation, threshold
  selection, held-out evaluation, and reported statistical cutoff.
- Reuse that distinguisher and its calibration procedure. Do not silently change
  token masking, truncation, score direction, or threshold transfer behavior.
- Record that this auditor requires token log probabilities; locally loaded
  weights implement the scoring interface. Do not claim text-only API access.

## 2. Freeze the source dataset and sampling protocol

- Source decision: use 2023 InstructCoder instead of older CodeSearchNet. A
  direct comparison against all ten available converted Pile GitHub partial
  shards found 303 old-pilot-eligible CodeSearchNet function matches and four
  InstructCoder rows with a full input/output code match. The four latter rows
  are excluded by ID. See Table 8's `overlap_checks/README.md` for method and
  the incomplete-corpus limitation. InstructCoder offers instruction + existing
  code as prompt and edited code as response.
- Verify downloadable files, version/revision, schema, licenses, and usable counts.
- Clean empty examples and exact duplicates before random assignment. The
  current pilot does not implement near-duplicate clustering; inspect that
  before any confirmatory claim.
- Use a fixed eligible pool and a secret keyed random ordering of stable record
  identifiers. Select equal-sized, disjoint S and S-prime from that ordering.
  Persist dataset fingerprints and assignments; keep the key out of public logs.
- Do not assign S and S-prime to different original dataset splits or different
  domains. InstructCoder does not expose repository identifiers, so repository
  overlap cannot be measured with this source.
- Initial pilot proposal: 20,000 examples in each group and a 384-token total
  training sequence budget. Measure token lengths and define a deterministic
  filtering/formatting policy that preserves a usable response before freezing it.
- Share the exact same S/S-prime assignment across the 1B and 1.4B families.
- Separately partition audit records into calibration and final evaluation
  groups. Proposed final size: 2,000 examples per group; reserve 2,000 per group
  for calibration. The parent trains on ALL of S, including S examples reserved
  for final auditing. 'Held out' means held out from auditor fitting, not from
  parent training. No model in this experiment trains on S-prime.
- Public GitHub content may already have appeared in base pretraining. Describe
  the intervention as additional exposure to a randomly selected shard, not
  guaranteed first-ever exposure. Audit the base as a negative control.

## 3. Verify downstream dataset provenance

Use the existing paper's family inventory as the authority for target profiles.
Inspect original training code where available, beyond dataset tags in cards.

| Profile | Dataset evidence | Implementation requirement |
| --- | --- | --- |
| 1B Leogrin HH-SFT | Anthropic/hh-rlhf; card states one epoch | Verify formatting, selected responses, and splits |
| 1.4B Helpful-SFT | Helpful subset of Anthropic/hh-rlhf; one epoch | Verify exact helpful subsets and recipe |
| 1.4B Helpful-DPO | Helpful subset of Anthropic/hh-rlhf; one epoch | Verify SFT initialization, reference model, and DPO recipe; preserve any prerequisite stages |
| 1.4B Tulu | allenai/tulu-v2-sft-mixture; three epochs | Verify template, revisions, and complete recipe |
| 1.4B LaMini | lamini/lamini_docs is plausible; model card says unknown | Do not label as exact reproduction without further evidence |
| 1.4B Finance | Financial continued pretraining plus instruction tuning | Exact corpus/augmentation is not established publicly; resolve or explicitly label approximation |

- Prepare verified profiles first; report unresolved profiles rather than silently
  replacing their data. LaMini and finance require a user decision before using
  an approximate recipe.
- Check downstream data against S and S-prime for exact and near overlaps. Fix
  any exclusions before training; apply the identical resulting D to T_D and C_D.
- Using a smaller subset or fewer epochs for runtime is a reduced-budget
  reproduction, even if the dataset source matches. Record that distinction.
- Source links:
  - https://huggingface.co/datasets/likaixin/InstructCoder
  - https://huggingface.co/Leogrin/eleuther-pythia1b-hh-sft
  - https://huggingface.co/lomahony/pythia-1.4b-helpful-sft
  - https://huggingface.co/lomahony/pythia-1.4b-helpful-dpo
  - https://huggingface.co/kykim0/pythia-1.4b-tulu-v2-mix
  - https://huggingface.co/herMaster/pythia1.4B-finetuned-on-lamini-docs
  - https://huggingface.co/OVHaiLLM/fin-pythia-1.4b
  - https://arxiv.org/abs/2401.14777

## 4. Pilot and measure cost

- Test data preparation and the complete pipeline with a tiny local model first.
- Benchmark representative GPU training/scoring steps, including memory, on the
  actual cluster. Estimate each parent, target, control, and audit job separately.
- Do not promise sub-hour completion or sufficient signal from example count alone.
- Use a separate pilot selection key and data to choose parent training strength
  and a reasonable token/epoch budget. Keep MIN-K as the requested distinguisher;
  report weak signal instead of selecting another scorer after looking at results.
- Confirm parent signal and sensible base/control behavior in the pilot. A larger
  shard does not automatically give stronger per-example memorization.
- Freeze the recipe before drawing the fresh confirmatory assignment; never
  retune on confirmatory final audit results or retry keys until significance.

## 5. Implement modular training and jobs

- Separate configuration, dataset preparation, keyed splitting, training, audit
  orchestration, and result compilation. Reuse existing audit utilities.
- Train one P per base and run seed, then reuse it for all downstream profiles.
- Train T_D from P and C_D from B with matched downstream data, seed, optimizer,
  steps, formatting, and training procedure. Honor DPO prerequisite stages.
- Save final model checkpoints and metadata; do not create halfway result rows.
- Provide separate submit-ready jobs for parent construction and each downstream
  profile, with dependencies, resumability checks, and isolated output paths.
- Do not start expensive jobs until the confirmed configuration and measured
  resource estimates are available.

## 6. Statistical review and final audit

- Random sampling without replacement makes disjoint S/S-prime dependent. The
  attachment's independent teacher-output proof cannot be copied unchanged.
  Check the paper's null and a valid finite-population/randomization argument
  before asserting the same Hoeffding cutoff controls false positives.
- Keep the requested distinguisher fixed. If the sampling scheme needs a different
  null calibration, explain and resolve that before confirmatory evaluation.
- Fit audit thresholds only on calibration records following the existing paper
  protocol. Freeze them before evaluating final records.
- Audit B, P, every T_D, and every C_D against the same S/S-prime audit split.
- Record TPR, FPR, signed difference, absolute shard advantage, ROC AUC, actual
  held-out counts, applicable significance cutoff, and reject/fail-to-reject.
- If justified for this design, the attachment's cutoff is sqrt(log(2/alpha)/m).
  At alpha=.05 and m=2,000 this is approximately .043. Do not silently substitute
  a new m into an incompatible existing implementation.
- Predefine primary comparisons and how multiple targets/seeds are handled;
  distinguish per-test significance from any family-wide claim.
- Report all outcomes, including base false positives, absent parent signal, and
  control rejections. Final parent failure does not authorize tuning and retesting.
- Include downstream validation loss/task metrics and timing to show that
  downstream training occurred and to contextualize retained audit signal.

## 7. Verification and delivery

- Verify deterministic selection, S/S-prime disjointness, duplicate handling,
  calibration/final separation, downstream exclusions, and model initialization.
- Test metric calculations, frozen threshold reuse, resume/config mismatch
  detection, and the full tiny-model parent/target/control/audit pipeline.
- Run a representative real-model GPU smoke test and validate scheduler scripts.
- Deliver documented configs, submission commands, measured time/memory estimates,
  and a compact report mapping every result to its checkpoint and data manifest.
- Never describe mocked/tiny-model tests as a successful full production run.

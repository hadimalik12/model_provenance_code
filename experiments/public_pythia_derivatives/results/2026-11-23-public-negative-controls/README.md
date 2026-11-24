# Public negative controls for Reviewer 2, Question 1

This run adds two separate controls to the public MIMIR GitHub witness audit.

1. `EleutherAI/gpt-neo-1.3B` is an independently developed non-Pythia model
   trained on the Pile.  It tests whether shared pretraining data alone can make
   the audit detect the GitHub witness.
2. `allenai/tulu-2-7b` is a Llama 2 model fine-tuned on the Tulu V2 SFT
   mixture.  It is compared with the already-scored
   `kykim0/pythia-1.4b-tulu-v2-mix`, which documents the same downstream
   mixture.  This tests whether sharing downstream instruction data alone
   reproduces the Pile-witness result across model families.

The suggested `allenai/tulu-7b` checkpoint was not used because it is a weight
diff that requires separate access to the original restricted LLaMA weights.
`allenai/tulu-2-7b` is a directly loadable public checkpoint and provides the
same intended cross-family, same-downstream-data comparison with Tulu V2.

Both new models use the exact frozen matched split already used by the public
audit: 680 construction records and 800 held-out evaluation records. MIN-K
20 percent is primary and mean log probability is secondary. Both scores come
from the same model forward passes. Score direction, thresholds, sample size,
and checkpoints were fixed before evaluation.

Validate and dry-run without a GPU:

```bash
python scripts/setup/validate_experiment_config.py \
  experiments/public_pythia_derivatives/configs/aistats27_public_negative_controls.yaml
python experiments/public_pythia_derivatives/runners/run_public_negative_controls.py \
  --dry-run
```

Run the non-scientific eight-example smoke test:

```bash
PUBLIC_CONTROL_MAX_EXAMPLES=8 \
PUBLIC_CONTROL_ARTIFACT_ROOT=artifacts/public_pythia_derivatives/public_negative_controls/smoke \
sbatch --account=gts-vzikas3 \
  experiments/public_pythia_derivatives/jobs/run_public_negative_controls.sbatch
```

Run the full held-out controls after the smoke test succeeds:

```bash
sbatch --account=gts-vzikas3 \
  experiments/public_pythia_derivatives/jobs/run_public_negative_controls.sbatch
```

The successful cluster submission used Llama-only smoke job `16073197` and
full job `16073200`. The full job had an `afterok:16073197` dependency. Both
completed successfully on one RTX6000.

The first smoke submission, job `16070016`, failed before model loading because
the batch script invoked the system Python instead of the project environment.
It used 28 GPU-seconds. Blocked dependent job `16070033` was cancelled. The
launcher now invokes `.venv_pace/bin/python` explicitly.

Corrected smoke job `16071062` successfully completed the GPT-Neo scoring and
audit, then exposed a missing `sentencepiece` dependency while loading the
Llama tokenizer. Its blocked full job `16071063` was cancelled. SentencePiece
is now pinned in `requirements.txt`, non-Pythia models cannot silently use a
Pythia tokenizer fallback, and the final smoke used Tulu's native SentencePiece
tokenizer. GPT-Neo's smoke outputs were preserved.

## Observed held-out results

The table reports 400 held-out members and 400 held-out nonmembers. The
single-audit cutoff is 0.0960. The two pre-specified primary control audits use
a Bonferroni family-wise cutoff of 0.1047; the two secondary-score audits form
a separate exploratory family with the same adjusted cutoff. Every new result
below remains positive after its applicable family-wise correction.

| Model | Role | Score | Signed advantage | AUC | Adjusted decision |
|---|---|---|---:|---:|---|
| GPT-Neo-1.3B | Same Pile, unrelated model | MIN-K 20% | 0.3050 | 0.7322 | Detects witness |
| Tulu-2-7B | Llama family, same downstream mixture | MIN-K 20% | 0.2725 | 0.6968 | Detects witness |
| GPT-Neo-1.3B | Same Pile, unrelated model | Mean log probability | 0.3900 | 0.7555 | Detects witness |
| Tulu-2-7B | Llama family, same downstream mixture | Mean log probability | 0.3350 | 0.7183 | Detects witness |
| Pythia-1.4B Tulu V2 | Cached comparator | MIN-K 20% | 0.1200 | 0.5814 | Detects witness |
| Pythia-1.4B Tulu V2 | Cached comparator | Mean log probability | 0.0675 | 0.5523 | No detection |

Shuffled-label advantages were 0.0050 and 0.0300 for GPT-Neo and 0.0100 and
0.0075 for Tulu-2-7B, all below the cutoff. This indicates that the positive
results are not caused by the audit implementation retaining the labels.

The successful production job used 9 minutes 48 seconds of RTX6000 time. All
smoke and failed setup attempts plus production used 28 minutes 23 seconds,
about 0.47 GPU-hours or approximately $0.10 at the published internal rate.

These controls directly show that the public audit can fire without Pythia
lineage. GPT-Neo shares Pile pretraining. More importantly, Llama-based Tulu
also fires and is stronger than the cached Pythia-Tulu comparator. Together
with the measured MIMIR split distinguishability, the public results should be
described as witness or split-signal detection, not proof of causal lineage.

A positive result means that the model carries detectable information about
the Pile witness split. It does not prove exact parentage or identify the route
by which the model accessed the data.

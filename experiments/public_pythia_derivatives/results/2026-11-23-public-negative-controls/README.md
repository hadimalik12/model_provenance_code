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

The actual cluster submission uses smoke job `16070016` and full job
`16070033`. The full job has an `afterok:16070016` dependency, so it cannot
spend GPU time unless the smoke job succeeds. Both request one RTX6000. At the
published internal rate of about $0.22 per GPU-hour, the two-hour hard limit on
each job caps their combined charge below $0.90; expected usage is lower.

A positive result means that the model carries detectable information about
the Pile witness split. It does not prove exact parentage or identify the route
by which the model accessed the data.

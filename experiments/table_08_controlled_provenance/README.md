# Table 8: controlled Pythia shard provenance

## Experiment profiles

Table 8 now has two sampling experiments:

- [`keyed_rank/`](keyed_rank/): the legacy keyed-rank run plus the additional
	LaMini and Finance SFT runs using the same sampler and distinguisher.
- [`global_shuffle/`](global_shuffle/): the active global-shuffle run.

The Python implementation and Slurm job templates remain shared at this level;
each profile owns its configuration and exposes its artifact runs through a
`runs/` directory.

This experiment creates its own lineage from Pythia-1B and Pythia-1.4B. The
source is the Python train split of
[CodeSearchNet](https://huggingface.co/datasets/claudios/code_search_net), pinned
to revision `a4ee8053e119ef04ee3d491ffef05bcf2725ae82`. Its documentation is the
prompt and its function code is the response. The configured sampler selects
equal, disjoint S and S-prime sets. The parent trains on all of S; S-prime is
unused by every training stage. The active configuration uses `global_shuffle`:
after preprocessing and deduplication, the complete eligible pool is shuffled
once with a private seeded RNG and split into consecutive arms. The legacy
`keyed_rank` sampler remains available by changing `source.sampling_method`.

Before selection, the full eligible CodeSearchNet pool was compared with the
416,000 available converted Pile GitHub documents. Text was normalized with
Unicode NFKC, case folding, and removal of whitespace and NUL characters. Any
record whose complete normalized prompt or complete normalized response
appeared in a checked Pile document was removed. This excluded 6,019 records
and left 237,918; the saved pool has zero detected matches under this rule.
The submission preflight verifies its checksum and filter manifest. See the
[overlap audit](overlap_checks/README.md) for the method and counts.

This establishes no detected overlap with the downloaded Pile GitHub sample.
It cannot establish zero overlap with the remaining roughly 97.8% of Pile's
GitHub documents, Pile-CC, or other Pile components that were not downloaded.

For each model size: `base → parent(S) → target(D)` and `base → control(D)`. The target and control use the same prepared downstream file and hyperparameters. The audit runs the repository's existing MIN-K20 scoring and balanced-accuracy threshold distinguisher on the base, parent, target, and control. The test split is held out from *threshold fitting*, although its S rows were used to train the parent.

## Submit together

From the repository root, submit the active global-shuffle profile:

```bash
./experiments/table_08_controlled_provenance/global_shuffle/jobs/submit.sh
```

The script first validates the filtered source and refuses a nonempty run
directory. It then submits one CPU source job. After that succeeds, the two
parent jobs, two base audits, and three downstream-data jobs become eligible.
Each target waits for both its parent and downstream data; each control waits
only for downstream data. Target/control audits wait for their checkpoint, and
the final report waits for all audits to settle. Independent jobs may run in
parallel as GPUs become available. Each GPU job loads one checkpoint and has
an eight-hour limit. Submit from the repository root; `TABLE8_PYTHON` and
`TABLE8_CONFIG` may override the defaults. Do not submit the same artifact
root twice.

The active downstream profiles are the 1B HH SFT dataset, the 1.4B helpful SFT subset, and the 1.4B Tulu SFT mixture. They use capped 4,000-example, one-epoch training for tractability. These are **reduced-budget reconstructions** on the original dataset sources, not numerical reproductions of the public checkpoints. The 1.4B Tulu public model was trained for three epochs on the full mixture. Finance needs unavailable/proprietary training material, LaMini's model card does not establish its exact dataset, and DPO requires a separate preference-training recipe. Those three public-model profiles are outside this submit script until their derivation can be supported without substituting a different experiment.

For a controlled-data comparison that does not attempt to reproduce those
public checkpoints, the keyed-rank experiment's
`keyed_rank/configs/additional_sft.yaml` adds two plainly labeled SFT
interventions: `lamini_sft` uses `MBZUAI/LaMini-instruction`, and
`finance_sft` uses `gbharti/finance-alpaca`. Both use the same capped
4,000-example, one-epoch recipe as the other downstream profiles. Submit them
to the keyed-rank artifact root with:

```bash
./experiments/table_08_controlled_provenance/keyed_rank/jobs/submit_additional_sft.sh
```

The profile configs propose 20,000 examples per S/S-prime arm with 2,000 per
arm for calibration and 2,000 per arm for final audit. Runtime and signal
strength have not been measured on the production GPUs. Treat this run as a
pilot, not a confirmatory result. For a confirmatory experiment, generate a
fresh selection key and artifact root after the recipe is fixed.

The private key is stored at each run's `prepared/selection_key.hex` with mode
0600. Its value is never printed. `prepared/manifest.json` records the source
revision, selected-set hashes, and row counts. Every model checkpoint has
`complete.json` with its data source, loss, duration, peak GPU memory, and
configuration fingerprint. Reports and per-checkpoint JSON are under the
profile-specific artifact roots.

The existing scorer computes MIN-K20 over the formatted prompt plus response. Parent training masks the prompt and updates on response tokens. The audit uses model token probabilities, so this is logprob-access auditing. For disjoint randomized groups, Table 8 reports an exact two-sided fixed-threshold label randomization p-value; the paper's Hoeffding cutoff is included only as a reference pending a formal comparison of sampling assumptions. Multiple target rows have individual p-values, without a family-wide error claim.

The shared implementation is under `shared/`: `prepare.py` (shard sampling),
`downstream.py` (shared D files), `train.py` (one model stage), `audit.py` (one
checkpoint's scores and metrics), and `compile.py` (available-result report).
The code refuses to treat partial model or score outputs as finished and checks
configuration fingerprints on resume.

The retained pool excludes complete-field matches under the checked Pile
reference scope. This does not establish zero overlap with unscanned Pile data.

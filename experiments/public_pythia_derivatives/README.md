# Public Pythia derivative audit

This package reproduces the public-LLM witness experiment in Section 5.2.1,
main-paper Table 2, and Appendix D.2 Table 6 of the submitted paper. The audit
covers the exact 14 public derivatives reported there:

| Parent family | Derived targets |
|---|---:|
| Pythia-1B | 1 |
| Pythia-1.4B | 6 |
| Pythia-6.9B | 5 |
| Pythia-12B | 2 |

The candidate shard is MIMIR GitHub. For every target, MIN-K 20% is calibrated
on 560 secret-shard and 340 control-shard construction examples and evaluated
on 400 held-out examples per shard. The level-0.05 reference threshold is
`0.096`. The same split also audits each candidate parent as a positive sanity
check, and every target receives shuffled-label and nonmember-vs-nonmember
controls.

The submitted values are versioned in
[`results/submitted_table_6.csv`](results/submitted_table_6.csv). The first six
1B/1.4B targets are also used by the INT8 experiment in
[`../pythia_int8_quantization/`](../pythia_int8_quantization/).

## Run

Submit the three size groups and the dependent report job:

```bash
python scripts/setup/validate_experiment_config.py \
  experiments/public_pythia_derivatives/configs/paper.yaml
bash experiments/public_pythia_derivatives/submit.sh
```

Optional scheduler routing is supplied without editing committed files:

```bash
PYTHIA_ACCOUNT=<account> \
PYTHIA_GPU_PARTITION=<gpu-partition> \
bash experiments/public_pythia_derivatives/submit.sh
```

The submission script prepares the deterministic shared split once, then runs
the three model-family jobs in parallel. The report job runs only after all
three size groups succeed and writes:

```text
artifacts/public_pythia_derivatives/paper_audit/reports/
  public_pythia_audit.csv
  public_pythia_audit.md
```

For an interactive GPU allocation, the same pipeline can be run directly:

```bash
.venv_pace/bin/python \
  experiments/public_pythia_derivatives/runners/run_audit.py \
  --seeds 0 --batch-size 4 --dtype float16 --resume
.venv_pace/bin/python \
  experiments/public_pythia_derivatives/reports/compile_results.py
```

`HF_TOKEN` may be supplied for gated Hugging Face assets. Generated datasets,
scores, and model outputs remain under `artifacts/` and are not committed.

The submitted `nnheui/pythia-1.4b-sft-full` result remains in the bundled
table, but its original Hugging Face checkpoint is no longer available. The
runner records that target explicitly and stops with a clear error rather than
silently substituting a different archive.

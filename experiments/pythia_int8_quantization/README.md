# Pythia INT8 quantization audit

This package reproduces the LLM.int8 transformation experiment summarized in
main-paper Table 2 and reported model-by-model in Appendix D.2 Table 7.

Six public Pythia derivatives are evaluated before and after LLM.int8
post-training quantization. The experiment holds fixed the MIMIR GitHub
secret/control shards, the 560/340 construction split, the 400-per-shard
held-out evaluation split, and the MIN-K 20% distinguisher. The level-0.05
reference threshold is `0.096`.

The exact submitted values are versioned in
[`results/submitted_table_7.csv`](results/submitted_table_7.csv).

## Run

Submit the complete six-target experiment:

```bash
sbatch experiments/pythia_int8_quantization/jobs/run_paper_targets.sbatch
```

Specify site-specific scheduler settings on the command line when needed:

```bash
sbatch --account=<account> --partition=<gpu-partition> \
  experiments/pythia_int8_quantization/jobs/run_paper_targets.sbatch
```

The job writes all generated data under
`artifacts/pythia_int8_quantization/paper_targets_seed0/` and compiles:

```text
reports/pythia_int8_quantization.csv
reports/pythia_int8_quantization.md
```

`jobs/smoke_int8_loader.sbatch` is a two-example numerical preflight.
`jobs/run_int8_robustness.sbatch` is a focused single-target diagnostic using
the same runner; neither is a separate paper experiment.

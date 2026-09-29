# LLM Quantization Robustness

This reviewer-response experiment tests whether the Pythia-1.4B GitHub shard
provenance signal survives post-training 8-bit weight quantization of a known
derived target.

The fixed comparison is:

- parent: `EleutherAI/pythia-1.4b` in FP16;
- target baseline: `herMaster/pythia1.4B-finetuned-on-lamini-docs` in FP16;
- transformed target: the same target loaded with bitsandbytes LLM.int8. Its
  non-linear/residual modules use FP32, while the linear weights remain INT8,
  to prevent non-finite logits observed with FP16 residuals on PACE GPUs.

All three representations use the same MIMIR GitHub construction and held-out
splits. MIN-K 20% is the sole audit and reporting metric, matching the generated
Table 2 setup. Both target representations also receive shuffled-label and
nonmember-vs-nonmember controls.

## Reproduction

Submit:

```bash
sbatch experiments/robustness_quantization/jobs/run_int8_robustness.sbatch
```

To run the exact six derived targets used by Table 2's generated vanilla audit,
submit `jobs/run_table2_targets_int8_robustness.sbatch`. Each target gets a
separate artifact root and is paired with its matching 1B or 1.4B parent.
After all six targets finish, compile their shared report with:

```bash
python experiments/robustness_quantization/reports/compile_table2_targets.py
```

Monitor:

```bash
squeue -u "$USER"
tail -f experiments/robustness_quantization/logs/quantization_int8_<jobid>.out
```

Results are written under
`artifacts/robustness_quantization/int8_pythia1_4b_lamini_seed0/`.

To reproduce the multi-target reviewer-response table, run and compile:

```bash
sbatch experiments/robustness_quantization/jobs/run_table2_targets_int8_robustness.sbatch
python experiments/robustness_quantization/reports/compile_table2_targets.py
```

The batch job fixes the six Table 2 targets, their matching Pythia parents,
seed 0, the MIMIR GitHub splits, and the LLM.int8 configuration. The compiler
writes the paper-style Markdown and machine-readable CSV to
`artifacts/robustness_quantization/table2_targets_seed0/reports/`.

## Artifact layout

The multi-target run follows the shared experiment-root layout used by Table 5:

```text
artifacts/robustness_quantization/table2_targets_seed0/
  prepared/{mimir_github,nonmember_control}/
  scores/{main,nonmember_control}/<target>/<representation>/
  results/{main,nonmember_control}/<target>/<representation>/
  run_specs/<target>.json
  config_locks/<target>.yaml
  reports/table2_target_quantization_robustness.{md,csv}
```

Prepared splits are shared across all six targets. The only paper-facing
multi-target report is the consolidated table under `reports/`; raw scores and
per-target audit results remain under `scores/` and `results/`.

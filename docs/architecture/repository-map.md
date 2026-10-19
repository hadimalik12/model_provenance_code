# Repository Map

The project has three layers. Keeping them separate is the main organizational
rule.

```text
experiments/<name>/       What a reader runs and why it exists
        configs/          Declarative run settings
        jobs/             Slurm orchestration only
        reports/          Experiment-specific interpretation and compilation

scripts/                  Reusable command-line building blocks
        data/             Prepare a dataset split
        scoring/          Produce per-example scores
        cluster/          Configure the HPC environment
        setup/            Validate environment and configs

src/shard_audit/          Python implementation shared by all commands
```

Do not add new batch scripts to `scripts/`, and do not add experiment-specific
constants to generic scoring or audit modules. Experiment directory names
describe the scientific question rather than a mutable manuscript table number.

## Legacy Material

All generated state belongs in `artifacts/`; downloaded Hugging Face data lives
in `.cache/huggingface/`. The prior `data/`, `outputs/`, and `scripts/reports/`
folders have been removed because they mixed generated state, orchestration, and
report generation. Experiment-specific code lives in its experiment package.

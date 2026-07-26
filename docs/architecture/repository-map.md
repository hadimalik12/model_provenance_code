# Repository Map

The project has three layers. Keeping them separate is the main organizational
rule.

```text
experiments/<name>/       What a reader runs and why it exists
        configs/          Declarative run settings
        jobs/             Slurm orchestration only
        reports/          Table-specific interpretation and compilation

scripts/                  Reusable command-line building blocks
        data/             Prepare a dataset split
        scoring/          Produce per-example scores
        experiments/      Calibrate and evaluate the distinguisher
        cluster/          Configure the HPC environment
        setup/            Validate environment and configs

src/shard_audit/          Python implementation shared by all commands
```

Do not add new batch scripts to `scripts/`, and do not add table-specific
constants to generic scoring or audit modules. A new research idea starts as a
folder under `experiments/exploratory/`; promote it to a numbered table folder
only once its question and outputs are stable.

## Legacy Material

All generated state belongs in `artifacts/`; downloaded Hugging Face data lives
in `.cache/huggingface/`. The prior `data/`, `outputs/`, and `scripts/reports/`
folders have been removed because they mixed generated state, orchestration, and
report generation. Table-specific code now lives in its experiment package.

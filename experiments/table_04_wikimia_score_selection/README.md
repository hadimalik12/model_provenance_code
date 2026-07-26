# Table 4: WikiMIA Score Selection

This experiment compares membership scores on WikiMIA. It is the real paper
Table 4 home. The model runners are reusable commands under `scripts/experiments`;
the model-specific report compilers are kept here because their interpretation
belongs to this table.

Available runners:

```bash
python experiments/table_04_wikimia_score_selection/runners/run_wikimia_pythia69b_experiment.py
python experiments/table_04_wikimia_score_selection/runners/run_wikimia_llama2_13b_experiment.py
python experiments/table_04_wikimia_score_selection/runners/run_wikimia_null_control.py
```

After scoring, run the matching compiler in `reports/`. A unified Slurm job is
intentionally not supplied yet: the existing model runners have different
resource and access requirements, so pretending they are interchangeable would
hide meaningful operational differences.

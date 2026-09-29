# Table 8: keyed-rank experiment

This is the legacy Table 8 sampling experiment. It preprocesses the filtered
CodeSearchNet pool and selects the lowest keyed ranks for disjoint `S` and
`S-prime` arms.

The historical runs are grouped under `keyed_rank/runs/`:

- `runs/csn_filtered_seed0/`
- `runs/additional_sft_seed0/`

The second run contains the additional LaMini and Finance SFT profiles. They
use the same keyed sampler and distinguisher, so they belong to this experiment
family rather than a separate Table 8 experiment.

Use the keyed experiment wrapper:

```bash
./experiments/table_08_controlled_provenance/keyed_rank/jobs/submit.sh
```

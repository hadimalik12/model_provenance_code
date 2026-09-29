# Table 8: global-shuffle experiment

This is the active Table 8 experiment. It applies the shared preprocessing and
deduplication pipeline, then globally shuffles the complete eligible pool with
a private seeded RNG before taking the same disjoint `S` and `S-prime` arm
sizes.

The run is grouped under:

`global_shuffle/runs/csn_global_shuffle_seed0/`

Use the global-shuffle experiment wrapper:

```bash
./experiments/table_08_controlled_provenance/global_shuffle/jobs/submit.sh
```

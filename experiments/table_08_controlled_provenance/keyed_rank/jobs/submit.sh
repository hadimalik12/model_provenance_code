#!/bin/bash
set -euo pipefail
TABLE8_CONFIG=experiments/table_08_controlled_provenance/keyed_rank/configs/config.yaml \
  bash experiments/table_08_controlled_provenance/shared/jobs/submit_all.sh

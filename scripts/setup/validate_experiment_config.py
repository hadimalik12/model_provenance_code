"""Validate an experiment YAML before submitting an HPC job.

The validator deliberately checks only stable repository conventions. It does
not require a GPU, datasets, or model downloads, so it is safe to run on the
login node.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


REQUIRED_TOP_LEVEL = {"experiment", "dataset", "scoring", "outputs"}
REQUIRED_EXPERIMENT_KEYS = {"id", "purpose", "variant"}
REQUIRED_OUTPUT_KEYS = {"run_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a provenance experiment configuration.")
    parser.add_argument("config", type=Path, help="Path to an experiment YAML file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    errors: list[str] = []
    missing = REQUIRED_TOP_LEVEL - config.keys()
    if missing:
        errors.append(f"missing top-level sections: {', '.join(sorted(missing))}")

    experiment = config.get("experiment", {})
    missing = REQUIRED_EXPERIMENT_KEYS - experiment.keys()
    if missing:
        errors.append(f"missing experiment keys: {', '.join(sorted(missing))}")

    outputs = config.get("outputs", {})
    missing = REQUIRED_OUTPUT_KEYS - outputs.keys()
    if missing:
        errors.append(f"missing outputs keys: {', '.join(sorted(missing))}")

    experiment_id = experiment.get("id", "")
    if experiment_id and "/" in experiment_id:
        errors.append("experiment.id must be a single directory name")

    if errors:
        print(f"Invalid config: {args.config}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Valid config: {args.config}")
    print(f"  experiment: {experiment['id']} ({experiment['variant']})")
    print(f"  purpose:    {experiment['purpose']}")
    print(f"  artifact:   artifacts/{experiment['id']}/{outputs['run_id']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

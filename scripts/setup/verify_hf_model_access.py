"""Fail fast when an experiment references an inaccessible Hugging Face model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shard_audit.reproducibility import model_revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True,
                        help="Hugging Face model ID; may be supplied repeatedly.")
    args = parser.parse_args()

    from transformers import AutoConfig

    token = os.environ.get("HF_TOKEN")
    for model_id in args.model:
        revision = model_revision(model_id)
        config = AutoConfig.from_pretrained(
            model_id, revision=revision, token=token
        )
        print(
            f"Accessible: {model_id}@{revision} "
            f"(model_type={config.model_type}, hidden_size={getattr(config, 'hidden_size', 'n/a')})"
        )


if __name__ == "__main__":
    main()

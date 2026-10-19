"""Fail fast when an experiment references an inaccessible Hugging Face model."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Pinned Hugging Face model as REPO_ID@REVISION; may be repeated.",
    )
    args = parser.parse_args()

    from transformers import AutoConfig

    token = os.environ.get("HF_TOKEN")
    for model_spec in args.model:
        if "@" not in model_spec:
            raise SystemExit(f"Model must include an immutable revision: {model_spec}")
        model_id, revision = model_spec.rsplit("@", 1)
        if len(revision) != 40:
            raise SystemExit(f"Revision must be a 40-character commit: {model_spec}")
        config = AutoConfig.from_pretrained(
            model_id, revision=revision, token=token
        )
        print(
            f"Accessible: {model_id}@{revision} "
            f"(model_type={config.model_type}, hidden_size={getattr(config, 'hidden_size', 'n/a')})"
        )


if __name__ == "__main__":
    main()

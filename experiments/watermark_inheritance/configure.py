"""Create the three immutable run configurations used by the paper experiment."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from src.shard_audit.reproducibility import (
    external_code_spec,
    lock_sha256,
    model_revision,
)

from .common import load_json, write_json


RUNS = (
    ("run_00", 10301, 101, 900101),
    ("run_01", 11301, 203, 900203),
    ("run_02", 12301, 307, 900307),
)


def repository_revision() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def base_config(shared_dir: Path) -> dict:
    shared_dir = shared_dir.resolve()
    model = shared_dir / "pythia-1b"
    return {
        "experiment": "watermark_inheritance",
        "model_id": "EleutherAI/pythia-1b",
        "model_revision": model_revision("EleutherAI/pythia-1b"),
        "model_path": str(model),
        "tokenizer_path": str(model),
        "upstream_path": str(shared_dir / "radioactive-watermark"),
        "shared_dir": str(shared_dir),
        "pairs": 10000,
        "calibration_pairs": 2000,
        "pilot_pairs": 128,
        "prompt_tokens": 64,
        "prompt_seed": 1729,
        "gamma": 0.25,
        "delta": 3.0,
        "ngram": 2,
        "salt_key": 35317,
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 256,
        "epochs": 3,
        "learning_rate": 1e-5,
        "effective_batch_size": 32,
        "max_length": 1024,
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "train_micro_batch": 4,
        "generation_batch_size": 32,
        "score_batch_size": 8,
        "radio_token_budget": 100000,
        "watermark_cache_size": 2048,
        "cpu_threads": 1,
        "pilot": False,
        "code_commit": repository_revision(),
        "upstream_commit": external_code_spec("radioactive_watermark")["revision"],
        "input_lock_sha256": lock_sha256(),
        "overlap_substring_min_chars": 32,
    }


def configure(root: Path, shared_dir: Path) -> list[Path]:
    common = base_config(shared_dir)
    created = []
    for run_id, seed, watermark_key, wrong_key in RUNS:
        run = root / run_id
        config = dict(common, run_id=run_id, seed=seed,
                      watermark_key=watermark_key, wrong_key=wrong_key)
        path = run / "config.json"
        if path.exists() and load_json(path) != config:
            raise RuntimeError(
                f"Refusing to replace a different run configuration: {path}"
            )
        write_json(path, config)
        write_json(run / "config.lock.yaml", config)
        created.append(path)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("artifacts/watermark_inheritance"),
    )
    parser.add_argument(
        "--shared-dir", type=Path, default=Path(".cache/watermark")
    )
    args = parser.parse_args()
    for path in configure(args.root, args.shared_dir):
        print(path)


if __name__ == "__main__":
    main()

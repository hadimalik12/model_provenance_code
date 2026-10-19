"""Immutable input metadata shared by all submitted-paper experiments."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "reproducibility" / "inputs.lock.json"


@lru_cache(maxsize=1)
def input_lock() -> dict:
    """Load and minimally validate the repository's immutable input lock."""
    payload = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported input lock schema: {payload.get('schema_version')}")
    if not isinstance(payload.get("models"), dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError("Input lock must define model and dataset mappings")
    return payload


def lock_sha256() -> str:
    """Return the digest recorded in generated manifests."""
    return hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()


def model_revision(repo_id: str) -> str:
    """Return an immutable model revision or fail instead of following ``main``."""
    models = input_lock()["models"]
    if repo_id not in models:
        raise KeyError(f"Model is missing from {LOCK_PATH}: {repo_id}")
    revision = models[repo_id]
    if not revision:
        details = input_lock().get("known_unavailable_inputs", {}).get(repo_id, {})
        reason = details.get("reason", "no immutable revision is available")
        raise RuntimeError(f"Cannot reproduce {repo_id}: {reason}")
    return str(revision)


def dataset_spec(name: str) -> dict:
    """Return a copy of a pinned dataset specification."""
    datasets = input_lock()["datasets"]
    if name not in datasets:
        raise KeyError(f"Dataset is missing from {LOCK_PATH}: {name}")
    spec = dict(datasets[name])
    revision = spec.get("revision")
    if not revision:
        raise RuntimeError(f"Dataset {name} has no immutable revision")
    return spec


def external_code_spec(name: str) -> dict:
    """Return a copy of a pinned external-code specification."""
    sources = input_lock().get("external_code", {})
    if name not in sources:
        raise KeyError(f"External source is missing from {LOCK_PATH}: {name}")
    return dict(sources[name])

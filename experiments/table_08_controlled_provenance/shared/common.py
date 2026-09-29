"""Shared paths, configuration, and atomic artifact helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "experiments/table_08_controlled_provenance/global_shuffle/configs/config.yaml"


def config(path: str | Path = DEFAULT_CONFIG) -> dict:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    required = {"artifact_root", "source", "training", "downstream", "families", "profiles"}
    if not isinstance(data, dict) or not required <= data.keys():
        raise ValueError(f"Missing required configuration fields: {required - set(data or {})}")
    return data


def artifacts(cfg: dict) -> Path:
    path = Path(cfg["artifact_root"])
    return path if path.is_absolute() else ROOT / path


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_manifest(cfg: dict) -> dict:
    path = artifacts(cfg) / "prepared" / "manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["config_fingerprint"] != fingerprint(cfg):
        raise ValueError(f"Config changed after data preparation: {path}")
    return data

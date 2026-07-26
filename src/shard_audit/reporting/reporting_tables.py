"""Shared helpers for table-replication report scripts."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Iterable


def load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def score_test_metrics(data: dict | None, score_name: str) -> dict:
    """Extract the held-out test metrics for a score from run_mia_experiment output."""
    if not isinstance(data, dict):
        return {}
    for entry in data.get("main_results", []):
        if isinstance(entry, dict) and entry.get("score_name") == score_name:
            return entry.get("test", {})
    if score_name in data and isinstance(data[score_name], dict):
        return data[score_name].get("test", {})
    return {}


def shuffled_control_metrics(data: dict | None, score_name: str) -> dict:
    if not isinstance(data, dict):
        return {}
    for entry in data.get("shuffled_label_control", []) or []:
        if isinstance(entry, dict) and entry.get("score_name") == score_name:
            return {
                "accuracy": entry.get("test_accuracy"),
                "shard_advantage": entry.get("test_advantage"),
                "auc": entry.get("test_auc"),
            }
    return {}


def finite(value, default=None):
    if value is None:
        return default
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return default
    return value


def verdict(advantage: float | None, gamma_value: float, *, signed: bool = False) -> str:
    if advantage is None:
        return "n/a"
    statistic = abs(advantage) if signed else advantage
    return "Reject H0 / Yes" if statistic > gamma_value else "Fail to reject / No"


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def write_csv(path: str, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

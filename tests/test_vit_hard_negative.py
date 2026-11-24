from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from experiments.vit_provenance.runners.run_hard_negative import (
    EXPECTED_MANIFEST,
    stratified_smoke,
    validate_shared_data,
    write_summary,
)


def test_smoke_selection_is_balanced_and_deterministic() -> None:
    records = [
        {"id": f"m{i}", "label": 1} for i in range(5)
    ] + [
        {"id": f"n{i}", "label": 0} for i in range(5)
    ]
    selected = stratified_smoke(records, 2)
    assert [record["id"] for record in selected] == ["m0", "m1", "n0", "n1"]


def test_exact_public_vit_manifest_is_required(tmp_path: Path) -> None:
    for name in ("train.jsonl", "test.jsonl"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(EXPECTED_MANIFEST), encoding="utf-8"
    )
    assert validate_shared_data(tmp_path)["seed"] == 0
    altered = dict(EXPECTED_MANIFEST, seed=1)
    (tmp_path / "manifest.json").write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="submitted protocol"):
        validate_shared_data(tmp_path)


def test_summary_uses_submitted_level_cutoff(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    results = {
        "main_results": [
            {
                "score_name": "clv_score",
                "test": {"n_pos": 100, "shard_advantage": 0.25},
            }
        ]
    }
    results_path = results_dir / "results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")
    summary = write_summary(tmp_path, results_path, smoke=False)
    assert summary["audit_cutoff"] == pytest.approx(math.sqrt(math.log(40) / 100))
    assert summary["decision"] == "reject"
    assert summary["scientific_result"] is True

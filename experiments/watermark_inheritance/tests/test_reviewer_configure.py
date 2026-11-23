import json

from experiments.watermark_inheritance.reviewer_configure import configure


def test_reviewer_configs_separate_model_and_shard_seeds(tmp_path, monkeypatch):
    study = {
        "dolly_units": [{"unit": "u", "model_seed": 1, "shard_seed": 2, "watermark_key": 3, "wrong_key": 4}],
        "ultrachat_units": [{"unit": "run_00", "model_seed": 5, "shard_seed": 6, "watermark_key": 7, "wrong_key": 8}],
        "precision_plan": {
            "dolly_independent_units_after_completion": 12,
            "dolly_zero_alarm_one_sided_95_percent_upper": 0.220922,
        },
    }
    path = tmp_path / "study.json"
    path.write_text(json.dumps(study))
    monkeypatch.setattr("experiments.watermark_inheritance.configure.repository_revision", lambda: "test")
    manifest = configure(tmp_path / "runs", tmp_path / "shared", path)
    config = json.loads((tmp_path / "runs/dolly/u/config.json").read_text())
    assert config["model_seed"] == 1
    assert config["shard_seed"] == 2
    assert config["independent_unit"] == "model:1|shard:2"
    assert len(config["reviewer_source_sha256"]) == 64
    assert manifest["model_seeds"] == [1, 5]
    assert manifest["shard_seeds"] == [2, 6]

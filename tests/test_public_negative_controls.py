from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_public_control_roster_and_revisions_are_frozen() -> None:
    from experiments.public_pythia_derivatives.runners.run_public_negative_controls import (
        CACHED_PYTHIA_TULU,
        MODELS,
        PRIMARY_SCORE,
        SECONDARY_SCORE,
    )

    assert MODELS["gpt_neo"].model_id == "EleutherAI/gpt-neo-1.3B"
    assert len(MODELS["gpt_neo"].revision) == 40
    assert MODELS["tulu_llama"].model_id == "allenai/tulu-2-7b"
    assert len(MODELS["tulu_llama"].revision) == 40
    assert CACHED_PYTHIA_TULU["model_id"] == "kykim0/pythia-1.4b-tulu-v2-mix"
    assert PRIMARY_SCORE == "min_k_20_logprob"
    assert SECONDARY_SCORE == "mean_logprob"


def test_public_control_dry_run_freezes_inputs_and_does_not_load_models(tmp_path: Path) -> None:
    artifact_root = tmp_path / "dry_run"
    completed = subprocess.run(
        [
            sys.executable,
            "experiments/public_pythia_derivatives/runners/run_public_negative_controls.py",
            "--models", "gpt_neo",
            "--artifact-root", str(artifact_root),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((artifact_root / "run_manifest.json").read_text())
    assert manifest["scientific_result"] is True
    assert manifest["splits"]["train.jsonl"]["rows"] == 680
    assert manifest["splits"]["test.jsonl"]["rows"] == 800
    assert "score_causal_lm_logprobs.py" in completed.stdout
    assert not (artifact_root / "gpt_neo" / "scores").exists()

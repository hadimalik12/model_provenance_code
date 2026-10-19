"""Fail-fast validation for immutable paper-experiment reproduction inputs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shard_audit.reproducibility import LOCK_PATH, input_lock, lock_sha256


SHA = re.compile(r"^[0-9a-f]{40}$")
REQUIREMENTS = REPO_ROOT / "reproducibility" / "requirements-lock.txt"


def locked_versions() -> dict[str, str]:
    result = {"torch": "2.2.0", "torchvision": "0.17.0"}
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        result[name.lower()] = version
    return result


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_lock(payload: dict) -> list[str]:
    problems = []
    for name, spec in payload["datasets"].items():
        revision = str(spec.get("revision", ""))
        if not SHA.fullmatch(revision):
            problems.append(f"dataset {name} is not pinned to a 40-character commit")
    known_unavailable = payload.get("known_unavailable_inputs", {})
    for repo_id, revision in payload["models"].items():
        if revision is None and repo_id in known_unavailable:
            problems.append(f"model {repo_id} is unavailable: {known_unavailable[repo_id]['reason']}")
        elif not SHA.fullmatch(str(revision or "")):
            problems.append(f"model {repo_id} is not pinned to a 40-character commit")
    for name, spec in payload.get("external_code", {}).items():
        if not SHA.fullmatch(str(spec.get("revision", ""))):
            problems.append(f"external source {name} is not pinned to a commit")
    return problems


def validate_environment() -> list[str]:
    problems = []
    for name, expected in locked_versions().items():
        observed = installed_version(name)
        normalized = observed.split("+", 1)[0] if observed else None
        if normalized != expected:
            problems.append(f"package {name}: expected {expected}, found {observed or 'missing'}")
    return problems


def validate_online(payload: dict) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    problems = []
    for name, spec in payload["datasets"].items():
        try:
            info = api.dataset_info(spec["repo_id"], revision=spec["revision"])
            if info.sha != spec["revision"]:
                problems.append(f"dataset {name}: resolved {info.sha}, expected {spec['revision']}")
        except Exception as exc:  # network and authorization diagnostics belong in one report
            problems.append(f"dataset {name}: {type(exc).__name__}: {exc}")
    for repo_id, revision in payload["models"].items():
        if revision is None:
            continue
        try:
            info = api.model_info(repo_id, revision=revision)
            if info.sha != revision:
                problems.append(f"model {repo_id}: resolved {info.sha}, expected {revision}")
        except Exception as exc:
            problems.append(f"model {repo_id}: {type(exc).__name__}: {exc}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="Resolve every pinned Hugging Face commit")
    parser.add_argument(
        "--allow-known-unavailable",
        action="store_true",
        help="Report known missing historical assets without failing",
    )
    parser.add_argument(
        "--skip-environment",
        action="store_true",
        help="Validate input metadata without comparing installed packages",
    )
    args = parser.parse_args()

    payload = input_lock()
    problems = validate_lock(payload)
    unavailable_prefix = "model nnheui/pythia-1.4b-sft-full is unavailable:"
    if args.allow_known_unavailable:
        problems = [problem for problem in problems if not problem.startswith(unavailable_prefix)]
    if not args.skip_environment:
        problems.extend(validate_environment())
    if args.online:
        problems.extend(validate_online(payload))

    report = {
        "ok": not problems,
        "input_lock": str(LOCK_PATH.relative_to(REPO_ROOT)),
        "input_lock_sha256": lock_sha256(),
        "requirements_lock": str(REQUIREMENTS.relative_to(REPO_ROOT)),
        "problems": problems,
    }
    print(json.dumps(report, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

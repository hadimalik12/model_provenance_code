"""Score one checkpoint, then use the paper's fixed MIN-K threshold auditor."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from .common import DEFAULT_CONFIG, ROOT, artifacts, config, fingerprint, read_jsonl, require_manifest, write_json


def checkpoint(cfg: dict, family: str, stage: str, profile: str | None) -> tuple[str, Path]:
    family_root = artifacts(cfg) / "models" / family
    if stage == "base":
        return cfg["families"][family], artifacts(cfg) / "audits" / family / "base"
    if stage == "parent":
        return str(family_root / "parent" / "final"), artifacts(cfg) / "audits" / family / "parent"
    if profile not in cfg["profiles"][family] or stage not in {"target", "control"}:
        raise ValueError("Unconfigured audit stage/profile")
    return str(family_root / profile / stage / "final"), artifacts(cfg) / "audits" / family / profile / stage


def two_sided_randomization_p(tp: int, fp: int, m: int) -> float:
    """Exact fixed-threshold label randomization test, conditional on pooled hits."""
    from scipy.stats import hypergeom
    if not 0 <= tp <= m or not 0 <= fp <= m:
        raise ValueError("Invalid hit counts")
    total_hits = tp + fp
    gap = abs(tp - fp)
    low = math.floor((total_hits - gap) / 2)
    high = math.ceil((total_hits + gap) / 2)
    return min(1.0, float(hypergeom.cdf(low, 2*m, total_hits, m) +
                          hypergeom.sf(high - 1, 2*m, total_hits, m)))


def summarize(cfg: dict, family: str, stage: str, profile: str | None) -> dict:
    from src.shard_audit.auditing.distinguishers import run_distinguisher
    _, audit_dir = checkpoint(cfg, family, stage, profile)
    scores_dir = audit_dir / "scores"
    train = read_jsonl(scores_dir / "train_scores.jsonl")
    test = read_jsonl(scores_dir / "test_scores.jsonl")
    for group in (train, test):
        if len(group) != 2 * (cfg["source"]["calibration_per_arm"] if group is train
                              else cfg["source"]["test_per_arm"]):
            raise ValueError("Score count differs from prepared audit size")
        if {r["label"] for r in group} != {0, 1}:
            raise ValueError("Both audit labels are required")
    score = "min_k_20_logprob"
    result = run_distinguisher([r["label"] for r in train], [r[score] for r in train],
                               [r["label"] for r in test], [r[score] for r in test],
                               score_name=score, criterion="balanced_accuracy", n_thresholds=2000)
    held = result["test"]
    m = held["n_pos"]
    if m != held["n_neg"]:
        raise ValueError("Arms must have equal held-out size")
    p = two_sided_randomization_p(held["tp"], held["fp"], m)
    output = {"family": family, "stage": stage, "profile": profile,
              "score": score, "calibrated_threshold": result["calibrated_threshold"],
              "calibration": result["train"], "held_out": held,
              "randomization_p_two_sided": p, "alpha": 0.05,
              "reject_at_0_05": p < 0.05,
              "gamma_reference": math.sqrt(math.log(40) / m),
              "gamma_reference_note": "Paper Hoeffding cutoff shown for comparison; exact label-randomization p is the primary decision for disjoint sampled arms.",
              "config_fingerprint": fingerprint(cfg)}
    write_json(audit_dir / "result.json", output)
    return output


def score(cfg: dict, family: str, stage: str, profile: str | None, batch_size: int) -> None:
    model, audit_dir = checkpoint(cfg, family, stage, profile)
    if stage != "base":
        complete = Path(model).parent / "complete.json"
        if not complete.exists():
            raise FileNotFoundError(f"Model training has not completed: {complete}")
        if json.loads(complete.read_text())["config_fingerprint"] != fingerprint(cfg):
            raise ValueError("Model/config mismatch")
    scores_dir = audit_dir / "scores"
    done = scores_dir / "table6_score.json"
    if done.exists():
        prior = json.loads(done.read_text())
        if prior != {"model": model, "config_fingerprint": fingerprint(cfg)}:
            raise ValueError("Score/config mismatch")
        return
    if scores_dir.exists() and any(scores_dir.iterdir()):
        raise FileExistsError(f"Partial scoring output requires inspection: {scores_dir}")
    prepared = artifacts(cfg) / "prepared"
    command = [sys.executable, str(ROOT / "scripts/scoring/score_causal_lm_logprobs.py"),
               "--model", model,
               "--train-file", str(prepared / "calibration.jsonl"),
               "--test-file", str(prepared / "test.jsonl"),
               "--output-dir", str(scores_dir), "--min-k-pcts", "20",
               "--batch-size", str(batch_size), "--dtype", cfg["training"]["dtype"],
               "--max-length", str(cfg["training"]["max_length"]), "--debug-examples", "0"]
    subprocess.run(command, cwd=ROOT, check=True)
    write_json(done, {"model": model, "config_fingerprint": fingerprint(cfg)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--family", required=True)
    parser.add_argument("--stage", choices=("base", "parent", "target", "control"), required=True)
    parser.add_argument("--profile")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    cfg = config(args.config)
    require_manifest(cfg)
    if args.family not in cfg["families"]:
        raise ValueError(args.family)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    score(cfg, args.family, args.stage, args.profile, args.batch_size)
    result = summarize(cfg, args.family, args.stage, args.profile)
    print(f"{args.family} {args.stage} {args.profile or ''}: advantage={result['held_out']['shard_advantage']:.4f}, p={result['randomization_p_two_sided']:.4g}")


if __name__ == "__main__":
    main()

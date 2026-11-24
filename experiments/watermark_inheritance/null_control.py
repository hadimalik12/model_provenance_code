"""Fresh-control null audit: compare S' with an independent S'' draw.

The detector threshold is frozen by the existing S-versus-S' construction
split.  S'' is never used to choose a score, threshold, checkpoint, or sample
size.  A valid shard-level null should have low held-out advantage between S'
and S'' under that already-frozen detector.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

from .common import Maryland, fingerprint, load_config, load_json, read_jsonl, write_json, write_jsonl
from .generate import (
    configured_shard_seed,
    generation_config_id,
    read_journal,
    sample_batch,
    sampling_seed,
    watermark_config_id,
)
from .prepare import file_sha256, validate_tokenizer
from .score import evaluate_shard, example_watermark_score, file_digest, infer_batch


SCHEMA = 1
SIDE = "s_double_prime"


def double_prime_seed(config: dict, pair_id: str, phase: str = "full") -> int:
    """A deterministic draw disjoint from the original integer side labels."""
    return sampling_seed(configured_shard_seed(config), phase, pair_id, SIDE)


def generation_identity(run_dir: Path, config: dict, pairs_path: Path) -> dict:
    return {
        "schema": SCHEMA,
        "role": "fresh_exchangeable_control_shard",
        "side": SIDE,
        "seed_rule": "sha256(shard_seed, phase, pair_id, s_double_prime)",
        "pairs_sha256": file_sha256(pairs_path),
        "generation_config_id": generation_config_id(config),
        "watermark_config_id": watermark_config_id(config),
        "model_path": str(Path(config["model_path"]).resolve()),
        "run_dir": str(run_dir.resolve()),
    }


def validate_double_prime(row: dict, pair: dict, config: dict) -> None:
    for field in ("pair_id", "source_id", "audit_split"):
        if row.get(field) != pair.get(field):
            raise ValueError(f"S'' record mismatched {field}: {pair['pair_id']}")
    ids = row.get("completion_ids")
    if not isinstance(ids, list) or not 1 <= len(ids) <= int(config.get("max_new_tokens", 256)):
        raise ValueError("Invalid S'' completion length")
    if row.get("completion_length") != len(ids):
        raise ValueError("S'' stored length disagrees with token IDs")
    expected = double_prime_seed(config, pair["pair_id"])
    if row.get("sampling_seed") != expected:
        raise ValueError("S'' sampling seed changed")
    if expected in {pair.get("sampling_seed_s"), pair.get("sampling_seed_s_prime")}:
        raise ValueError("S'' reused an S or S' sampling seed")
    if row.get("watermark_config_id") != watermark_config_id(config):
        raise ValueError("S'' watermark configuration changed")


def generate(run_dir: Path, device: str = "cuda:0") -> Path:
    """Generate one fresh, same-distribution completion for every fixed prompt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = run_dir.resolve()
    config = load_config(run_dir)
    pairs_path = run_dir / "prepared/pairs.jsonl"
    pairs = read_jsonl(pairs_path)
    if len(pairs) != int(config["pairs"]):
        raise ValueError("S'' generation requires the complete immutable pair corpus")
    identity = generation_identity(run_dir, config, pairs_path)
    output = run_dir / "prepared/s_double_prime.jsonl"
    metadata = run_dir / "prepared/s_double_prime.meta.json"
    if output.exists():
        if load_json(metadata) != identity:
            raise ValueError("Existing S'' metadata does not match this run")
        rows = read_jsonl(output)
        if len(rows) != len(pairs):
            raise ValueError("Existing S'' corpus is incomplete")
        for row, pair in zip(rows, pairs):
            validate_double_prime(row, pair, config)
        print(json.dumps({"event": "s_double_prime_exists", "pairs": len(rows)}), flush=True)
        return output

    journal = run_dir / "generated/s_double_prime.inprogress.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal_metadata = journal.with_suffix(".meta.json")
    if journal_metadata.exists() and load_json(journal_metadata) != identity:
        raise ValueError("Interrupted S'' generation has a different identity")
    write_json(journal_metadata, identity)
    completed = read_journal(journal)
    if len(completed) > len(pairs):
        raise ValueError("S'' journal contains too many records")
    for row, pair in zip(completed, pairs):
        validate_double_prime(row, pair, config)

    torch.set_num_threads(int(config.get("cpu_threads", 1)))
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
    validate_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], local_files_only=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.requires_grad_(False)
    watermarker = Maryland(tokenizer, config)
    batch_size = max(1, int(config.get("generation_batch_size", 32)))
    started = time.monotonic()
    new_tokens = 0
    with journal.open("a") as stream:
        for offset in range(len(completed), len(pairs), batch_size):
            batch = pairs[offset:offset + batch_size]
            seeds = [double_prime_seed(config, row["pair_id"]) for row in batch]
            outputs = sample_batch(
                model, tokenizer, watermarker,
                [row["prompt_ids"] for row in batch], seeds, config, device,
            )
            for pair, seed, ids in zip(batch, seeds, outputs):
                row = {
                    "pair_id": pair["pair_id"],
                    "source_id": pair["source_id"],
                    "audit_split": pair["audit_split"],
                    "completion_ids": ids,
                    "completion_text": tokenizer.decode(ids, skip_special_tokens=True),
                    "completion_length": len(ids),
                    "sampling_seed": seed,
                    "stop_reason": "eos" if ids and ids[-1] == tokenizer.eos_token_id else "length",
                    "watermark_config_id": watermark_config_id(config),
                }
                validate_double_prime(row, pair, config)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                completed.append(row)
            stream.flush()
            os.fsync(stream.fileno())
            new_tokens += sum(map(len, outputs))
            elapsed = time.monotonic() - started
            print(json.dumps({
                "event": "s_double_prime_progress", "pairs": len(completed),
                "total": len(pairs), "tokens_per_second": new_tokens / max(elapsed, 1e-9),
            }), flush=True)
    write_jsonl(output, completed)
    write_json(metadata, identity)
    return output


def checkpoint_identity(checkpoint_path: Path) -> list[list[int | str]]:
    files = []
    for pattern in ("*.safetensors", "pytorch_model*.bin", "config.json", "stage_state.json"):
        for path in sorted(checkpoint_path.glob(pattern)):
            stat = path.stat()
            files.append([path.name, stat.st_size, stat.st_mtime_ns])
    if not files:
        raise FileNotFoundError(f"No checkpoint files at {checkpoint_path}")
    return files


def null_evaluation(original_rows: list[dict], double_rows: list[dict], threshold: float,
                    alpha: float = 0.05) -> dict:
    """Evaluate the held-out S'-versus-S'' null with a frozen threshold."""
    if len(original_rows) != len(double_rows):
        raise ValueError("S' and S'' score counts differ")
    audit = [
        (left, right) for left, right in zip(original_rows, double_rows)
        if left["audit_split"] == "audit"
    ]
    if not audit or any(left["pair_id"] != right["pair_id"] for left, right in audit):
        raise ValueError("S' and S'' held-out pairs are not aligned")
    result = evaluate_shard(
        [left["s_prime"]["z"] for left, _ in audit],
        [right["z"] for _, right in audit],
        threshold,
    )
    # Match the paired Hoeffding decision rule prespecified in the manuscript:
    # P(|mean(D)| > t) <= 2 exp(-m t^2 / 2) for D in [-1, 1].
    cutoff = math.sqrt(2 * math.log(2 / alpha) / len(audit))
    return {
        **result,
        "alpha": alpha,
        "single_audit_advantage_cutoff": cutoff,
        "advantage_below_cutoff": result["advantage"] <= cutoff,
        "two_sided_null_rejected": result["p_shard"] < alpha,
        "interpretation": "S' and S'' are exchangeable controls; low advantage is required.",
    }


def score(run_dir: Path, checkpoint_name: str, checkpoint_path: Path,
          device: str = "cuda:0", alpha: float = 0.05,
          original_score_name: str | None = None) -> Path:
    """Score only S'' and compare it with cached S' using the frozen detector."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir, checkpoint_path = run_dir.resolve(), checkpoint_path.resolve()
    config = load_config(run_dir)
    pairs = read_jsonl(run_dir / "prepared/pairs.jsonl")
    double_path = run_dir / "prepared/s_double_prime.jsonl"
    double = read_jsonl(double_path)
    for row, pair in zip(double, pairs):
        validate_double_prime(row, pair, config)
    original_score_name = original_score_name or checkpoint_name
    original_dir = run_dir / "scores" / original_score_name
    original_rows = read_jsonl(original_dir / "examples.jsonl")
    thresholds = load_json(original_dir / "thresholds.json")
    threshold = float(thresholds["detectors"]["watermark"]["threshold"])
    identity = {
        "schema": SCHEMA,
        "comparison": "s_prime_vs_s_double_prime",
        "checkpoint": checkpoint_name,
        "original_score_name": original_score_name,
        "checkpoint_files": checkpoint_identity(checkpoint_path),
        "s_double_prime_sha256": file_sha256(double_path),
        "original_examples_sha256": file_digest(original_dir / "examples.jsonl"),
        "frozen_thresholds_sha256": file_digest(original_dir / "thresholds.json"),
        "alpha": alpha,
    }
    output_dir = run_dir / "null_scores" / checkpoint_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        if metrics["identity"] != fingerprint(identity):
            raise ValueError("Completed S'/S'' audit identity changed")
        print(json.dumps({"event": "null_audit_exists", "metrics": str(metrics_path)}), flush=True)
        return metrics_path

    predictions_path = output_dir / "predictions.jsonl"
    partial = output_dir / "predictions.inprogress.jsonl"
    completed = read_journal(partial)
    for prediction, pair, null_row in zip(completed, pairs, double):
        if prediction["pair_id"] != pair["pair_id"] or len(prediction["predictions"]) != len(null_row["completion_ids"]):
            raise ValueError("Interrupted S'' predictions are misaligned")

    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, local_files_only=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    ).to(device).eval()
    model.requires_grad_(False)
    batch_size = max(1, int(config.get("score_batch_size", 8)))
    with partial.open("a") as stream:
        for offset in range(len(completed), len(pairs), batch_size):
            batch_pairs = pairs[offset:offset + batch_size]
            batch_double = double[offset:offset + batch_size]
            outputs = infer_batch(
                model,
                [(pair["prompt_ids"], row["completion_ids"])
                 for pair, row in zip(batch_pairs, batch_double)],
                device, len(tokenizer),
                tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )
            for pair, result in zip(batch_pairs, outputs):
                record = {"pair_id": pair["pair_id"], **result}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                completed.append(record)
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps({"event": "s_double_prime_scoring", "pairs": len(completed), "total": len(pairs)}), flush=True)
    write_jsonl(predictions_path, completed)

    watermarker = Maryland(tokenizer, config)
    scored = []
    for pair, null_row, prediction in zip(pairs, double, completed):
        result, _ = example_watermark_score(
            pair["prompt_ids"], null_row["completion_ids"], prediction["predictions"], watermarker,
        )
        scored.append({"pair_id": pair["pair_id"], "audit_split": pair["audit_split"], **result})
    write_jsonl(output_dir / "examples.jsonl", scored)
    evaluation = null_evaluation(original_rows, scored, threshold, alpha)
    metrics = {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "checkpoint": checkpoint_name,
        "comparison": "s_prime_vs_s_double_prime",
        "frozen_threshold": threshold,
        "frozen_on": "S versus S' construction pairs only",
        "identity": fingerprint(identity),
        "watermark": evaluation,
    }
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, allow_nan=False), flush=True)
    return metrics_path


def report(root: Path, output_dir: Path) -> dict:
    """Summarize only the two retained negative-control experiments."""
    rows = []
    specifications = [
        (root / "dolly", "unit_*", "C3", 12),
        (root / "ultrachat_controls", "run_*", "UC12500", 3),
    ]
    for parent, pattern, checkpoint, _ in specifications:
        for run in sorted(parent.glob(pattern)):
            model_path = run / "scores" / checkpoint / "metrics.json"
            null_path = run / "null_scores" / checkpoint / "metrics.json"
            if model_path.exists():
                metric = load_json(model_path)["shard"]["watermark"]
                rows.append({
                    "run_id": load_config(run)["run_id"], "checkpoint": checkpoint,
                    "control": "independently_trained_model", "comparison": "S_vs_S_prime",
                    "signed_gap": metric["signed_gap"], "advantage": metric["advantage"],
                    "p_value": metric["p_shard"],
                    "false_alarm": metric["p_shard"] < 0.05 and metric["signed_gap"] > 0,
                })
            if null_path.exists():
                metric = load_json(null_path)["watermark"]
                rows.append({
                    "run_id": load_config(run)["run_id"], "checkpoint": checkpoint,
                    "control": "fresh_control_shard", "comparison": "S_prime_vs_S_double_prime",
                    "signed_gap": metric["signed_gap"], "advantage": metric["advantage"],
                    "p_value": metric["p_shard"],
                    "false_alarm": metric["two_sided_null_rejected"],
                })
    if not rows:
        raise ValueError("No completed negative-control results found")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "negative_controls.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "observed_completed_runs_only",
        "model_controls": sum(row["control"] == "independently_trained_model" for row in rows),
        "model_control_false_alarms": sum(row["control"] == "independently_trained_model" and row["false_alarm"] for row in rows),
        "fresh_shard_controls": sum(row["control"] == "fresh_control_shard" for row in rows),
        "fresh_shard_false_alarms": sum(row["control"] == "fresh_control_shard" and row["false_alarm"] for row in rows),
        "expected": {"dolly": 12, "ultrachat": 3},
        "interpretation": "Model controls test false attribution across training randomness; S'/S'' controls test shard exchangeability under a frozen detector.",
    }
    write_json(output_dir / "negative_controls.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--run-dir", required=True, type=Path)
    generate_parser.add_argument("--device", default="cuda:0")
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--run-dir", required=True, type=Path)
    score_parser.add_argument("--checkpoint-name", required=True)
    score_parser.add_argument("--checkpoint-path", required=True, type=Path)
    score_parser.add_argument("--original-score-name")
    score_parser.add_argument("--device", default="cuda:0")
    score_parser.add_argument("--alpha", type=float, default=0.05)
    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--root", type=Path, default=Path("artifacts/watermark_inheritance/aistats27_reviewer"))
    report_parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "generate":
        generate(args.run_dir, args.device)
    elif args.command == "score":
        score(
            args.run_dir, args.checkpoint_name, args.checkpoint_path,
            args.device, args.alpha, args.original_score_name,
        )
    else:
        report(args.root, args.output_dir or args.root / "reports")


if __name__ == "__main__":
    main()

"""LLaMA-2-13B WikiMIA length-32 membership experiment.

Orchestrates: data validation → HF-token check → scoring → threshold experiment.

Memory note:
  LLaMA-2-13B at float16 requires ~26 GB — more than a single V100-16GB.
  This script uses device_map="auto" (accelerate) to split model layers
  across GPU and CPU RAM.  With 361 GB available RAM this completes
  faithfully but ~8–15× slower than GPU-only inference.

Gated model note:
  meta-llama/Llama-2-13b-hf requires accepting Meta's license at
  https://huggingface.co/meta-llama/Llama-2-13b-hf and an approved HF token.
  Pass --hf-token <TOKEN> or set HF_TOKEN in the environment.
  If access is not granted the script stops with a clear error message.

Usage:
    python scripts/experiments/run_wikimia_llama2_13b_experiment.py \\
        --data-dir        data/processed/wikimia_length32 \\
        --score-dir       data/scores/wikimia_llama2_13b_length32 \\
        --run-dir         outputs/runs/wikimia_llama2_13b_length32 \\
        --model           meta-llama/Llama-2-13b-hf \\
        --dtype           float16 \\
        --batch-size      1 \\
        --primary-score   mean_logprob
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.shard_audit.mia_scores import compute_all_scores
from src.shard_audit.distinguishers import run_distinguisher
from src.shard_audit.metrics import score_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SCORE_KEYS = ["mean_logprob", "min_k_5_logprob", "min_k_10_logprob",
              "min_k_20_logprob", "min_k_40_logprob"]

MODEL_ID = "meta-llama/Llama-2-13b-hf"


# ------------------------------------------------------------------ #
# I/O helpers
# ------------------------------------------------------------------ #

def _load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(records: list, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------------------------------------------ #
# Step 0: Verify HF token access
# ------------------------------------------------------------------ #

def check_hf_access(model_name: str, token: str) -> None:
    """Confirm token can download model files. Raises SystemExit on failure."""
    from transformers import AutoConfig
    logger.info("Verifying HF token access to %s ...", model_name)
    try:
        AutoConfig.from_pretrained(model_name, token=token)
        logger.info("HF token access: OK")
    except Exception as e:
        msg = str(e)
        if "403" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
            logger.error(
                "\n"
                "=== BLOCKER: Model access denied ===\n"
                "Model  : %s\n"
                "Reason : HF token is not authorized for this gated model.\n"
                "Action : Accept Meta's license at\n"
                "         https://huggingface.co/meta-llama/Llama-2-13b-hf\n"
                "         and wait for approval before re-running.\n"
                "=====================================",
                model_name,
            )
        else:
            logger.error("HF config download failed: %s", msg)
        sys.exit(1)


# ------------------------------------------------------------------ #
# Step 1: Validate data
# ------------------------------------------------------------------ #

def validate_data(data_dir: str) -> dict:
    train_path    = os.path.join(data_dir, "train.jsonl")
    test_path     = os.path.join(data_dir, "test.jsonl")
    manifest_path = os.path.join(data_dir, "manifest.json")

    for p, label in [(train_path, "train.jsonl"), (test_path, "test.jsonl"),
                     (manifest_path, "manifest.json")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing {label} in {data_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    train = _load_jsonl(train_path)
    test  = _load_jsonl(test_path)

    n1_tr = sum(1 for r in train if r["label"] == 1)
    n0_tr = sum(1 for r in train if r["label"] == 0)
    n1_te = sum(1 for r in test  if r["label"] == 1)
    n0_te = sum(1 for r in test  if r["label"] == 0)

    assert n1_tr == n0_tr, f"Train imbalance: {n1_tr} member vs {n0_tr} nonmember"
    assert n1_te == n0_te, f"Test imbalance: {n1_te} member vs {n0_te} nonmember"

    h_tr = {r["text_hash"] for r in train}
    h_te = {r["text_hash"] for r in test}
    overlap = len(h_tr & h_te)
    assert overlap == 0, f"Train/test overlap: {overlap} texts"

    logger.info("Data validated: train=%d (%d+%d), test=%d (%d+%d), overlap=%d",
                len(train), n1_tr, n0_tr, len(test), n1_te, n0_te, overlap)
    return manifest


# ------------------------------------------------------------------ #
# Step 2: Load model with device_map="auto"
# ------------------------------------------------------------------ #

def _load_llama2(model_name: str, dtype_str: str, token: str):
    """Load LLaMA-2-13B with device_map='auto' for CPU-offload fallback."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.float16)

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token set to eos_token (%s)", tokenizer.eos_token)

    logger.info("Loading model with device_map='auto', dtype=%s", dtype_str)
    logger.info("(LLaMA-2-13B needs ~26 GB — may offload layers to CPU RAM)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        token=token,
    )
    model.eval()

    # Report actual device distribution
    try:
        from accelerate import dispatch_model
        devices = set(str(p.device) for p in model.parameters())
        logger.info("Model parameter devices: %s", sorted(devices))
    except Exception:
        pass

    return model, tokenizer


def _extract_logprobs_auto(texts: list, model, tokenizer, max_length: int) -> list:
    """Extract per-token log-probs when model uses device_map='auto'."""
    import torch
    import torch.nn.functional as F

    # Determine the device of the model's input embedding layer
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = torch.device("cpu")

    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encodings["input_ids"].to(input_device)
    attention_mask = encodings["attention_mask"].to(input_device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float().cpu()

    input_ids_cpu = input_ids.cpu()
    attention_mask_cpu = attention_mask.cpu()

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids_cpu[:, 1:]
    shift_mask   = attention_mask_cpu[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    results = []
    for i in range(len(texts)):
        mask_i = shift_mask[i].bool()
        results.append(token_logprobs[i][mask_i].tolist())

    return results


# ------------------------------------------------------------------ #
# Step 3: Score
# ------------------------------------------------------------------ #

def score_model(
    data_dir: str,
    score_dir: str,
    model_name: str,
    dtype_str: str,
    token: str,
    k_pcts: list,
    batch_size: int,
    max_length: int,
) -> bool:
    train_out = os.path.join(score_dir, "train_scores.jsonl")
    test_out  = os.path.join(score_dir, "test_scores.jsonl")

    if os.path.isfile(train_out) and os.path.isfile(test_out):
        logger.info("Scores already exist in %s — skipping scoring.", score_dir)
        return False

    model, tokenizer = _load_llama2(model_name, dtype_str, token)

    def _score_split(records, split_name):
        results = []
        n = len(records)
        for i in range(0, n, batch_size):
            batch = records[i:i + batch_size]
            texts = [r["text"] for r in batch]
            batch_lp = _extract_logprobs_auto(texts, model, tokenizer, max_length)
            for rec, lp in zip(batch, batch_lp):
                sc = compute_all_scores(rec["text"], lp, k_pcts=tuple(k_pcts))
                results.append({
                    "id":                rec["id"],
                    "label":             rec["label"],
                    "phase_split":       rec.get("phase_split", split_name),
                    "text_hash":         rec["text_hash"],
                    "model":             model_name,
                    "num_input_tokens":  len(lp) + 1,
                    "num_scored_tokens": len(lp),
                    "mean_logprob":      round(sc["mean_logprob"], 6),
                    "mean_loss":         round(sc["mean_loss"], 6),
                    **{f"min_k_{k}_logprob": round(sc[f"min_k_{k}_logprob"], 6)
                       for k in k_pcts},
                    "zlib_norm_logprob": sc.get("zlib_norm_logprob"),
                })
            if (i // batch_size + 1) % 20 == 0:
                logger.info("  Scored %d / %d %s examples...",
                            min(i + batch_size, n), n, split_name)
        return results

    train_records = _load_jsonl(os.path.join(data_dir, "train.jsonl"))
    test_records  = _load_jsonl(os.path.join(data_dir, "test.jsonl"))

    logger.info("Scoring %d train examples (this may take several hours with CPU offload)...",
                len(train_records))
    train_scores = _score_split(train_records, "train")

    logger.info("Scoring %d test examples ...", len(test_records))
    test_scores = _score_split(test_records, "test")

    diag_train = score_diagnostics(train_scores, score_keys=tuple(SCORE_KEYS))
    diag_test  = score_diagnostics(test_scores,  score_keys=tuple(SCORE_KEYS))
    logger.info("=== Score Diagnostics (Train) ===")
    for k, d in diag_train.items():
        logger.info("  %-28s AUC=%.4f dir=%s", k, d.get("auc", float("nan")),
                    "OK" if d.get("direction_ok") else "INVERTED")
    logger.info("=== Score Diagnostics (Test) ===")
    for k, d in diag_test.items():
        logger.info("  %-28s AUC=%.4f dir=%s", k, d.get("auc", float("nan")),
                    "OK" if d.get("direction_ok") else "INVERTED")

    import torch
    # Report device distribution for provenance
    devices = sorted(set(str(p.device) for p in model.parameters()))

    manifest = {
        "model":             model_name,
        "device_map":        "auto",
        "parameter_devices": devices,
        "dtype":             dtype_str,
        "batch_size":        batch_size,
        "k_pcts":            k_pcts,
        "max_length":        max_length,
        "n_train_scored":    len(train_scores),
        "n_test_scored":     len(test_scores),
        "score_diagnostics_train": diag_train,
        "score_diagnostics_test":  diag_test,
        "timestamp":         datetime.utcnow().isoformat() + "Z",
    }
    _write_jsonl(train_scores, train_out)
    _write_jsonl(test_scores, test_out)
    _write_json(manifest, os.path.join(score_dir, "manifest.json"))
    logger.info("Scores written to %s", score_dir)

    del model
    torch.cuda.empty_cache()
    return True


# ------------------------------------------------------------------ #
# Step 4: Threshold experiment + shuffled-label control
# ------------------------------------------------------------------ #

def run_experiment(
    score_dir: str,
    run_dir: str,
    primary_score: str,
    criterion: str = "balanced_accuracy",
    shuffled_seed: int = 7,
) -> dict:
    results_path = os.path.join(run_dir, "results.json")
    if os.path.isfile(results_path):
        logger.info("Experiment results already exist — skipping.")
        with open(results_path) as f:
            return json.load(f)

    train_records = _load_jsonl(os.path.join(score_dir, "train_scores.jsonl"))
    test_records  = _load_jsonl(os.path.join(score_dir, "test_scores.jsonl"))

    train_labels = [r["label"] for r in train_records]
    test_labels  = [r["label"] for r in test_records]

    available_keys = [k for k in SCORE_KEYS if k in train_records[0]]

    main_results = []
    for key in available_keys:
        train_sc = [r[key] for r in train_records]
        test_sc  = [r[key] for r in test_records]
        res = run_distinguisher(
            train_labels, train_sc, test_labels, test_sc,
            score_name=key, criterion=criterion,
        )
        main_results.append(res)

    logger.info("\n--- Shuffled-label control ---")
    rng = random.Random(shuffled_seed)
    shuffled_train_labels = list(train_labels)
    rng.shuffle(shuffled_train_labels)
    shuffled_results = []
    for key in available_keys:
        train_sc = [r[key] for r in train_records]
        test_sc  = [r[key] for r in test_records]
        res = run_distinguisher(
            shuffled_train_labels, train_sc, test_labels, test_sc,
            score_name=key, criterion=criterion,
        )
        shuffled_results.append(res)

    output = {
        "main_results":           main_results,
        "shuffled_label_control": shuffled_results,
        "primary_score":          primary_score,
        "timestamp":              datetime.utcnow().isoformat() + "Z",
    }

    os.makedirs(run_dir, exist_ok=True)
    _write_json(output, results_path)
    logger.info("Experiment results written to %s", results_path)

    logger.info("\n=== Test-Split Results ===")
    logger.info("%-28s %6s %6s %6s %6s", "Score", "Acc", "AUC", "Adv", "T@1FP")
    for r in main_results:
        t = r["test"]
        logger.info("%-28s %6.3f %6.3f %6.3f %6s",
                    r["score_name"],
                    t.get("accuracy", float("nan")),
                    t.get("auc") or float("nan"),
                    t.get("shard_advantage", float("nan")),
                    f"{t.get('tpr_at_1_fpr') or float('nan'):.3f}")

    return output


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def parse_args():
    p = argparse.ArgumentParser(
        description="LLaMA-2-13B WikiMIA shard-membership experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",       default="data/processed/wikimia_length32")
    p.add_argument("--score-dir",      default="data/scores/wikimia_llama2_13b_length32")
    p.add_argument("--run-dir",        default="outputs/runs/wikimia_llama2_13b_length32")
    p.add_argument("--model",          default=MODEL_ID)
    p.add_argument("--dtype",          default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--batch-size",     type=int, default=1)
    p.add_argument("--max-length",     type=int, default=128)
    p.add_argument("--min-k-pcts",     default="5,10,20,40")
    p.add_argument("--primary-score",  default="mean_logprob")
    p.add_argument("--hf-token",       default=None,
                   help="HuggingFace token for gated model access. "
                        "Defaults to HF_TOKEN env var.")
    return p.parse_args()


def main():
    args = parse_args()
    k_pcts = [int(k) for k in args.min_k_pcts.split(",")]
    token  = args.hf_token or os.environ.get("HF_TOKEN", "")

    logger.info("=== LLaMA-2-13B WikiMIA Experiment ===")
    logger.info("Data dir:   %s", args.data_dir)
    logger.info("Score dir:  %s", args.score_dir)
    logger.info("Run dir:    %s", args.run_dir)
    logger.info("Model:      %s", args.model)
    logger.info("dtype:      %s", args.dtype)

    logger.info("\n[0/4] Checking HF token access...")
    if not token:
        logger.error(
            "No HF token found. Set HF_TOKEN in the environment or pass --hf-token."
        )
        sys.exit(1)
    check_hf_access(args.model, token)

    logger.info("\n[1/4] Validating data...")
    manifest = validate_data(args.data_dir)
    logger.info("WikiMIA split: %s  train=%d  test=%d  seed=%d",
                manifest.get("wikimia_split"),
                manifest.get("n_train"), manifest.get("n_test"),
                manifest.get("seed"))

    logger.info("\n[2/4] Scoring %s (device_map='auto')...", args.model)
    score_model(
        data_dir=args.data_dir,
        score_dir=args.score_dir,
        model_name=args.model,
        dtype_str=args.dtype,
        token=token,
        k_pcts=k_pcts,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    logger.info("\n[3/4] Running threshold experiment...")
    run_experiment(
        score_dir=args.score_dir,
        run_dir=args.run_dir,
        primary_score=args.primary_score,
    )

    print(f"\n=== ALL DONE ===")
    print(f"Scores:  {args.score_dir}")
    print(f"Results: {args.run_dir}")
    print(f"\nNext step:")
    print(f"  python scripts/reports/report_wikimia_llama2_13b.py \\")
    print(f"    --results-file   {args.run_dir}/results.json \\")
    print(f"    --test-scores    {args.score_dir}/test_scores.jsonl \\")
    print(f"    --data-manifest  {args.data_dir}/manifest.json \\")
    print(f"    --score-manifest {args.score_dir}/manifest.json \\")
    print(f"    --opt-results    outputs/runs/wikimia_opt67b/results.json \\")
    print(f"    --pythia-results outputs/runs/wikimia_pythia69b_length32/results.json \\")
    print(f"    --output-dir     outputs/reports/wikimia_llama2_13b_length32")


if __name__ == "__main__":
    main()

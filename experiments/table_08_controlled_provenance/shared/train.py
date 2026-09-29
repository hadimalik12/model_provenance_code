"""Single-model SFT stage; one GPU job for each parent, target, or control."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .common import DEFAULT_CONFIG, artifacts, config, fingerprint, read_jsonl, require_manifest, write_json


def tokenize_row(row: dict, tokenizer, max_length: int) -> dict | None:
    prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
    if not answer_ids:
        return None
    # Keep the end of a long prompt and the beginning of its response.
    answer_ids = answer_ids[:max_length - 1]
    prompt_ids = prompt_ids[-(max_length - len(answer_ids)):]
    ids = prompt_ids + answer_ids
    if not prompt_ids or len(ids) < 2:
        return None
    labels = [-100] * len(prompt_ids) + answer_ids
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class PaddedRows:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, rows: list[dict]) -> dict:
        import torch
        length = max(len(r["input_ids"]) for r in rows)
        return {"input_ids": torch.tensor([r["input_ids"] + [self.pad_id] * (length - len(r["input_ids"])) for r in rows]),
                "attention_mask": torch.tensor([r["attention_mask"] + [0] * (length - len(r["attention_mask"])) for r in rows]),
                "labels": torch.tensor([r["labels"] + [-100] * (length - len(r["labels"])) for r in rows])}


def stage_paths(cfg: dict, family: str, stage: str, profile: str | None):
    base = artifacts(cfg)
    family_root = base / "models" / family
    if stage == "parent":
        return cfg["families"][family], base / "prepared" / "s_train.jsonl", family_root / "parent"
    if profile not in cfg["profiles"][family]:
        raise ValueError(f"Profile {profile!r} is not configured for {family}")
    data = base / "downstream" / profile / "train.jsonl"
    if stage == "target":
        return str(family_root / "parent" / "final"), data, family_root / profile / "target"
    if stage == "control":
        return cfg["families"][family], data, family_root / profile / "control"
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--family", required=True)
    parser.add_argument("--stage", choices=("parent", "target", "control"), required=True)
    parser.add_argument("--profile")
    parser.add_argument("--allow-cpu-smoke", action="store_true",
                        help="Only for tiny local end-to-end validation")
    args = parser.parse_args()
    cfg = config(args.config)
    require_manifest(cfg)
    if args.family not in cfg["families"]:
        raise ValueError(args.family)
    source, dataset_path, output = stage_paths(cfg, args.family, args.stage, args.profile)
    if (output / "complete.json").exists():
        prior = json.loads((output / "complete.json").read_text())
        if prior["config_fingerprint"] != fingerprint(cfg):
            raise ValueError(f"Config mismatch for existing model: {output}")
        print(f"Already complete: {output}")
        return
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Partial output requires inspection before retry: {output}")
    if args.stage == "target" and not (Path(source).parent / "complete.json").exists():
        raise FileNotFoundError(f"Parent has not completed: {source}")
    if args.stage != "parent":
        data_manifest = json.loads((dataset_path.parent / "manifest.json").read_text())
        if data_manifest["config_fingerprint"] != fingerprint(cfg):
            raise ValueError("Downstream data/config mismatch")
    from datasets import Dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    settings = cfg["training"]
    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.allow_cpu_smoke:
        raise RuntimeError("Table 8 production training requires a CUDA GPU")
    if not has_cuda:
        torch.set_num_threads(min(4, torch.get_num_threads()))
    dtype = torch.float32 if not has_cuda else {"float16": torch.float16,
                                               "bfloat16": torch.bfloat16,
                                               "float32": torch.float32}[settings["dtype"]]
    if has_cuda and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bfloat16")
    if has_cuda and settings["optimizer"] == "paged_adamw_8bit":
        import bitsandbytes as bnb
        probe = torch.nn.Parameter(torch.zeros(16, device="cuda"))
        probe.grad = torch.ones_like(probe)
        try:
            bnb.optim.PagedAdamW8bit([probe]).step()
        except Exception as exc:
            raise RuntimeError("bitsandbytes 8-bit optimizer failed the GPU preflight") from exc
        del probe
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # AMP's gradient scaler requires FP32 trainable weights. Loading FP16
    # weights here produces FP16 gradients and fails at the first update.
    model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=torch.float32)
    model.config.use_cache = False
    rows = read_jsonl(dataset_path)
    # Full-parameter 1.4B training is tight on 16 GiB V100s.  Reducing only
    # this family's sequence length leaves the model, data, and optimization
    # recipe unchanged while keeping activations below the device limit.
    max_length = min(settings["max_length"], 256) if args.family == "1_4b" else settings["max_length"]
    encoded = [tokenize_row(r, tokenizer, max_length) for r in rows]
    encoded = [r for r in encoded if r is not None]
    if not encoded:
        raise ValueError("No trainable examples after tokenization")
    output.mkdir(parents=True)
    train_args = TrainingArguments(
        output_dir=str(output / "trainer"),
        per_device_train_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["grad_accumulation"],
        learning_rate=settings["learning_rate"],
        num_train_epochs=settings["epochs"] if args.stage == "parent" else cfg["downstream"]["epochs"],
        optim=settings["optimizer"] if has_cuda else "adamw_torch",
        fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16,
        gradient_checkpointing=True, save_strategy="no", evaluation_strategy="no",
        logging_steps=50, report_to="none", remove_unused_columns=False,
        dataloader_num_workers=0, seed=settings["seed"],
    )
    trainer = Trainer(model=model, args=train_args, train_dataset=Dataset.from_list(encoded),
                      data_collator=PaddedRows(tokenizer.pad_token_id))
    started = time.time()
    result = trainer.train()
    model.config.use_cache = True
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(output / "final")
    write_json(output / "complete.json", {"stage": args.stage, "family": args.family,
               "profile": args.profile, "source": source, "data": str(dataset_path),
               "n_examples": len(encoded), "train_loss": result.training_loss,
               "elapsed_seconds": time.time() - started,
               "peak_cuda_gb": torch.cuda.max_memory_allocated() / 2**30 if has_cuda else None,
               "config_fingerprint": fingerprint(cfg)})
    print(f"Complete: {output}")


if __name__ == "__main__":
    main()

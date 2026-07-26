"""Augmented Log-Probability Scoring: target encoder + parent head.

Combines Algorithm 2's augmentation pipeline with the classification correctness
of the parent classification head. For each image, generates K augmented views,
extracts target encoder features, projects through parent's 1000-class head,
and computes the average correct class log-probability across all views.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.shard_audit.scoring.vit_scores import (
    encoder_features,
    load_parent_head,
    make_random_init_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def _load_image(rec):
    p = rec["image_path"]
    if not os.path.isabs(p):
        p = os.path.join(REPO_ROOT, p)
    return Image.open(p).convert("RGB")


def make_augmenter(image_size=224, image_mean=None, image_std=None):
    if image_mean is None: image_mean = [0.5, 0.5, 0.5]
    if image_std is None:  image_std  = [0.5, 0.5, 0.5]
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.10), antialias=True),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std),
    ])


@torch.no_grad()
def aug_score_batch_correct(
    images, labels, target_model, parent_head, augmenter, n_aug, device, dtype,
):
    """Score a batch: augment -> target encoder -> parent head -> correct log-probability."""
    target_model.eval()
    B = len(images)
    K = n_aug

    # Build (B*K) augmented tensors
    tensors = []
    for img in images:
        for _ in range(K):
            tensors.append(augmenter(img))
    x = torch.stack(tensors, dim=0).to(device=device, dtype=dtype)

    # Extract target encoder features -> project through parent head
    feats = encoder_features(target_model, x)           # [B*K, hidden]
    logits = parent_head(feats)                          # [B*K, 1000]
    log_p = F.log_softmax(logits, dim=-1)

    # Replicate labels for the augmented views
    labels = labels.to(device)
    labels_bk_flat = labels.repeat_interleave(K)         # [B*K]
    
    # Gather correct class log-probability
    correct_log_p = log_p.gather(1, labels_bk_flat.unsqueeze(1)).squeeze(1) # [B*K]
    correct_log_p_bk = correct_log_p.view(B, K).float()  # [B, K]
    
    # Compute mean across K augmentations
    correct_logprob_mean = correct_log_p_bk.mean(dim=-1) # [B]

    out = []
    for b in range(B):
        out.append({
            "aug_correct_logprob_mean": float(correct_logprob_mean[b].item()),
        })
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Augmented Correct Log-Probability scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target-model", required=True)
    p.add_argument("--parent-model", required=True,
                    help="HF id of parent model whose classifier head is used.")
    p.add_argument("--random-init", action="store_true")
    p.add_argument("--random-seed", type=int, default=0)
    p.add_argument("--train-file", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-aug", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    
    # Skip if outputs already exist
    train_out = os.path.join(args.output_dir, "train_scores.jsonl")
    test_out = os.path.join(args.output_dir, "test_scores.jsonl")
    manifest_out = os.path.join(args.output_dir, "manifest.json")
    if os.path.exists(train_out) and os.path.exists(test_out) and os.path.exists(manifest_out):
        logger.info("Scores already exist in %s. Skipping scoring.", args.output_dir)
        return

    torch.manual_seed(args.seed)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = (torch.float16 if device == "cuda" else torch.float32) if args.dtype == "auto" else dtype_map[args.dtype]

    logger.info("=== Augmented Log-Probability Scoring ===")
    logger.info("Target:  %s (random_init=%s)", args.target_model, args.random_init)
    logger.info("Parent:  %s", args.parent_model)
    logger.info("n_aug=%d  batch=%d  device=%s  dtype=%s", args.n_aug, args.batch_size, device, dtype)

    # Load parent head (Linear 768->1000) and its processor
    parent_head, parent_proc = load_parent_head(args.parent_model, device=device, dtype=dtype)

    # Load target model
    if args.random_init:
        target_model = make_random_init_model(
            args.target_model, seed=args.random_seed, device=device, dtype=dtype)
        target_label = f"{args.target_model}::random_init_seed{args.random_seed}"
    else:
        target_model = AutoModelForImageClassification.from_pretrained(
            args.target_model, torch_dtype=dtype).to(device).eval()
        target_label = args.target_model

    # Use parent's processor for normalization (critical for head alignment)
    mean, std = parent_proc.image_mean, parent_proc.image_std
    size = (parent_proc.size.get("height") or parent_proc.size.get("shortest_edge")
            or parent_proc.crop_size.get("height", 224))
    augmenter = make_augmenter(image_size=int(size), image_mean=mean, image_std=std)

    train_records = _load_jsonl(args.train_file)
    test_records = _load_jsonl(args.test_file)
    if args.max_examples:
        train_records = train_records[:args.max_examples]
        test_records = test_records[:args.max_examples]
    logger.info("Loaded %d train, %d test records", len(train_records), len(test_records))

    score_keys = ("aug_correct_logprob_mean",)

    def _score_split(records, split_name):
        out = []
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            imgs = [_load_image(r) for r in batch]
            lbls = torch.tensor([r["imagenet_class"] for r in batch], dtype=torch.long)
            stats = aug_score_batch_correct(
                imgs, lbls, target_model, parent_head, augmenter,
                n_aug=args.n_aug, device=device, dtype=dtype)
            for rec, s in zip(batch, stats):
                out.append({
                    "id": rec["id"], "label": rec["label"],
                    "phase_split": rec.get("phase_split", split_name),
                    "image_hash": rec["image_hash"],
                    "imagenet_class": rec["imagenet_class"],
                    "model": target_label,
                    **{k: round(s[k], 6) for k in score_keys},
                })
            n_done = min(start + args.batch_size, len(records))
            if (start // args.batch_size + 1) % 10 == 0:
                logger.info("  %s: %d/%d", split_name, n_done, len(records))
        return out

    logger.info("Scoring train split...")
    train_scores = _score_split(train_records, "train")
    logger.info("Scoring test split...")
    test_scores = _score_split(test_records, "test")

    # Diagnostics
    def _diag(records):
        d = {}
        for k in score_keys:
            pos = [r[k] for r in records if r["label"] == 1]
            neg = [r[k] for r in records if r["label"] == 0]
            if not pos or not neg: continue
            all_v = sorted([(v,1) for v in pos] + [(v,0) for v in neg])
            ranks, i = {}, 0
            while i < len(all_v):
                j = i
                while j+1 < len(all_v) and all_v[j+1][0] == all_v[i][0]: j += 1
                avg = (i+j)/2+1
                for kk in range(i,j+1): ranks[kk] = avg
                i = j+1
            sumr = sum(ranks[idx] for idx,(_, lbl) in enumerate(all_v) if lbl==1)
            n_pos, n_neg = len(pos), len(neg)
            u = sumr - n_pos*(n_pos+1)/2
            auc = u/(n_pos*n_neg) if n_pos and n_neg else 0.5
            d[k] = {"mean_1": sum(pos)/n_pos, "mean_0": sum(neg)/n_neg,
                     "auc": auc, "dir_ok": auc >= 0.5}
        return d

    for name, recs in [("Train", train_scores), ("Test", test_scores)]:
        logger.info("=== %s diagnostics ===", name)
        for k, v in _diag(recs).items():
            logger.info("  %-22s  AUC=%.4f  dir=%s", k, v["auc"],
                        "OK" if v["dir_ok"] else "INVERTED")

    os.makedirs(args.output_dir, exist_ok=True)
    _write_jsonl(train_scores, os.path.join(args.output_dir, "train_scores.jsonl"))
    _write_jsonl(test_scores, os.path.join(args.output_dir, "test_scores.jsonl"))

    manifest = {
        "target_model": args.target_model, "target_label": target_label,
        "parent_model": args.parent_model, "random_init": args.random_init,
        "n_aug": args.n_aug, "batch_size": args.batch_size,
        "score_keys": list(score_keys),
        "n_train": len(train_scores), "n_test": len(test_scores),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n=== DONE ===\n  Output: {args.output_dir}")
    print(f"  train_scores: {len(train_scores)}\n  test_scores: {len(test_scores)}")


if __name__ == "__main__":
    main()

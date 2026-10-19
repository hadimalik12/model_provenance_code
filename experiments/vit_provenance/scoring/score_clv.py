"""Score ViT targets with the submitted centered-logit-variance distinguisher.

For each image, the scorer queries the target under eight deterministic weak
views. If ``p_k`` is the target probability vector for view ``k``, it centers
``log(p_k)`` within the view and reports the negative population variance over
views, averaged over output classes. Higher values indicate greater stability.
Only target output vectors are used; labels and internal activations are not.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.transforms import functional as vision_f
from transformers import AutoImageProcessor, AutoModelForImageClassification


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shard_audit.scoring.vit_scores import (
    centered_logit_variance,
    make_random_init_model,
)
from src.shard_audit.reproducibility import lock_sha256, model_revision


LOGGER = logging.getLogger(__name__)
VIEW_SPECS = (
    (0.50, 0.50, False, 1.00, 1.00, 1.00),
    (0.00, 0.00, False, 1.02, 1.00, 1.00),
    (1.00, 0.00, True, 1.00, 1.02, 1.00),
    (0.00, 1.00, True, 0.98, 1.00, 1.02),
    (1.00, 1.00, False, 1.00, 0.98, 0.98),
    (0.50, 0.00, False, 1.01, 1.01, 1.00),
    (0.50, 1.00, True, 0.99, 0.99, 1.00),
    (0.50, 0.50, True, 1.00, 1.00, 1.00),
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_image(record: dict) -> Image.Image:
    path = Path(record["image_path"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    return Image.open(path).convert("RGB")


def fixed_weak_views(
    image: Image.Image,
    *,
    image_size: int,
    image_mean: list[float],
    image_std: list[float],
) -> list[torch.Tensor]:
    """Return the protocol's eight fixed, deterministic weak views."""
    resize_size = round(image_size * 1.10)
    resized = vision_f.resize(image, [resize_size, resize_size], antialias=True)
    margin = resize_size - image_size
    views = []
    for x_fraction, y_fraction, flip, brightness, contrast, saturation in VIEW_SPECS:
        left = round(margin * x_fraction)
        top = round(margin * y_fraction)
        view = vision_f.crop(resized, top, left, image_size, image_size)
        if flip:
            view = vision_f.hflip(view)
        if brightness != 1.0:
            view = vision_f.adjust_brightness(view, brightness)
        if contrast != 1.0:
            view = vision_f.adjust_contrast(view, contrast)
        if saturation != 1.0:
            view = vision_f.adjust_saturation(view, saturation)
        tensor = vision_f.to_tensor(view)
        views.append(vision_f.normalize(tensor, image_mean, image_std))
    return views


@torch.no_grad()
def clv_score_batch(
    images: list[Image.Image],
    target_model,
    *,
    image_size: int,
    image_mean: list[float],
    image_std: list[float],
    device: str,
    dtype: torch.dtype,
) -> list[float]:
    """Compute the exact centered-logit-variance statistic for a batch."""
    target_model.eval()
    tensors = [
        tensor
        for image in images
        for tensor in fixed_weak_views(
            image,
            image_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
        )
    ]
    batch = torch.stack(tensors).to(device=device, dtype=dtype)
    logits = target_model(pixel_values=batch).logits.float()
    scores = centered_logit_variance(
        logits, batch_size=len(images), n_views=len(VIEW_SPECS)
    )
    return [float(value) for value in scores.cpu()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a ViT with eight-view centered-logit variance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument(
        "--target-revision",
        help="Immutable Hugging Face model commit; defaults to the input lock",
    )
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--max-examples", type=int)
    return parser.parse_args()


def select_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(value: str, device: str) -> torch.dtype:
    if value == "auto":
        return torch.float16 if device == "cuda" else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = select_device(args.device)
    dtype = select_dtype(args.dtype, device)
    local_target = Path(args.target_model).exists()
    if local_target and args.target_revision:
        raise ValueError("--target-revision cannot be used with a local checkpoint")
    revision = None if local_target else (
        args.target_revision or model_revision(args.target_model)
    )

    if args.random_init:
        target_model = make_random_init_model(
            args.target_model,
            seed=args.random_seed,
            device=device,
            dtype=dtype,
            revision=revision,
        )
        target_label = f"{args.target_model}::random_init_seed{args.random_seed}"
    else:
        model_kwargs = {"torch_dtype": dtype}
        if revision:
            model_kwargs["revision"] = revision
        target_model = AutoModelForImageClassification.from_pretrained(
            args.target_model, **model_kwargs
        ).to(device).eval()
        target_label = args.target_model

    processor_kwargs = {"use_fast": True}
    if revision:
        processor_kwargs["revision"] = revision
    processor = AutoImageProcessor.from_pretrained(
        args.target_model, **processor_kwargs
    )
    image_size = int(
        processor.size.get("height")
        or processor.size.get("shortest_edge")
        or processor.crop_size.get("height", 224)
    )
    image_mean = list(processor.image_mean)
    image_std = list(processor.image_std)

    train_records = load_jsonl(args.train_file)
    test_records = load_jsonl(args.test_file)
    if args.max_examples is not None:
        train_records = train_records[: args.max_examples]
        test_records = test_records[: args.max_examples]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def score_split(records: list[dict], name: str) -> None:
        destination = args.output_dir / f"{name}_scores.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for start in range(0, len(records), args.batch_size):
                rows = records[start : start + args.batch_size]
                scores = clv_score_batch(
                    [load_image(row) for row in rows],
                    target_model,
                    image_size=image_size,
                    image_mean=image_mean,
                    image_std=image_std,
                    device=device,
                    dtype=dtype,
                )
                for row, score in zip(rows, scores):
                    handle.write(
                        json.dumps(
                            {
                                "id": row["id"],
                                "label": row["label"],
                                "phase_split": row.get("phase_split", name),
                                "image_hash": row["image_hash"],
                                "model": target_label,
                                "clv_score": score,
                            }
                        )
                        + "\n"
                    )
                LOGGER.info("%s: %d/%d", name, min(start + len(rows), len(records)), len(records))

    score_split(train_records, "train")
    score_split(test_records, "test")
    manifest = {
        "target_model": args.target_model,
        "target_revision": revision,
        "target_label": target_label,
        "input_lock_sha256": lock_sha256(),
        "random_init": args.random_init,
        "score": "negative_mean_population_variance_of_view_centered_log_probabilities",
        "score_key": "clv_score",
        "n_fixed_views": len(VIEW_SPECS),
        "view_specs": VIEW_SPECS,
        "image_size": image_size,
        "n_train": len(train_records),
        "n_test": len(test_records),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()

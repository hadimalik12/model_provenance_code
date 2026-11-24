"""Small, dependency-light wrappers for frozen text and image encoders."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def frozen_text_embeddings(
    texts: Sequence[str],
    *,
    model_id: str,
    revision: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision).eval().to(resolved_device)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start:start + batch_size]), padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(resolved_device)
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            batches.append(pooled.float().cpu().numpy())
    return np.concatenate(batches, axis=0)


def frozen_image_embeddings(
    paths: Sequence[Path],
    *,
    model_id: str,
    revision: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if resolved_device == "auto":
        resolved_device = "cpu"
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision).eval().to(resolved_device)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            encoded = processor(images=images, return_tensors="pt").to(resolved_device)
            # AutoModel resolves CLIP checkpoints to the joint CLIPModel, whose
            # forward method also requires text input.  Its image feature helper
            # is the correct image-only interface.  ViT-style encoders use the
            # ordinary model forward path below.
            if hasattr(model, "get_image_features"):
                pooled = model.get_image_features(**encoded)
            else:
                output = model(**encoded)
                if getattr(output, "pooler_output", None) is not None:
                    pooled = output.pooler_output
                else:
                    pooled = output.last_hidden_state[:, 0]
            pooled = torch.nn.functional.normalize(pooled, dim=1)
            batches.append(pooled.float().cpu().numpy())
    return np.concatenate(batches, axis=0)

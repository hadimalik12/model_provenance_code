"""Small model-construction helpers shared by the ViT witness audit."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as torch_f


def centered_logit_variance(
    logits: torch.Tensor, *, batch_size: int, n_views: int = 8
) -> torch.Tensor:
    """Return negative mean population variance of view-centered log probabilities."""
    if logits.ndim != 2 or logits.shape[0] != batch_size * n_views:
        raise ValueError(
            f"logits must have shape [batch_size * {n_views}, classes]"
        )
    classes = logits.shape[-1]
    log_probabilities = torch_f.log_softmax(logits.float(), dim=-1)
    centered = log_probabilities - log_probabilities.mean(dim=-1, keepdim=True)
    centered = centered.view(batch_size, n_views, classes)
    return -centered.var(dim=1, unbiased=False).mean(dim=-1)


def make_random_init_model(
    reference_model_id: str,
    seed: int = 0,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
):
    """Build a deterministic random-initialization sanity-check model."""
    from transformers import AutoConfig, AutoModelForImageClassification

    config = AutoConfig.from_pretrained(reference_model_id)
    torch.manual_seed(seed)
    model = AutoModelForImageClassification.from_config(config)
    if dtype is not None:
        model = model.to(dtype)
    return model.to(device).eval()

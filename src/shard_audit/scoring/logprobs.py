"""Per-token log-probability extraction from a causal language model.

Causal shift definition:
  Given token ids [t_0, t_1, ..., t_n], logits at position i-1 predict t_i.
  So log p(t_i | t_{<i}) is computed by shifting logits left by one.
  Token t_0 has no prefix and is never scored; num_scored_tokens = n.

Padding:
  If the tokenizer has no pad token, eos_token is used as pad.
  Padding positions are excluded from scoring via the attention mask.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_device(device_str: str = "auto"):
    """Resolve 'auto' to cuda if available, else cpu."""
    import torch
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model_and_tokenizer(
    model_name: str,
    device,
    dtype_str: str = "auto",
):
    """Load a HuggingFace causal LM and its tokenizer.

    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as err:
        base_fallback = "EleutherAI/pythia-1.4b"
        if "1b" in model_name.lower() and "1.4" not in model_name.lower():
            base_fallback = "EleutherAI/pythia-1b"
        elif "6.9b" in model_name.lower():
            base_fallback = "EleutherAI/pythia-6.9b"
        elif "12b" in model_name.lower():
            base_fallback = "EleutherAI/pythia-12b"
        logger.warning(
            "Failed to load tokenizer for '%s' (%s). Falling back to base tokenizer '%s'.",
            model_name, err, base_fallback
        )
        tokenizer = AutoTokenizer.from_pretrained(base_fallback)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token was None; set to eos_token (%s)", tokenizer.eos_token)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_str == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = dtype_map.get(dtype_str, torch.float32)

    dev_str = str(device)
    if "cuda" in dev_str:
        device_map = "auto"
    else:
        device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    model_device = getattr(model, "device", device)
    logger.info("Loaded %s on %s with dtype=%s", model_name, model_device, dtype)
    return model, tokenizer


def extract_token_logprobs(
    texts: list,
    model,
    tokenizer,
    device,
    max_length: int = 512,
) -> list:
    """Extract per-token log-probabilities for a batch of texts.

    Args:
        texts: list of raw strings
        model: HuggingFace causal LM (model.eval() assumed)
        tokenizer: corresponding tokenizer with pad_token set
        device: torch.device
        max_length: max tokenized length

    Returns:
        list of lists of floats; result[i] is the scored token log-probs for texts[i].
        The length of result[i] is (num_input_tokens_i - 1) after mask filtering.
    """
    import torch
    import torch.nn.functional as F

    # Empty/whitespace-only records otherwise tokenize to a zero-length (and
    # sometimes floating-point) tensor.  Use a deterministic sentinel so an
    # occasional malformed record cannot crash an hours-long Slurm job.
    safe_texts = [text if isinstance(text, str) and text.strip() else "<empty>" for text in texts]
    encodings = tokenizer(
        safe_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    target_device = getattr(model, "device", device)
    # Some tokenizers represent an empty sequence with a floating tensor.
    # Embedding indices must always be integral for every supported CausalLM.
    input_ids = encodings["input_ids"].to(device=target_device, dtype=torch.long)
    attention_mask = encodings["attention_mask"].to(target_device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

    # Causal shift: logits[:, :-1] predict input_ids[:, 1:]
    shift_logits = logits[:, :-1, :]          # (B, L-1, V)
    shift_labels = input_ids[:, 1:]           # (B, L-1)
    shift_mask = attention_mask[:, 1:]        # (B, L-1)  — excludes padding in suffix

    log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, L-1, V)
    token_logprobs = log_probs.gather(
        -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)           # (B, L-1)

    results = []
    for i in range(len(safe_texts)):
        mask_i = shift_mask[i].bool()
        lp_i = token_logprobs[i][mask_i].float().cpu().tolist()
        results.append(lp_i)

    return results


def load_output_head(model_name: str, device, dtype_str: str = "auto"):
    """Load a causal LM and return only its output projection head.

    Table 5 scores target hidden states with the parent model's vocabulary head.
    This captures the paper's parent-head transfer protocol while preserving the
    target model's transformer body.
    """
    parent_model, _ = load_model_and_tokenizer(model_name, device, dtype_str)
    head = parent_model.get_output_embeddings()
    if head is None:
        raise ValueError(f"Model {model_name!r} does not expose an output head.")
    return parent_model, head


def _get_transformer_body(model):
    """Return a CausalLM's transformer body across supported Transformers APIs."""
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        return getter()
    body = getattr(model, "base_model", None)
    if body is None:
        raise ValueError(
            f"Model type {type(model).__name__} exposes neither get_base_model() "
            "nor base_model for parent-head scoring."
        )
    return body


def extract_token_logprobs_with_output_head(
    texts: list,
    target_model,
    parent_output_head,
    tokenizer,
    device,
    max_length: int = 512,
) -> list:
    """Score text with a target body and a parent causal-LM output head.

    The tokenizer must be compatible with the parent head vocabulary. Pythia
    parent and derived target checkpoints share that vocabulary, which is the
    supported Table 5 use case.
    """
    import torch
    import torch.nn.functional as F

    safe_texts = [text if isinstance(text, str) and text.strip() else "<empty>" for text in texts]
    encodings = tokenizer(
        safe_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    target_device = getattr(target_model, "device", device)
    input_ids = encodings["input_ids"].to(device=target_device, dtype=torch.long)
    attention_mask = encodings["attention_mask"].to(target_device)

    base_model = _get_transformer_body(target_model)
    target_hidden_size = getattr(target_model.config, "hidden_size", None)
    parent_hidden_size = getattr(parent_output_head, "in_features", None)
    if target_hidden_size is not None and parent_hidden_size is not None and target_hidden_size != parent_hidden_size:
        raise ValueError(
            "Target hidden size does not match the parent output-head input: "
            f"{target_hidden_size} != {parent_hidden_size}."
        )
    with torch.no_grad():
        hidden_states = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        head_device = next(parent_output_head.parameters()).device
        logits = parent_output_head(hidden_states.to(head_device))

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:].to(logits.device)
    shift_mask = attention_mask[:, 1:].to(logits.device)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    results = []
    for i in range(len(safe_texts)):
        results.append(token_logprobs[i][shift_mask[i].bool()].float().cpu().tolist())
    return results


def build_debug_record(
    text: str,
    token_logprobs: list,
    tokenizer,
    record_id: str,
    label: int,
) -> dict:
    """Build a debug dict showing tokens, ids, and their log-probs."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # scored_tokens correspond to tokens[1:] (causal shift)
    scored_ids = input_ids[1:len(token_logprobs) + 1]
    scored_tokens = tokens[1:len(token_logprobs) + 1]
    return {
        "id": record_id,
        "label": label,
        "text_prefix": text[:120],
        "tokens": tokens[:24],
        "token_ids": input_ids[:24],
        "scored_tokens": scored_tokens[:24],
        "token_logprobs": [round(x, 4) for x in token_logprobs[:24]],
        "note": (
            "token_logprobs[j] = log p(scored_tokens[j] | prefix), "
            "i.e. logprob for scored_tokens[j] comes from the previous position"
        ),
    }

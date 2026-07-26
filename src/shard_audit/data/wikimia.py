"""WikiMIA dataset loading utilities.

Dataset: swj0419/WikiMIA
Splits:  WikiMIA_length32, WikiMIA_length64, WikiMIA_length128, WikiMIA_length256

Label convention:
  label = 1: seen/member during pretraining
  label = 0: unseen/nonmember during pretraining

Text column: 'input'
Label column: 'label'
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DATASET_ID = "swj0419/WikiMIA"
KNOWN_SPLITS = [
    "WikiMIA_length32",
    "WikiMIA_length64",
    "WikiMIA_length128",
    "WikiMIA_length256",
]


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_wikimia_split(split: str, token: Optional[str] = None) -> list:
    """Load one WikiMIA split and return a list of {text, label, text_hash} dicts.

    Args:
        split:  one of KNOWN_SPLITS
        token:  HF token (not needed, dataset is public)

    Returns:
        list of dicts with keys: text, label, text_hash
    """
    from datasets import load_dataset  # noqa: PLC0415

    logger.info("Loading WikiMIA split: %s from %s", split, DATASET_ID)
    ds = load_dataset(DATASET_ID, split=split, token=token)

    records = []
    for row in ds:
        text = row["input"]
        label = int(row["label"])
        records.append({
            "text": text,
            "label": label,
            "text_hash": _text_hash(text),
        })

    n0 = sum(1 for r in records if r["label"] == 0)
    n1 = sum(1 for r in records if r["label"] == 1)
    logger.info("Loaded %d records: label0=%d label1=%d", len(records), n0, n1)
    return records


def inspect_all_splits(token: Optional[str] = None) -> list:
    """Return diagnostic info for all known splits."""
    from datasets import load_dataset  # noqa: PLC0415
    results = []
    for split in KNOWN_SPLITS:
        try:
            ds = load_dataset(DATASET_ID, split=split, token=token)
            labels = ds["label"]
            texts = ds["input"]
            n = len(ds)
            n0 = labels.count(0)
            n1 = labels.count(1)
            wcs = [len(t.split()) for t in texts]
            wc_min = min(wcs)
            wc_max = max(wcs)
            wc_mean = sum(wcs) / len(wcs) if wcs else 0
            results.append({
                "split": split,
                "n": n,
                "n_label0": n0,
                "n_label1": n1,
                "min_per_class": min(n0, n1),
                "wc_min": wc_min,
                "wc_max": wc_max,
                "wc_mean": round(wc_mean, 1),
                "feasible_500_500_test": min(n0, n1) >= 700,
                "feasible_200_200_test": min(n0, n1) >= 400,
            })
        except Exception as e:
            results.append({"split": split, "error": str(e)})
    return results

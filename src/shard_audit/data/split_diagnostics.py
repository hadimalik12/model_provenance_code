"""Reusable two-sample diagnostics for candidate witness/control shards.

These diagnostics can find measured distribution shift.  Failure to find shift
does not prove that two shards are exchangeable or drawn from equal
distributions.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\w\s]", re.UNICODE)


def lexical_js_divergence(
    left: Iterable[str], right: Iterable[str], *, min_count: int = 2
) -> dict[str, float | int]:
    """Return unigram Jensen-Shannon divergence in bits."""
    a = Counter(token.lower() for text in left for token in TOKEN_RE.findall(text))
    b = Counter(token.lower() for text in right for token in TOKEN_RE.findall(text))
    vocab = sorted(token for token in a.keys() | b.keys() if a[token] + b[token] >= min_count)
    if not vocab:
        return {"js_divergence_bits": 0.0, "vocabulary_size": 0,
                "left_token_count": sum(a.values()), "right_token_count": sum(b.values())}
    pa = np.asarray([a[token] for token in vocab], dtype=float)
    pb = np.asarray([b[token] for token in vocab], dtype=float)
    divergence = float(jensenshannon(pa / pa.sum(), pb / pb.sum(), base=2.0) ** 2)
    return {
        "js_divergence_bits": divergence,
        "vocabulary_size": len(vocab),
        "left_token_count": int(sum(a.values())),
        "right_token_count": int(sum(b.values())),
    }


def numeric_feature_diagnostics(
    features: dict[str, Sequence[float]], labels: Sequence[int]
) -> dict[str, dict[str, float]]:
    """Summarize numeric balance and two-sample KS statistics."""
    y = np.asarray(labels, dtype=int)
    result: dict[str, dict[str, float]] = {}
    for name, values in features.items():
        x = np.asarray(values, dtype=float)
        a, b = x[y == 1], x[y == 0]
        pooled = math.sqrt((float(np.var(a)) + float(np.var(b))) / 2.0)
        ks = ks_2samp(a, b, method="auto")
        result[name] = {
            "member_mean": float(np.mean(a)),
            "member_std": float(np.std(a)),
            "nonmember_mean": float(np.mean(b)),
            "nonmember_std": float(np.std(b)),
            "standardized_mean_difference": (
                float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else 0.0
            ),
            "ks_statistic": float(ks.statistic),
            "ks_pvalue_unadjusted": float(ks.pvalue),
        }
    # The numeric KS checks form one descriptive family. Holm adjustment is
    # valid under arbitrary dependence and keeps both raw effects and p-values
    # visible.
    ordered = sorted(result, key=lambda name: result[name]["ks_pvalue_unadjusted"])
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * result[name]["ks_pvalue_unadjusted"])
        running = max(running, adjusted)
        result[name]["ks_pvalue_holm"] = running
    return result


def text_duplicate_diagnostics(
    texts: Sequence[str], labels: Sequence[int], *, threshold: float = 0.8
) -> dict[str, float | int]:
    """Count exact and high-overlap cross-shard text pairs.

    Near duplicates use cosine similarity of binary word 5-gram vectors.  This
    is deterministic and much cheaper than an all-pairs dense comparison.
    """
    y = np.asarray(labels, dtype=int)
    left = np.flatnonzero(y == 1)
    right = np.flatnonzero(y == 0)
    normalized = [" ".join(TOKEN_RE.findall(text.lower())) for text in texts]
    left_hashes = Counter(hashlib.sha256(normalized[i].encode()).hexdigest() for i in left)
    right_hashes = Counter(hashlib.sha256(normalized[i].encode()).hexdigest() for i in right)
    exact = sum(left_hashes[h] * right_hashes[h] for h in left_hashes.keys() & right_hashes.keys())

    vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(5, 5), binary=True, use_idf=False,
        norm="l2", lowercase=True,
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return {"exact_cross_shard_pairs": int(exact), "near_cross_shard_pairs": 0,
                "near_threshold": threshold, "maximum_cross_shard_similarity": 0.0}
    similarities = matrix[left] @ matrix[right].T
    maximum = float(similarities.max()) if similarities.nnz else 0.0
    return {
        "exact_cross_shard_pairs": int(exact),
        "near_cross_shard_pairs": int((similarities >= threshold).sum()),
        "near_threshold": float(threshold),
        "maximum_cross_shard_similarity": maximum,
    }


def classifier_two_sample_test(
    features: np.ndarray,
    labels: Sequence[int],
    *,
    seed: int = 0,
    test_fraction: float = 0.3,
    permutations: int = 100,
) -> dict[str, float | int | str]:
    """Held-out linear C2ST with a label-permutation calibration.

    The same held-out indices and classifier settings are used for the observed
    statistic and every permutation.  The reported p-value is finite-sample
    corrected as ``(1 + exceedances) / (1 + permutations)``.
    """
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("features must be [n, d] and match labels")
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_fraction, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    x_test = scaler.transform(x[test_idx])

    def fit_score(permuted: np.ndarray) -> tuple[float, float, float]:
        clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
        clf.fit(x_train, permuted[train_idx])
        probability = clf.predict_proba(x_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        return (
            float(accuracy_score(permuted[test_idx], prediction)),
            float(balanced_accuracy_score(permuted[test_idx], prediction)),
            float(roc_auc_score(permuted[test_idx], probability)),
        )

    observed_accuracy, observed_balanced, observed_auc = fit_score(y)
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    for i in range(permutations):
        permuted = y.copy()
        # Preserve the stratified class counts in both partitions. This is a
        # conditional randomization test of label/embedding association.
        permuted[train_idx] = rng.permutation(y[train_idx])
        permuted[test_idx] = rng.permutation(y[test_idx])
        null[i] = fit_score(permuted)[1]
    pvalue = float((1 + np.sum(null >= observed_balanced)) / (permutations + 1))
    return {
        "test": "held-out linear classifier two-sample test",
        "n_examples": int(len(y)),
        "embedding_dimensions": int(x.shape[1]),
        "heldout_fraction": float(test_fraction),
        "accuracy": observed_accuracy,
        "balanced_accuracy": observed_balanced,
        "roc_auc": observed_auc,
        "permutations": int(permutations),
        "permutation_pvalue": pvalue,
        "null_balanced_accuracy_mean": float(np.mean(null)) if len(null) else float("nan"),
        "null_balanced_accuracy_95pct": (
            float(np.quantile(null, 0.95)) if len(null) else float("nan")
        ),
    }


def resolve_image_path(record: dict, manifest_dir: Path) -> Path:
    """Resolve absolute, repository-relative, or manifest-relative image paths."""
    candidate = Path(record["image_path"])
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    relative = manifest_dir / candidate
    if relative.exists():
        return relative.resolve()
    fallback = manifest_dir / "images" / candidate.name
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Cannot resolve image for {record.get('id')}: {candidate}")


def perceptual_hash(path: Path) -> int:
    """Return a deterministic 64-bit DCT perceptual hash."""
    from PIL import Image
    from scipy.fft import dctn

    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    low_frequency = dctn(pixels, type=2, norm="ortho")[:8, :8].reshape(-1)
    threshold = float(np.median(low_frequency[1:]))
    bits = low_frequency >= threshold
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def image_duplicate_diagnostics(
    paths: Sequence[Path], labels: Sequence[int], *, hamming_threshold: int = 5
) -> dict[str, float | int]:
    """Count exact-byte and perceptual near duplicates across two image shards."""
    y = np.asarray(labels, dtype=int)
    member = np.flatnonzero(y == 1)
    nonmember = np.flatnonzero(y == 0)
    digests = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    member_hashes = Counter(digests[i] for i in member)
    nonmember_hashes = Counter(digests[i] for i in nonmember)
    exact = sum(
        member_hashes[h] * nonmember_hashes[h]
        for h in member_hashes.keys() & nonmember_hashes.keys()
    )
    signatures = [perceptual_hash(path) for path in paths]
    near = 0
    minimum = 64
    for i in member:
        for j in nonmember:
            distance = (signatures[i] ^ signatures[j]).bit_count()
            minimum = min(minimum, distance)
            if distance <= hamming_threshold and digests[i] != digests[j]:
                near += 1
    return {
        "exact_cross_shard_pairs": int(exact),
        "perceptual_near_cross_shard_pairs_excluding_exact": int(near),
        "perceptual_hash_hamming_threshold": int(hamming_threshold),
        "minimum_cross_shard_hamming_distance": int(minimum),
    }

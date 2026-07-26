"""Train/test splitting utilities for shard membership auditing.

Produces class-balanced, deterministic splits with no text-hash overlap
between train and test sets.
"""

import hashlib
import random
from typing import Optional


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stratified_train_test_split(
    member_records: list,
    nonmember_records: list,
    num_train_per_class: int = 100,
    num_test_per_class: int = 100,
    seed: int = 0,
    num_train_member: Optional[int] = None,
    num_train_nonmember: Optional[int] = None,
    num_test_member: Optional[int] = None,
    num_test_nonmember: Optional[int] = None,
) -> tuple:
    """Split member and nonmember records into train/test sets.

    Args:
        member_records: dicts with at least 'text' and 'label'==1
        nonmember_records: dicts with at least 'text' and 'label'==0
        num_train_per_class: default fallback per-class count for train
        num_test_per_class: default fallback per-class count for test
        seed: RNG seed for determinism
        num_train_member: exact member count for train
        num_train_nonmember: exact non-member count for train
        num_test_member: exact member count for test
        num_test_nonmember: exact non-member count for test

    Returns:
        (train_records, test_records) — each a shuffled mix of member+nonmember
    """
    n_train_m = num_train_member if num_train_member is not None else num_train_per_class
    n_train_nm = num_train_nonmember if num_train_nonmember is not None else num_train_per_class
    n_test_m = num_test_member if num_test_member is not None else num_test_per_class
    n_test_nm = num_test_nonmember if num_test_nonmember is not None else num_test_per_class

    m_needed = n_train_m + n_test_m
    nm_needed = n_train_nm + n_test_nm

    if len(member_records) < m_needed:
        raise ValueError(f"Not enough member records: need {m_needed}, got {len(member_records)}")
    if len(nonmember_records) < nm_needed:
        raise ValueError(f"Not enough nonmember records: need {nm_needed}, got {len(nonmember_records)}")

    rng = random.Random(seed)

    def _select(records, n_train, n_test):
        shuffled = list(records)
        rng.shuffle(shuffled)
        train = shuffled[:n_train]
        test = shuffled[n_train : n_train + n_test]
        return train, test

    m_train, m_test = _select(member_records, n_train_m, n_test_m)
    nm_train, nm_test = _select(nonmember_records, n_train_nm, n_test_nm)

    # Verify no hash overlap between train and test
    train_hashes = {r["text_hash"] for r in m_train + nm_train}
    test_hashes = {r["text_hash"] for r in m_test + nm_test}
    overlap = train_hashes & test_hashes
    if overlap:
        raise ValueError(
            f"Hash overlap between train and test: {len(overlap)} examples. "
            "This indicates near-duplicate or identical texts across the MIMIR splits."
        )

    def _tag(records, phase):
        tagged = []
        for r in records:
            tagged.append({**r, "phase_split": phase})
        return tagged

    train_all = _tag(m_train, "train") + _tag(nm_train, "train")
    test_all = _tag(m_test, "test") + _tag(nm_test, "test")

    rng.shuffle(train_all)
    rng.shuffle(test_all)

    return train_all, test_all

"""Regression test for MIMIR source pools containing a shared text."""

from scripts.data.prepare_mimir_domain import _drop_cross_label_duplicates
from src.shard_audit.data.splitting import stratified_train_test_split


def _record(identifier, text_hash, label):
    return {"id": identifier, "text_hash": text_hash, "text": identifier, "label": label}


def test_cross_label_duplicate_is_removed_before_split():
    members = [_record("m0", "shared", 1), _record("m1", "m1", 1), _record("m2", "m2", 1)]
    nonmembers = [_record("n0", "shared", 0), _record("n1", "n1", 0), _record("n2", "n2", 0)]
    nonmembers, removed = _drop_cross_label_duplicates(members, nonmembers)
    assert removed == 1
    assert {record["text_hash"] for record in members}.isdisjoint(
        {record["text_hash"] for record in nonmembers}
    )

    train, test = stratified_train_test_split(
        members, nonmembers, num_train_per_class=1, num_test_per_class=1, seed=0
    )
    assert {record["text_hash"] for record in train}.isdisjoint(
        {record["text_hash"] for record in test}
    )

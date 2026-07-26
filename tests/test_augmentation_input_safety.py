"""Regression tests for tokenizable outputs from the reusable augmenters."""

from src.shard_audit.scoring.text_augmentations import augment_code_snippet_4views
from src.shard_audit.scoring.text_augmentations_domains import augment_by_domain
from src.shard_audit.scoring.table5_safe_augmentations import table5_variant_views


def test_blank_code_input_has_nonempty_views():
    assert all(view.strip() for view in augment_code_snippet_4views("   "))


def test_blank_domain_inputs_have_nonempty_views():
    for domain in ("arxiv", "dm_mathematics", "github", "pile_cc", "wikipedia_en"):
        assert all(view.strip() for view in augment_by_domain("\n\t", domain))
        variants = table5_variant_views("\n\t", domain)
        assert all(view.strip() for views in variants.values() for view in views)


def test_comment_only_code_remains_nonempty():
    comment = "# configuration only\n# retain this rule"
    assert all(view.strip() for view in augment_code_snippet_4views(comment))

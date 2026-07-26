"""Regression checks for non-empty Table 5 augmentation views."""

from src.shard_audit.scoring.table5_safe_augmentations import table5_variant_views


def test_comment_only_code_keeps_nonempty_views():
    """Python's AST drops comments, so that view must fall back to source text."""
    text = "# ProGuard configuration comment\n# Keep this rule"
    variants = table5_variant_views(text, "github")
    assert variants["ablation_python_ast_normalize"] == [text]
    assert all(view.strip() for views in variants.values() for view in views)


def test_all_domains_return_nonempty_views():
    for domain in ("arxiv", "dm_mathematics", "github", "pile_cc", "wikipedia_en"):
        variants = table5_variant_views("Example x = 2. <b>Text</b> \\macro {x}", domain)
        assert len(variants["safe_full"]) == 4
        assert all(view.strip() for views in variants.values() for view in views)

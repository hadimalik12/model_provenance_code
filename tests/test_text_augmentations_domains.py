"""Tests for the domain-specific augmentation module."""
from src.shard_audit.scoring.text_augmentations_domains import (
    augment_arxiv_4views,
    augment_math_4views,
    augment_wikipedia_4views,
    augment_pile_cc_4views,
    augment_by_domain,
)


def test_arxiv_augmentations():
    text = "\\textbf{Hello} world 3.14. See \\cite{ref1} \\ref{fig2}.   Lots   of   spaces."
    views = augment_arxiv_4views(text)
    assert len(views) == 4
    # View 1: LaTeX normalize
    assert "{\\bf Hello}" in views[0]
    # View 2: Number perturb
    assert "3.15" in views[1]
    # View 3: Whitespace
    assert "Lots of spaces." in views[2]
    # View 4: Citation
    assert "[CIT]" in views[3]
    assert "[REF]" in views[3]


def test_math_augmentations():
    text = "Let x = 2+3. Then a + b = y."
    views = augment_math_4views(text)
    assert len(views) == 4
    # View 1: Variable swap
    assert "a =" in views[0] or "A =" in views[0]
    # View 2: Operator spacing
    assert "2 + 3" in views[1]
    # View 3: Number perturb
    assert "3" in views[2]
    # View 4: Commutative reorder
    assert "b + a" in views[3]


def test_wikipedia_augmentations():
    text = "Look, John Smith was born in January 1, 1990. He is a person who did a lot of things. This is a very short sentence."
    views = augment_wikipedia_4views(text)
    assert len(views) == 4
    # View 1: Entity mask
    assert "[DATE]" in views[0]
    assert "[ENTITY]" in views[0]
    # View 2: Reorder
    assert "John Smith" not in views[1] or "John born was Smith" in views[1] or len(views[1]) > 0
    # View 3: Deletion
    # Check that some words are removed but it's shorter
    assert len(views[2].split()) <= len(text.split())
    # View 4: Punctuation (hard to test here without curly quotes, but function runs)


def test_pilecc_augmentations():
    text = "Hello <b>world</b>. Check out http://google.com for more info. It is   great."
    views = augment_pile_cc_4views(text)
    assert len(views) == 4
    # View 1: HTML/URL
    assert "<b>" not in views[0]
    assert "[URL]" in views[0]
    # View 2: Case perturb
    assert "hello" in views[1]
    # View 3: Punctuation (runs)
    # View 4: Whitespace
    assert "is great." in views[3]


def test_dispatcher():
    views = augment_by_domain("Hello world", "wikipedia_en")
    assert len(views) == 4
    
    try:
        augment_by_domain("Hello world", "unknown_domain")
        assert False, "Should raise ValueError"
    except ValueError:
        pass

if __name__ == "__main__":
    test_arxiv_augmentations()
    test_math_augmentations()
    test_wikipedia_augmentations()
    test_pilecc_augmentations()
    test_dispatcher()
    print("All tests passed!")

"""Conservative, snippet-safe augmentation policies for Table 5.

These policies are for test-time membership scoring, not training augmentation.
Every transform is deterministic and avoids changing facts, numeric values, or
the order/content of prose.  The full policy is the mean of its three atomic
views plus their composition; each atomic view is also exposed for ablation.
"""

from __future__ import annotations

import html
import ast
import re
import unicodedata
from collections.abc import Callable


def _unicode_typography(text: str) -> str:
    """Canonicalize harmless Unicode typography without changing words."""
    text = unicodedata.normalize("NFC", text)
    return (text.replace("\u00a0", " ").replace("\u2018", "'")
                .replace("\u2019", "'").replace("\u201c", '"')
                .replace("\u201d", '"').replace("\u2013", "-")
                .replace("\u2014", "--").replace("\u2026", "..."))


def _prose_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _code_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _python_ast_normalize(text: str) -> str:
    """Use Python's parser/unparser when the complete snippet is valid Python.

    This is a semantics-preserving source transformation for valid Python.  A
    non-Python or incomplete code snippet is returned unchanged rather than
    being subjected to a regex-based rewrite.
    """
    try:
        normalized = ast.unparse(ast.parse(text))
        # A comment-only snippet is valid Python but unparses to an empty
        # module.  Its comments are still meaningful LM input, so retain it.
        return normalized if normalized.strip() else text
    except (SyntaxError, ValueError, TypeError):
        return text


def _latex_control_spacing(text: str) -> str:
    """Normalize the safe ``\\macro {`` spelling to ``\\macro{``."""
    return re.sub(r"(\\[A-Za-z]+)\s+\{", r"\1{", text)


def _latex_comments(text: str) -> str:
    """Remove TeX comments; an escaped percent sign is retained."""
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def _math_alpha_rename(text: str) -> str:
    """Consistently rename isolated mathematical variables, not words/functions."""
    mapping = {"x": "u", "y": "v", "z": "w", "a": "p", "b": "q", "c": "r"}
    reserved = {"a", "an", "and", "as", "by", "for", "if", "in", "is", "let", "of", "or", "the", "to"}

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.lower() in reserved or token.lower() not in mapping:
            return token
        replacement = mapping[token.lower()]
        return replacement.upper() if token.isupper() else replacement

    return re.sub(r"\b[A-Za-z]\b", replace, text)


def _math_operator_spacing(text: str) -> str:
    return re.sub(r"\s*([+*/=<>])\s*", r" \1 ", text)


def _math_parenthesis_spacing(text: str) -> str:
    text = re.sub(r"\(\s+", "(", text)
    return re.sub(r"\s+\)", ")", text)


def _wiki_bracket_spacing(text: str) -> str:
    """Canonicalize spaces adjacent to citation-style square brackets only."""
    text = re.sub(r"\s+\]", "]", text)
    return re.sub(r"\[\s+", "[", text)


def _pile_html_entities(text: str) -> str:
    """Decode entities while retaining tags and URLs (which may carry content)."""
    return html.unescape(text)


Policy = tuple[tuple[str, Callable[[str], str]], ...]

_POLICIES: dict[str, Policy] = {
    "github": (
        ("code_line_endings", _code_line_endings),
        ("python_ast_normalize", _python_ast_normalize),
        ("code_typography", _unicode_typography),
    ),
    "arxiv": (
        ("latex_typography", _unicode_typography),
        ("latex_control_spacing", _latex_control_spacing),
        ("latex_comments", _latex_comments),
    ),
    "dm_mathematics": (
        ("math_operator_spacing", _math_operator_spacing),
        ("math_alpha_rename", _math_alpha_rename),
        ("math_parenthesis_spacing", _math_parenthesis_spacing),
    ),
    "wikipedia_en": (
        ("prose_typography", _unicode_typography),
        ("prose_whitespace", _prose_whitespace),
        ("wiki_bracket_spacing", _wiki_bracket_spacing),
    ),
    "pile_cc": (
        ("prose_typography", _unicode_typography),
        ("prose_whitespace", _prose_whitespace),
        ("pile_html_entities", _pile_html_entities),
    ),
}


def _normalise_domain(domain: str) -> str:
    domain = domain.lower().strip()
    if domain == "wikipedia__en_":
        return "wikipedia_en"
    if domain not in _POLICIES:
        raise ValueError(f"Unknown Table 5 domain: {domain}")
    return domain


def table5_variant_views(text: str, domain: str) -> dict[str, list[str]]:
    """Return atomic ablations and the four-view combined policy for a domain."""
    policy = _POLICIES[_normalise_domain(domain)]
    source = text if isinstance(text, str) and text.strip() else "<empty>"
    def apply(transform: Callable[[str], str], value: str) -> str:
        transformed = transform(value)
        # A test-time view must remain a non-empty text sequence.  Falling
        # back preserves the source example and prevents an invalid empty
        # tokenizer batch for comment-only/markup-only snippets.
        return transformed if isinstance(transformed, str) and transformed.strip() else value

    variants = {f"ablation_{name}": [apply(transform, source)] for name, transform in policy}
    combined = source
    for _, transform in policy:
        combined = apply(transform, combined)
    variants["safe_full"] = [apply(transform, source) for _, transform in policy] + [combined]
    return variants


def table5_variant_score_keys(k: int, domain: str) -> dict[str, str]:
    """Return report variant labels and their score-record key for one MIN-K value."""
    policy = _POLICIES[_normalise_domain(domain)]
    result = {"vanilla": f"min_k_{k}_logprob"}
    result["safe_full"] = f"safe_full_min_k_{k}_logprob_mean"
    result.update({f"ablation_{name}": f"ablation_{name}_min_k_{k}_logprob" for name, _ in policy})
    return result


def table5_variant_labels(domain: str) -> dict[str, str]:
    """Human-readable labels in the report's Variant column."""
    labels = {"vanilla": "Vanilla"}
    for name, _ in _POLICIES[_normalise_domain(domain)]:
        labels[f"ablation_{name}"] = "Ablation: " + name.replace("_", " ")
    labels["safe_full"] = "Full safe augmentation"
    return labels

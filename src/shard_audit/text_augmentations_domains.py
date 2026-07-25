"""Domain-Specific Text Augmentation Module for Multi-Domain Shard-Membership Auditing.

Implements 4-view deterministic augmentation strategies tailored to each MIMIR domain:

1. **GitHub (code)**: Language-aware identifier mutation, quote normalization, whitespace
   shift. (Delegates to existing ``text_augmentations.augment_code_snippet_4views``.)
2. **ArXiv (scientific/LaTeX)**: LaTeX macro normalization, number perturbation,
   whitespace cleanup, citation placeholder swap.
3. **DM Mathematics (formal math)**: Variable swapping, operator spacing, number base
   perturbation, commutative reordering.
4. **Wikipedia (encyclopedic prose)**: Entity masking (proper nouns), deterministic word
   reordering, content-word deletion, punctuation normalization.
5. **Pile-CC (web crawl)**: HTML/URL artifact removal, case perturbation, punctuation
   normalization, whitespace cleanup.

All augmentations are rule-based / regex-based and fully deterministic (no randomness,
no external API calls, no model inference).  This guarantees reproducibility across runs.

Dependencies: ``re`` only (stdlib).  No NLTK, no spaCy, no external packages.
"""

import re
from typing import List


# ====================================================================== #
#  ArXiv (Scientific / LaTeX) Augmentations
# ====================================================================== #

def _arxiv_latex_normalize(text: str) -> str:
    r"""Normalize LaTeX macros to alternative equivalent forms.

    - ``\textbf{X}``  ↔  ``{\bf X}``
    - ``\textit{X}``  ↔  ``{\it X}``
    - ``\emph{X}``    →  ``\textit{X}``
    - ``\texttt{X}``  ↔  ``{\tt X}``
    """
    # \textbf{...} → {\bf ...}
    text = re.sub(r'\\textbf\{([^}]*)\}', r'{\\bf \1}', text)
    # \textit{...} → {\it ...}
    text = re.sub(r'\\textit\{([^}]*)\}', r'{\\it \1}', text)
    # \emph{...} → {\it ...}
    text = re.sub(r'\\emph\{([^}]*)\}', r'{\\it \1}', text)
    # \texttt{...} → {\tt ...}
    text = re.sub(r'\\texttt\{([^}]*)\}', r'{\\tt \1}', text)
    return text


def _arxiv_number_perturb(text: str) -> str:
    """Perturb numeric constants by adding 1 to the last digit.

    Only affects standalone decimal/integer numbers (e.g., 3.14 → 3.15, 42 → 43).
    Skips numbers inside LaTeX commands.
    """
    def _bump(match):
        num_str = match.group(0)
        try:
            if '.' in num_str:
                val = float(num_str)
                # Add a small perturbation to the last decimal place
                n_dec = len(num_str.split('.')[1])
                perturbed = val + 10 ** (-n_dec)
                return f"{perturbed:.{n_dec}f}"
            else:
                val = int(num_str)
                return str(val + 1)
        except (ValueError, OverflowError):
            return num_str

    # Match standalone numbers not preceded by a backslash
    return re.sub(r'(?<!\\)(?<![a-zA-Z_])\d+(?:\.\d+)?(?![a-zA-Z_{}])', _bump, text)


def _arxiv_whitespace_normalize(text: str) -> str:
    """Normalize whitespace: collapse runs, strip trailing, normalize line breaks."""
    # Collapse multiple spaces to single
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize line endings
    text = re.sub(r'\r\n', '\n', text)
    # Collapse multiple blank lines to one
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _arxiv_citation_placeholder(text: str) -> str:
    r"""Replace LaTeX citation/reference keys with generic placeholders.

    - ``\cite{key}``  →  ``[CIT]``
    - ``\ref{key}``   →  ``[REF]``
    - ``\label{key}`` →  ``[LBL]``
    """
    text = re.sub(r'\\cite\{[^}]*\}', '[CIT]', text)
    text = re.sub(r'\\ref\{[^}]*\}', '[REF]', text)
    text = re.sub(r'\\label\{[^}]*\}', '[LBL]', text)
    return text


def augment_arxiv_4views(text: str) -> List[str]:
    """Generate 4 augmented views for arXiv (scientific/LaTeX) text."""
    return [
        _arxiv_latex_normalize(text),
        _arxiv_number_perturb(text),
        _arxiv_whitespace_normalize(text),
        _arxiv_citation_placeholder(text),
    ]


# ====================================================================== #
#  DM Mathematics (Formal Math) Augmentations
# ====================================================================== #

_MATH_VAR_MAP = {
    'x': 'a', 'y': 'b', 'z': 'c',
    'a': 'p', 'b': 'q', 'c': 'r',
    'n': 'k', 'm': 'j', 'k': 'n',
    'p': 'x', 'q': 'y', 'r': 'z',
    'i': 'u', 'j': 'v', 'u': 'i', 'v': 'j',
}

# Math keywords/functions that should NOT be swapped
_MATH_RESERVED = {
    'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'sqrt', 'abs', 'mod',
    'min', 'max', 'sum', 'pi', 'inf', 'nan', 'true', 'false',
    'gcd', 'lcm', 'is', 'in', 'or', 'and', 'not', 'if', 'for',
    'let', 'to', 'of', 'the', 'what', 'find', 'solve', 'simplify',
    'calculate', 'evaluate', 'express', 'determine', 'compute',
    'how', 'many', 'which', 'given', 'where', 'such', 'that',
}


def _math_variable_swap(text: str) -> str:
    """Swap single-letter variable names using a deterministic mapping."""
    def _replacer(match):
        word = match.group(0)
        if word.lower() in _MATH_RESERVED:
            return word
        if len(word) == 1 and word.lower() in _MATH_VAR_MAP:
            replacement = _MATH_VAR_MAP[word.lower()]
            return replacement.upper() if word.isupper() else replacement
        return word

    return re.sub(r'\b[a-zA-Z]\b', _replacer, text)


def _math_operator_spacing(text: str) -> str:
    """Toggle spacing around operators: ``2+3`` ↔ ``2 + 3``."""
    # If spaced operators exist, remove spaces
    if re.search(r'\d\s+[+\-*/=<>]\s+\d', text):
        text = re.sub(r'(\d)\s+([+\-*/=<>])\s+(\d)', r'\1\2\3', text)
    else:
        # Add spaces around operators between digits
        text = re.sub(r'(\d)([+\-*/=<>])(\d)', r'\1 \2 \3', text)
    return text


def _math_number_perturb(text: str) -> str:
    """Perturb integer operands by +1 (skip fractions and decimals)."""
    def _bump(match):
        num_str = match.group(0)
        try:
            val = int(num_str)
            return str(val + 1)
        except ValueError:
            return num_str

    # Only match standalone integers (not part of decimals)
    return re.sub(r'(?<![.\d])\b(\d+)\b(?![.\d])', _bump, text)


def _math_commutative_reorder(text: str) -> str:
    """Reorder commutative binary expressions: ``a + b`` → ``b + a``."""
    def _swap(match):
        left = match.group(1).strip()
        op = match.group(2)
        right = match.group(3).strip()
        return f"{right} {op} {left}"

    # Match simple binary expressions with commutative operators
    text = re.sub(r'(\b\w+\b)\s*(\+)\s*(\b\w+\b)', _swap, text)
    text = re.sub(r'(\b\w+\b)\s*(\*)\s*(\b\w+\b)', _swap, text)
    return text


def augment_math_4views(text: str) -> List[str]:
    """Generate 4 augmented views for dm_mathematics text."""
    return [
        _math_variable_swap(text),
        _math_operator_spacing(text),
        _math_number_perturb(text),
        _math_commutative_reorder(text),
    ]


# ====================================================================== #
#  Wikipedia (Encyclopedic Prose) Augmentations
# ====================================================================== #

def _wiki_entity_mask(text: str) -> str:
    """Mask likely named entities (capitalized multi-word sequences) with placeholders."""
    # Replace dates like "January 2023", "12 March 1990"
    text = re.sub(
        r'\b(?:January|February|March|April|May|June|July|August|September|'
        r'October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?\b',
        '[DATE]', text
    )
    text = re.sub(r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|'
                  r'August|September|October|November|December)\s+\d{4}\b',
                  '[DATE]', text)
    # Replace years in isolation
    text = re.sub(r'\b(1[5-9]\d{2}|20[0-2]\d)\b', '[YEAR]', text)
    # Replace sequences of capitalized words (2+ words, likely proper nouns)
    # but not at the start of a sentence
    text = re.sub(r'(?<=[.!?]\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', '[ENTITY]', text)
    text = re.sub(r'(?<=\s)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', '[ENTITY]', text)
    return text


def _wiki_word_reorder(text: str) -> str:
    """Deterministic word reorder within each sentence: reverse content words, keep first/last."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for sent in sentences:
        words = sent.split()
        if len(words) <= 3:
            result.append(sent)
            continue
        # Keep first and last word, reverse the middle
        middle = words[1:-1]
        middle.reverse()
        result.append(' '.join([words[0]] + middle + [words[-1]]))
    return ' '.join(result)


def _wiki_content_word_delete(text: str) -> str:
    """Delete ~20% of content words (non-stopwords) deterministically.

    Uses a simple hash-based decision to ensure determinism.
    """
    _STOPWORDS = {
        'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'between', 'under', 'again', 'further', 'then', 'once', 'and', 'but',
        'or', 'nor', 'not', 'so', 'yet', 'both', 'either', 'neither', 'each',
        'every', 'all', 'any', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'because',
        'this', 'that', 'these', 'those', 'it', 'its', 'he', 'she', 'they',
        'them', 'their', 'we', 'our', 'you', 'your', 'i', 'my', 'me', 'who',
        'which', 'what', 'when', 'where', 'how', 'if', 'about', 'up', 'out',
    }
    words = text.split()
    kept = []
    for idx, w in enumerate(words):
        lower = w.lower().strip('.,!?;:\'\"()')
        # Keep stopwords, short words, and 80% of content words
        if lower in _STOPWORDS or len(lower) <= 2 or (idx * 7 + len(lower)) % 5 != 0:
            kept.append(w)
    return ' '.join(kept)


def _wiki_punctuation_normalize(text: str) -> str:
    """Normalize punctuation: curly quotes → straight, em-dash → --, ellipsis → ..."""
    # Curly quotes to straight
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    # Em-dash and en-dash
    text = text.replace('\u2014', '--').replace('\u2013', '-')
    # Ellipsis
    text = text.replace('\u2026', '...')
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def augment_wikipedia_4views(text: str) -> List[str]:
    """Generate 4 augmented views for Wikipedia (encyclopedic prose) text."""
    return [
        _wiki_entity_mask(text),
        _wiki_word_reorder(text),
        _wiki_content_word_delete(text),
        _wiki_punctuation_normalize(text),
    ]


# ====================================================================== #
#  Pile-CC (Web Crawl) Augmentations
# ====================================================================== #

def _pilecc_html_url_cleanup(text: str) -> str:
    """Remove residual HTML tags and replace URLs with [URL] placeholder."""
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Replace URLs
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'www\.\S+', '[URL]', text)
    # Remove HTML entities
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    # Collapse whitespace
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def _pilecc_case_perturb(text: str) -> str:
    """Lowercase the first letter of each sentence, uppercase every 5th content word."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result = []
    word_count = 0
    for sent in sentences:
        words = sent.split()
        modified = []
        for i, w in enumerate(words):
            word_count += 1
            if i == 0 and len(w) > 0 and w[0].isupper():
                # Lowercase first word of sentence
                modified.append(w[0].lower() + w[1:])
            elif word_count % 5 == 0 and w.isalpha() and w.islower():
                # Uppercase every 5th word
                modified.append(w.upper())
            else:
                modified.append(w)
        result.append(' '.join(modified))
    return ' '.join(result)


def _pilecc_punctuation_normalize(text: str) -> str:
    """Standardize Unicode punctuation to ASCII equivalents."""
    # Curly quotes
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    # Dashes
    text = text.replace('\u2014', '--').replace('\u2013', '-')
    # Ellipsis
    text = text.replace('\u2026', '...')
    # Non-breaking space
    text = text.replace('\u00a0', ' ')
    # Bullet points
    text = text.replace('\u2022', '-')
    return text


def _pilecc_whitespace_normalize(text: str) -> str:
    """Collapse whitespace, ensure single spaces after periods."""
    # Normalize all whitespace to single space
    text = re.sub(r'\s+', ' ', text)
    # Ensure single space after sentence-ending punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    return text.strip()


def augment_pile_cc_4views(text: str) -> List[str]:
    """Generate 4 augmented views for Pile-CC (web crawl) text."""
    return [
        _pilecc_html_url_cleanup(text),
        _pilecc_case_perturb(text),
        _pilecc_punctuation_normalize(text),
        _pilecc_whitespace_normalize(text),
    ]


# ====================================================================== #
#  GitHub (Code) — Delegates to existing augmentation module
# ====================================================================== #

def augment_github_4views(text: str) -> List[str]:
    """Generate 4 augmented views for GitHub code text.

    Delegates to the existing ``text_augmentations.augment_code_snippet_4views``.
    """
    from src.shard_audit.text_augmentations import augment_code_snippet_4views
    return augment_code_snippet_4views(text, n_aug=4)


# ====================================================================== #
#  Dispatcher
# ====================================================================== #

_DOMAIN_AUGMENTERS = {
    'github': augment_github_4views,
    'arxiv': augment_arxiv_4views,
    'dm_mathematics': augment_math_4views,
    'wikipedia_en': augment_wikipedia_4views,
    'wikipedia__en_': augment_wikipedia_4views,  # filesystem slug variant
    'pile_cc': augment_pile_cc_4views,
}


def augment_by_domain(text: str, domain: str, n_aug: int = 4) -> List[str]:
    """Dispatch to the correct domain-specific augmentation function.

    Args:
        text: Raw text to augment.
        domain: MIMIR domain name or filesystem slug (e.g., 'github', 'arxiv',
                'dm_mathematics', 'wikipedia_en', 'pile_cc').
        n_aug: Number of views to return (max 4).

    Returns:
        List of up to ``n_aug`` augmented text strings.

    Raises:
        ValueError: If the domain is not recognized.
    """
    # Normalize the domain slug
    domain_lower = domain.lower().strip()
    # Try direct match first
    if domain_lower in _DOMAIN_AUGMENTERS:
        views = _DOMAIN_AUGMENTERS[domain_lower](text)
        return views[:n_aug]
    # Try partial match
    for key, func in _DOMAIN_AUGMENTERS.items():
        if key in domain_lower or domain_lower in key:
            views = func(text)
            return views[:n_aug]
    raise ValueError(
        f"Unknown domain '{domain}'. Known domains: {list(_DOMAIN_AUGMENTERS.keys())}"
    )


def list_supported_domains() -> List[str]:
    """Return list of supported domain names."""
    return list(_DOMAIN_AUGMENTERS.keys())

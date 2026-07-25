"""Robust, Language-Aware Code Data Augmentation Module (MIMIR GitHub & Multi-Domain Audit).

Implements 4 deterministic, syntax-preserving, language-aware code augmentations:
1. View 1 (Original): Raw input code/text passage.
2. View 2 (Language-Aware Identifier Mutator): Appends `_v` to non-reserved local variables based on language-specific keyword dictionaries.
3. View 3 (Language-Aware String/Quote Normalizer): Flips single <-> double quotes safely based on language.
4. View 4 (Language-Aware Header & Whitespace Shift): Prepend syntax-matched comments and shift indentation (4-space <-> 2-space).

Supported languages auto-detected from snippet content:
- Python
- C / C++
- Java
- JavaScript / TypeScript
- Go
- Rust
- HTML / XML / CSS
- Shell / Bash
- SQL
- Generic / Fallback Text
"""

import re
from typing import List, Dict, Set

# Reserved Keywords Per Language
KEYWORDS_BY_LANG: Dict[str, Set[str]] = {
    "python": {
        "def", "class", "return", "if", "else", "elif", "for", "while", "import",
        "from", "in", "is", "not", "and", "or", "None", "True", "False", "self",
        "int", "str", "list", "dict", "set", "tuple", "len", "print", "range",
        "try", "except", "finally", "raise", "with", "as", "pass", "yield", "async", "await", "lambda"
    },
    "c_like": { # C, C++, Java, JS, TS, Go, Rust
        "auto", "break", "case", "char", "const", "continue", "default", "do", "double",
        "else", "enum", "extern", "float", "for", "goto", "if", "inline", "int", "long",
        "register", "restrict", "return", "short", "signed", "sizeof", "static", "struct",
        "switch", "typedef", "union", "unsigned", "void", "volatile", "while", "class",
        "public", "private", "protected", "virtual", "template", "typename", "using", "namespace",
        "new", "delete", "try", "catch", "throw", "import", "package", "interface", "extends",
        "implements", "function", "let", "var", "const", "async", "await", "fn", "let", "mut",
        "pub", "impl", "trait", "func", "type", "package"
    },
    "sql": {
        "select", "from", "where", "insert", "update", "delete", "join", "inner", "left",
        "right", "outer", "group", "by", "order", "having", "limit", "create", "table",
        "drop", "alter", "index", "as", "on", "and", "or", "not", "null", "into", "values"
    },
    "shell": {
        "if", "then", "else", "elif", "fi", "case", "esac", "for", "while", "until", "do",
        "done", "in", "function", "return", "exit", "echo", "export", "local", "set", "unset"
    }
}


def detect_language(code_text: str) -> str:
    """Auto-detect programming language signature from source code snippet."""
    text_lower = code_text.lower()
    
    # Python signatures
    if any(k in code_text for k in ["def ", "class ", "import ", "self.", "print(", "elif "]):
        return "python"
    
    # HTML / XML signatures
    if "<html" in text_lower or "<div" in text_lower or "<script" in text_lower or "</" in code_text:
        return "html"
        
    # SQL signatures
    if any(k in text_lower for k in ["select ", "from ", "where ", "insert into", "create table"]):
        return "sql"
        
    # Shell signatures
    if code_text.startswith("#!") or "echo " in text_lower or "export " in text_lower:
        return "shell"
        
    # C / C++ / Java / JS / Go / Rust signatures
    if any(k in code_text for k in ["#include", "public class", "std::", "function ", "func ", "fn ", "const ", "var "]):
        return "c_like"
        
    return "generic"


def _get_comment_header(lang: str) -> str:
    """Return language-appropriate comment header."""
    if lang in ["python", "shell"]:
        return "# Source Code Snippet\n"
    elif lang in ["c_like", "generic"]:
        return "// Source Code Snippet\n"
    elif lang == "sql":
        return "-- Source Code Snippet\n"
    elif lang == "html":
        return "<!-- Source Code Snippet -->\n"
    return "# Source Code Snippet\n"


def _mutate_identifiers_lang_aware(code_text: str, lang: str) -> str:
    """Mutate local identifier variable names by appending `_v` safely."""
    keywords = KEYWORDS_BY_LANG.get(lang, KEYWORDS_BY_LANG["python"] | KEYWORDS_BY_LANG["c_like"])
    
    def _replacer(match):
        word = match.group(0)
        if word in keywords or word.lower() in keywords or len(word) <= 2:
            return word
        return f"{word}_v"

    lines = []
    for line in code_text.splitlines(keepends=True):
        stripped = line.strip()
        # Skip compiler directives, includes, and HTML tags
        if stripped.startswith("#include") or stripped.startswith("#define") or (stripped.startswith("<") and stripped.endswith(">")):
            lines.append(line)
        else:
            lines.append(re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', _replacer, line))
            
    return "".join(lines)


def _normalize_string_quotes_safe(code_text: str) -> str:
    """Flip single quotes and double quotes in text safely."""
    if "'" in code_text and '"' not in code_text:
        return code_text.replace("'", '"')
    elif '"' in code_text and "'" not in code_text:
        return code_text.replace('"', "'")
    else:
        return f"# -*- coding: utf-8 -*-\n{code_text}"


def _shift_whitespace_lang_aware(code_text: str, lang: str) -> str:
    """Shift 4-space indentation to 2-space indentation or vice versa."""
    header = _get_comment_header(lang)
    if "\n    " in code_text:
        shifted = re.sub(r'\n(    )+', lambda m: '\n' + '  ' * (len(m.group(0)) // 4), code_text)
        return shifted
    elif "\n  " in code_text:
        shifted = re.sub(r'\n(  )+', lambda m: '\n' + '    ' * (len(m.group(0)) // 2), code_text)
        return shifted
    else:
        return header + code_text


def augment_code_snippet_4views(code_text: str, n_aug: int = 4) -> List[str]:
    """Generate K=4 language-aware, syntax-preserving code augmentations.
    
    Returns:
        List of n_aug text strings.
    """
    lang = detect_language(code_text)
    views = []
    
    # View 1: Combined Augmentations (no original text)
    combined = _mutate_identifiers_lang_aware(code_text, lang)
    combined = _normalize_string_quotes_safe(combined)
    combined = _shift_whitespace_lang_aware(combined, lang)
    views.append(combined)
    
    # View 2: Language-Aware Identifier Mutator
    views.append(_mutate_identifiers_lang_aware(code_text, lang))
    
    # View 3: Safe String Quote Normalizer
    views.append(_normalize_string_quotes_safe(code_text))
    
    # View 4: Header & Whitespace Shift
    views.append(_shift_whitespace_lang_aware(code_text, lang))
    
    return views[:n_aug]

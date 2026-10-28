"""Conservative text cleaning for enterprise documents."""

from __future__ import annotations

import re

_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize document text without destroying semantic structure.

    Cleaning is intentionally conservative. Markdown headings and paragraph
    boundaries are preserved because later chunking can use that structure.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []
    for line in text.split("\n"):
        normalized = _HORIZONTAL_WHITESPACE.sub(" ", line).strip()
        lines.append(normalized)

    cleaned = "\n".join(lines)
    cleaned = _EXCESS_BLANK_LINES.sub("\n\n", cleaned)

    return cleaned.strip()

"""Compact display codes for registry category labels."""

from __future__ import annotations

import re


def compact_category_label(label: str | None) -> str:
    """Return a dot-separated 3-4 character code for a category label."""
    tokens = list(_category_words(label or ""))
    if not tokens:
        return ""
    return ".".join(_compact_word(token) for token in tokens)


def _category_words(label: str):
    for chunk in re.split(r"\s+", label.strip()):
        if not chunk:
            continue
        parts = re.findall(r"[A-Za-z0-9]+", chunk)
        if not parts:
            continue
        if "&" in chunk and len(parts) == 2 and all(len(part) == 1 for part in parts):
            yield f"{parts[0]}n{parts[1]}"
        elif len(parts) > 1 and any(len(part) < 3 for part in parts):
            yield "".join(parts)
        else:
            yield from parts


def _compact_word(word: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "", word).upper()
    overrides = {
        "DARK": "DRK",
        "DRIVER": "DRV",
        "HOUSE": "HOUS",
        "CHANT": "CHNT",
        "ORGANIC": "ORG",
        "WAREHOUSE": "WHSE",
    }
    if normalized in overrides:
        return overrides[normalized]
    if len(normalized) <= 4:
        return normalized
    return normalized[:4]

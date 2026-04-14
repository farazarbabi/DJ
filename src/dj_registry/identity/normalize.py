"""Text normalization for artist, title, and mix fields."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """General text normalization: unicode, case, whitespace."""
    if not text:
        return ""
    # Unicode NFC normalization
    text = unicodedata.normalize("NFC", text)
    text = text.lower().strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_artist(artist: str) -> str:
    """Normalize artist name for matching."""
    text = normalize_text(artist)
    if not text:
        return ""
    # Normalize featuring syntax
    text = re.sub(r"\b(featuring|feat\.?|ft\.?)\b", "feat", text)
    # Normalize separators
    text = re.sub(r"\s*[,&]\s*", " & ", text)
    text = re.sub(r"\s+and\s+", " & ", text)
    text = re.sub(r"\s+vs\.?\s+", " & ", text)
    # Collapse whitespace again
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_title(title: str) -> str:
    """Normalize track title for matching."""
    text = normalize_text(title)
    if not text:
        return ""
    # Remove featuring from title (already in artist)
    text = re.sub(r"\s*[\(\[](feat\.?|ft\.?|featuring)\s+[^\)\]]+[\)\]]", "", text)
    text = re.sub(r"\s+(feat\.?|ft\.?|featuring)\s+.*$", "", text)
    # Strip surrounding brackets/parens (mix info should be extracted separately)
    text = text.strip()
    return text


# Canonical mix forms
_MIX_CANONICAL: dict[str, str] = {
    "original mix": "original mix",
    "original": "original mix",
    "extended mix": "extended mix",
    "extended": "extended mix",
    "extended remix": "extended mix",
    "radio edit": "radio edit",
    "radio mix": "radio edit",
    "dub mix": "dub mix",
    "dub": "dub mix",
    "vip": "vip",
    "vip mix": "vip",
    "instrumental": "instrumental",
    "instrumental mix": "instrumental",
    "live": "live",
    "live version": "live",
}


def normalize_mix(mix: str) -> str:
    """Normalize mix/version name for matching."""
    text = normalize_text(mix)
    if not text:
        return ""

    # Strip surrounding parens/brackets
    text = re.sub(r"^[\(\[]+", "", text)
    text = re.sub(r"[\)\]]+$", "", text)
    text = text.strip()

    # Check canonical forms
    canonical = _MIX_CANONICAL.get(text)
    if canonical:
        return canonical

    return text


def extract_mix_from_title(title: str) -> tuple[str, str]:
    """Extract mix/version info from title.

    Returns (clean_title, mix_name).
    """
    # Match "(Something Mix/Remix/Edit)" at the end
    m = re.search(
        r"\s*[\(\[](.*?(?:mix|remix|edit|dub|version|vip|instrumental|live).*?)[\)\]]\s*$",
        title, re.IGNORECASE,
    )
    if m:
        mix = m.group(1).strip()
        clean = title[:m.start()].strip()
        return clean, normalize_mix(mix)

    return title, ""

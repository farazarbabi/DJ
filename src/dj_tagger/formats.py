"""Tag string formatting and parsing."""

from __future__ import annotations

import re

from .moods import mood_tag_pattern, normalize_mood_code
from .vocals import (
    normalize_vocal_profile,
    vocal_profile_from_has_vocals,
    vocal_profile_tag_pattern,
)

_MOOD_PATTERN = mood_tag_pattern()
_VOCAL_PATTERN = vocal_profile_tag_pattern()
_CATEGORY_PATTERN = r"[A-Za-z0-9]{3,4}(?:\.[A-Za-z0-9]{3,4})*"
_SEP = "|"
_SEP_RE = re.escape(_SEP)
_OPTIONAL_CATEGORY_AND_GROUP = (
    r"(?:" + _SEP_RE + r"((?!G\d{3}$)" + _CATEGORY_PATTERN + r"))?"
    r"(?:" + _SEP_RE + r"(G\d{3}))?"
)


def format_tag(
    energy: int | None,
    camelot: str | None,
    bpm: int | None = None,
    *,
    vibe: str | None = None,
    has_vocals: bool | None = None,
    group_id: str | None = None,
    vocal_profile: str | None = None,
    category: str | None = None,
) -> str:
    """Build the final comment tag string.

    Returns e.g. ``"9A|E3|HYPN|INST"`` or
    ``"9A|E3|HYPN|INST|DRK.TECH.HOUS.DRV"``.
    Order: KEY|ENERGY|VIBE|VOCAL[|CATEGORY][|GID]

    The ``bpm`` parameter is accepted but intentionally not emitted in the
    current tag format. The wiring is kept so it can be reintroduced later.
    """
    del bpm  # currently unused; kept in signature for forward compatibility
    vocal_code = _clean_vocab_code(normalize_vocal_profile(vocal_profile), _VOCAL_PATTERN)
    vocal_code = vocal_code or vocal_profile_from_has_vocals(has_vocals) or "??"
    vibe_code = _clean_vocab_code(normalize_mood_code(vibe), _MOOD_PATTERN) or "??"
    parts = [
        camelot or "??",
        f"E{energy}" if energy is not None else "E?",
        vibe_code,
        vocal_code,
    ]
    clean_category = _clean_category(category)
    if clean_category:
        parts.append(clean_category)
    if group_id:
        parts.append(group_id)
    return _SEP.join(parts)


# COMMENT format: KEY|ENERGY|VIBE|VOCAL[|CATEGORY][|GID]
_TAG_PATTERN = re.compile(
    r"^(\d{1,2}[AB]|\?\?)"
    + _SEP_RE +
    r"E([1-5?])"
    + _SEP_RE +
    r"(" + _MOOD_PATTERN + r")"
    + _SEP_RE +
    r"(" + _VOCAL_PATTERN + r")"
    + _OPTIONAL_CATEGORY_AND_GROUP +
    r"$"
)


def parse_tag(tag_string: str) -> dict[str, str] | None:
    """Parse a tag string back into components. Returns None if not a valid tag.
    """
    s = tag_string.strip()

    m = _TAG_PATTERN.match(s)
    if m:
        result = {
            "key": m.group(1),
            "energy": m.group(2),
            "vibe": normalize_mood_code(m.group(3)),
            "vocal": normalize_vocal_profile(m.group(4)),
        }
        _add_category_and_group(result, m.group(5), m.group(6))
        return result

    return None


def _clean_category(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9_.]+", "", value.strip())


def _clean_vocab_code(value: str | None, pattern: str) -> str:
    if not value:
        return ""
    return value if re.fullmatch(pattern, value) else ""


def _add_category_and_group(result: dict[str, str], category: str | None, group_id: str | None) -> None:
    if category and not group_id:
        match = re.match(r"^(.+)" + _SEP_RE + r"(G\d{3})$", category)
        if match:
            category = match.group(1)
            group_id = match.group(2)
    if category:
        result["category"] = category
    if group_id:
        result["group_id"] = group_id

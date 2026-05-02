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
_OPTIONAL_CATEGORY_AND_GROUP = (
    r"(?:_((?!G\d{3}$)" + _CATEGORY_PATTERN + r"))?"
    r"(?:_(G\d{3}))?"
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

    Returns e.g. ``"9A_126_E3_HYPN_INST"`` or
    ``"9A_126_E3_HYPN_INST_DRK.TECH.HOUS.DRV"``.
    Order: KEY_BPM_ENERGY_VIBE_VOCAL[_CATEGORY][_GID]
    """
    vocal_code = _clean_vocab_code(normalize_vocal_profile(vocal_profile), _VOCAL_PATTERN)
    vocal_code = vocal_code or vocal_profile_from_has_vocals(has_vocals) or "??"
    vibe_code = _clean_vocab_code(normalize_mood_code(vibe), _MOOD_PATTERN) or "??"
    parts = [
        camelot or "??",
        str(bpm) if bpm is not None else "???",
        f"E{energy}" if energy is not None else "E?",
        vibe_code,
        vocal_code,
    ]
    clean_category = _clean_category(category)
    if clean_category:
        parts.append(clean_category)
    if group_id:
        parts.append(group_id)
    return "_".join(parts)


# COMMENT format: KEY_BPM_ENERGY_VIBE_VOCAL[_CATEGORY][_GID]
_TAG_PATTERN = re.compile(
    r"^(\d{1,2}[AB]|\?\?)"
    r"_"
    r"(\d{2,3}|\?\?\?)"
    r"_"
    r"E([1-5?])"
    r"_"
    r"(" + _MOOD_PATTERN + r")"
    r"_"
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
            "bpm": m.group(2),
            "energy": m.group(3),
            "vibe": normalize_mood_code(m.group(4)),
            "vocal": normalize_vocal_profile(m.group(5)),
        }
        _add_category_and_group(result, m.group(6), m.group(7))
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
        match = re.match(r"^(.+)_(G\d{3})$", category)
        if match:
            category = match.group(1)
            group_id = match.group(2)
    if category:
        result["category"] = category
    if group_id:
        result["group_id"] = group_id

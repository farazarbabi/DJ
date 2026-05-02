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
_LEGACY_CATEGORY_PATTERN = r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*"
_OPTIONAL_CATEGORY_AND_GROUP = (
    r"(?:_((?!G\d{3}$)" + _CATEGORY_PATTERN + r"))?"
    r"(?:_(G\d{3}))?"
)
_OPTIONAL_LEGACY_CATEGORY_AND_GROUP = (
    r"(?:_((?!G\d{3}$)" + _LEGACY_CATEGORY_PATTERN + r"))?"
    r"(?:_(G\d{3}))?"
)


def format_tag(
    energy: int | None,
    camelot: str | None,
    bpm: int | None = None,
    structure: str | None = None,
    vibe: str | None = None,
    has_vocals: bool | None = None,
    group_id: str | None = None,
    *,
    vocal_profile: str | None = None,
    category: str | None = None,
    category_id: str | None = None,
) -> str:
    """Build the final comment tag string.

    Returns e.g. ``"9A_126_E3_HYPN_INST"`` or
    ``"9A_126_E3_HYPN_INST_DRK.TECH.HOUS.DRV"``.
    Order: KEY_BPM_ENERGY_MOOD_VOCAL[_CATEGORY][_GID]

    ``structure`` is accepted for compatibility with older callers, but is no
    longer emitted in the COMMENT tag.
    """
    del structure
    vocal_code = normalize_vocal_profile(vocal_profile) or vocal_profile_from_has_vocals(has_vocals) or "??"
    parts = [
        camelot or "??",
        str(bpm) if bpm is not None else "???",
        f"E{energy}" if energy is not None else "E?",
        normalize_mood_code(vibe) or "??",
        vocal_code,
    ]
    clean_category = _clean_category(category or category_id)
    if clean_category:
        parts.append(clean_category)
    if group_id:
        parts.append(group_id)
    return "_".join(parts)


# Current format: KEY_BPM_ENERGY_MOOD_VOC[_CATEGORY][_GID]
_TAG_PATTERN_V5 = re.compile(
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

# Legacy v4: KEY_ENERGY_MOOD_VOC_BPM[_CATEGORY][_GID]
_TAG_PATTERN_V4 = re.compile(
    r"^(\d{1,2}[AB]|\?\?)"
    r"_"
    r"E([1-5?])"
    r"_"
    r"(" + _MOOD_PATTERN + r")"
    r"_"
    r"(" + _VOCAL_PATTERN + r")"
    r"_"
    r"(\d{2,3}|\?\?\?)"
    + _OPTIONAL_LEGACY_CATEGORY_AND_GROUP +
    r"$"
)

# Legacy v3: KEY_ENERGY_MOOD_STRUCT_VOC_BPM[_CATEGORY][_GID]
_TAG_PATTERN_V3 = re.compile(
    r"^(\d{1,2}[AB]|\?\?)"
    r"_"
    r"E([1-5?])"
    r"_"
    r"(" + _MOOD_PATTERN + r")"
    r"_"
    r"(\d{2}[GHDBL]|\?\?)"
    r"_"
    r"(" + _VOCAL_PATTERN + r")"
    r"_"
    r"(\d{2,3}|\?\?\?)"
    + _OPTIONAL_LEGACY_CATEGORY_AND_GROUP +
    r"$"
)

# Legacy v2: E# | KEY | BPM | STRUCT | VIBE | VOC [| GID]
_TAG_PATTERN_V2 = re.compile(
    r"^E([1-5?])"
    r"\s*\|\s*"
    r"(\d{1,2}[AB]|\?\?)"
    r"\s*\|\s*"
    r"(\d{2,3}|\?\?\?)"
    r"\s*\|\s*"
    r"(\d{2}[GHDBL]|\?\?)"
    r"\s*\|\s*"
    r"(" + _MOOD_PATTERN + r")"
    r"\s*\|\s*"
    r"(" + _VOCAL_PATTERN + r")"
    r"(?:\s*\|\s*(G\d{3}))?$"
)

# Legacy v1: E# | KEY | STRUCT | VIBE | VOC
_TAG_PATTERN_V1 = re.compile(
    r"^E([1-5?])"
    r"\s*\|\s*"
    r"(\d{1,2}[AB]|\?\?)"
    r"\s*\|\s*"
    r"(\d{2}[GHDBL]|\?\?)"
    r"\s*\|\s*"
    r"(" + _MOOD_PATTERN + r")"
    r"\s*\|\s*"
    r"(" + _VOCAL_PATTERN + r")$"
)


def parse_tag(tag_string: str) -> dict[str, str] | None:
    """Parse a tag string back into components. Returns None if not a valid tag.

    Handles v5 (current), v4/v3/v2/v1 legacy formats.
    """
    s = tag_string.strip()

    m = _TAG_PATTERN_V5.match(s)
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

    m = _TAG_PATTERN_V4.match(s)
    if m:
        result = {
            "key": m.group(1),
            "energy": m.group(2),
            "vibe": normalize_mood_code(m.group(3)),
            "vocal": normalize_vocal_profile(m.group(4)),
            "bpm": m.group(5),
        }
        _add_category_and_group(result, m.group(6), m.group(7))
        return result

    m = _TAG_PATTERN_V3.match(s)
    if m:
        result = {
            "key": m.group(1),
            "energy": m.group(2),
            "vibe": normalize_mood_code(m.group(3)),
            "structure": m.group(4),
            "vocal": normalize_vocal_profile(m.group(5)),
            "bpm": m.group(6),
        }
        _add_category_and_group(result, m.group(7), m.group(8))
        return result

    m = _TAG_PATTERN_V2.match(s)
    if m:
        result = {
            "energy": m.group(1),
            "key": m.group(2),
            "bpm": m.group(3),
            "structure": m.group(4),
            "vibe": normalize_mood_code(m.group(5)),
            "vocal": normalize_vocal_profile(m.group(6)),
        }
        if m.group(7):
            result["group_id"] = m.group(7)
        return result

    m = _TAG_PATTERN_V1.match(s)
    if m:
        return {
            "energy": m.group(1),
            "key": m.group(2),
            "structure": m.group(3),
            "vibe": normalize_mood_code(m.group(4)),
            "vocal": normalize_vocal_profile(m.group(5)),
        }

    return None


def _clean_category(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9_.]+", "", value.strip())


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

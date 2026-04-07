"""Tag string formatting and parsing."""

from __future__ import annotations

import re


def format_tag(
    energy: int | None,
    camelot: str | None,
    bpm: int | None = None,
    structure: str | None = None,
    vibe: str | None = None,
    has_vocals: bool | None = None,
    group_id: str | None = None,
) -> str:
    """Build the final comment tag string.

    Returns e.g. ``"9A_E3_HYPN_64H_NV_126"`` or ``"9A_E3_HYPN_64H_NV_126_G001"``
    Order: KEY_ENERGY_VIBE_STRUCTURE_VOCAL_BPM[_GID]
    """
    parts = [
        camelot or "??",
        f"E{energy}" if energy is not None else "E?",
        vibe or "??",
        structure or "??",
        "V" if has_vocals else "NV" if has_vocals is not None else "??",
        str(bpm) if bpm is not None else "???",
    ]
    if group_id:
        parts.append(group_id)
    return "_".join(parts)


# Current format: KEY_ENERGY_VIBE_STRUCT_VOC_BPM[_GID]
_TAG_PATTERN_V3 = re.compile(
    r"^(\d{1,2}[AB]|\?\?)"
    r"_"
    r"E([1-5?])"
    r"_"
    r"(HYPN|DRK|RAW|DEEP|TRIB|MEL|ACID|ATM|\?\?)"
    r"_"
    r"(\d{2}[GHDBL]|\?\?)"
    r"_"
    r"(V|NV|\?\?)"
    r"_"
    r"(\d{2,3}|\?\?\?)"
    r"(?:_(G\d{3}))?$"
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
    r"(HYPN|DRK|RAW|DEEP|TRIB|MEL|ACID|ATM|\?\?)"
    r"\s*\|\s*"
    r"(V|NV|\?\?)"
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
    r"(HYPN|DRK|RAW|DEEP|TRIB|MEL|ACID|ATM|\?\?)"
    r"\s*\|\s*"
    r"(V|NV|\?\?)$"
)


def parse_tag(tag_string: str) -> dict[str, str] | None:
    """Parse a tag string back into components. Returns None if not a valid tag.

    Handles v3 (current), v2 (legacy with BPM), and v1 (legacy without BPM).
    """
    s = tag_string.strip()

    # Try v3 first (current: KEY_ENERGY_VIBE_STRUCT_VOC_BPM[_GID])
    m = _TAG_PATTERN_V3.match(s)
    if m:
        result = {
            "key": m.group(1),
            "energy": m.group(2),
            "vibe": m.group(3),
            "structure": m.group(4),
            "vocal": m.group(5),
            "bpm": m.group(6),
        }
        if m.group(7):
            result["group_id"] = m.group(7)
        return result

    # Try v2 (legacy: E# | KEY | BPM | STRUCT | VIBE | VOC [| GID])
    m = _TAG_PATTERN_V2.match(s)
    if m:
        result = {
            "energy": m.group(1),
            "key": m.group(2),
            "bpm": m.group(3),
            "structure": m.group(4),
            "vibe": m.group(5),
            "vocal": m.group(6),
        }
        if m.group(7):
            result["group_id"] = m.group(7)
        return result

    # Fall back to v1 (legacy, no BPM)
    m = _TAG_PATTERN_V1.match(s)
    if m:
        return {
            "energy": m.group(1),
            "key": m.group(2),
            "structure": m.group(3),
            "vibe": m.group(4),
            "vocal": m.group(5),
        }

    return None

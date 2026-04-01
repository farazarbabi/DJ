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

    Returns e.g. ``"E3 | 9A | 126 | 64H | HYPN | NV"``
    or with group: ``"E3 | 9A | 126 | 64H | HYPN | NV | G017"``
    """
    parts = [
        f"E{energy}" if energy is not None else "E?",
        camelot or "??",
        str(bpm) if bpm is not None else "???",
        structure or "??",
        vibe or "??",
        "V" if has_vocals else "NV" if has_vocals is not None else "??",
    ]
    if group_id is not None:
        parts.append(group_id)
    return " | ".join(parts)


# New format: E# | KEY | BPM | STRUCT | VIBE | VOC [| GID]
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

# Legacy format: E# | KEY | STRUCT | VIBE | VOC
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

    Handles both v1 (without BPM) and v2 (with BPM and optional GID) formats.
    """
    s = tag_string.strip()

    # Try v2 first (with BPM)
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

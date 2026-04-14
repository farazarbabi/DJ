"""Extended tag extraction — reads all standard tag fields including ISRC."""

from __future__ import annotations

import logging
from pathlib import Path

from ..key_utils import parse_any_key

logger = logging.getLogger(__name__)


def extract_tags(path: str) -> dict[str, str]:
    """Extract all readable standard tags from an audio file.

    Returns a dict with keys: title, artist, album, genre, bpm, key_raw,
    key_standard, key_camelot, comment, isrc.
    Empty string for missing fields.
    """
    ext = Path(path).suffix.lower()
    try:
        if ext == ".mp3":
            return _extract_mp3(path)
        if ext in (".aiff", ".aif"):
            return _extract_aiff(path)
    except Exception:
        logger.warning("Tag extraction failed for %s", path, exc_info=True)

    return _empty_tags()


def _empty_tags() -> dict[str, str]:
    return {
        "title": "", "artist": "", "album": "", "genre": "",
        "bpm": "", "key_raw": "", "key_standard": "", "key_camelot": "",
        "comment": "", "isrc": "",
    }


def _normalize_key_fields(tags: dict[str, str]) -> None:
    """Parse key_raw into standard + camelot if possible."""
    raw = tags.get("key_raw", "").strip()
    if not raw:
        return
    parsed = parse_any_key(raw)
    if parsed:
        tags["key_standard"], tags["key_camelot"] = parsed


def _first_text(tag_value) -> str:
    """Extract text from various mutagen tag value types."""
    if tag_value is None:
        return ""
    if isinstance(tag_value, list):
        return str(tag_value[0]) if tag_value else ""
    return str(tag_value)


# ---------------------------------------------------------------------------
# MP3 (ID3)
# ---------------------------------------------------------------------------

def _extract_mp3(path: str) -> dict[str, str]:
    from mutagen.mp3 import MP3

    tags = _empty_tags()
    audio = MP3(path)
    if audio.tags is None:
        return tags

    id3 = audio.tags
    tags["title"] = _first_text(id3.get("TIT2"))
    tags["artist"] = _first_text(id3.get("TPE1"))
    tags["album"] = _first_text(id3.get("TALB"))
    tags["genre"] = _first_text(id3.get("TCON"))
    tags["bpm"] = _first_text(id3.get("TBPM"))
    tags["key_raw"] = _first_text(id3.get("TKEY"))
    tags["isrc"] = _first_text(id3.get("TSRC"))

    # Comment: try generic first
    for k, v in id3.items():
        if k.startswith("COMM:"):
            tags["comment"] = str(v)
            break

    _normalize_key_fields(tags)
    return tags


# ---------------------------------------------------------------------------
# AIFF (ID3)
# ---------------------------------------------------------------------------

def _extract_aiff(path: str) -> dict[str, str]:
    from mutagen.aiff import AIFF

    tags = _empty_tags()
    audio = AIFF(path)
    if audio.tags is None:
        return tags

    id3 = audio.tags
    tags["title"] = _first_text(id3.get("TIT2"))
    tags["artist"] = _first_text(id3.get("TPE1"))
    tags["album"] = _first_text(id3.get("TALB"))
    tags["genre"] = _first_text(id3.get("TCON"))
    tags["bpm"] = _first_text(id3.get("TBPM"))
    tags["key_raw"] = _first_text(id3.get("TKEY"))
    tags["isrc"] = _first_text(id3.get("TSRC"))

    for k, v in id3.items():
        if k.startswith("COMM:"):
            tags["comment"] = str(v)
            break

    _normalize_key_fields(tags)
    return tags

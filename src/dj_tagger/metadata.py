"""Read and write comment tags across audio formats via mutagen."""

from __future__ import annotations

import logging
from pathlib import Path

from mutagen import MutagenError

from .constants import TAG_IDENTIFIER
from .formats import parse_tag

logger = logging.getLogger(__name__)


def read_existing_tag(path: str) -> str | None:
    """Read the existing dj_tagger tag from a file. Returns None if not found."""
    ext = Path(path).suffix.lower()
    try:
        if ext == ".mp3":
            return _read_mp3(path)
        if ext == ".flac":
            return _read_flac(path)
        if ext in (".aiff", ".aif"):
            return _read_aiff(path)
        if ext == ".wav":
            return _read_wav(path)
        if ext == ".m4a":
            return _read_m4a(path)
    except Exception:
        logger.debug("Could not read existing tag from %s", path, exc_info=True)
    return None


def write_tag(path: str, tag_string: str, *, dry_run: bool = False) -> None:
    """Write the tag string into the file's COMMENT metadata field."""
    if dry_run:
        return

    ext = Path(path).suffix.lower()
    try:
        if ext == ".mp3":
            _write_mp3(path, tag_string)
        elif ext == ".flac":
            _write_flac(path, tag_string)
        elif ext in (".aiff", ".aif"):
            _write_aiff(path, tag_string)
        elif ext == ".wav":
            _write_wav(path, tag_string)
        elif ext == ".m4a":
            _write_m4a(path, tag_string)
        else:
            raise ValueError(f"Unsupported format: {ext}")
    except MutagenError:
        logger.error("Failed to write tag to %s", path, exc_info=True)
        raise

    logger.debug("Wrote tag to %s: %s", path, tag_string)


# ---------------------------------------------------------------------------
# MP3 (ID3)
# ---------------------------------------------------------------------------

def _read_mp3(path: str) -> str | None:
    from mutagen.mp3 import MP3
    audio = MP3(path)
    if audio.tags is None:
        return None
    # Check for our tagged comment first
    key = f"COMM:{TAG_IDENTIFIER}:eng"
    if key in audio.tags:
        text = str(audio.tags[key])
        if parse_tag(text):
            return text
    # Fall back to generic comment
    for k, v in audio.tags.items():
        if k.startswith("COMM:"):
            text = str(v)
            if parse_tag(text):
                return text
    return None


def _write_mp3(path: str, tag_string: str) -> None:
    from mutagen.mp3 import MP3
    from mutagen.id3 import COMM, ID3NoHeaderError

    try:
        audio = MP3(path)
    except ID3NoHeaderError:
        audio = MP3(path)
        audio.add_tags()

    if audio.tags is None:
        audio.add_tags()

    # Write both a tagged comment (for our detection) and a generic comment
    # (for DJ software like Rekordbox/Traktor/Serato)
    audio.tags.add(
        COMM(encoding=0, lang="eng", desc=TAG_IDENTIFIER, text=tag_string)
    )
    audio.tags.add(
        COMM(encoding=0, lang="eng", desc="", text=tag_string)
    )
    audio.save()


# ---------------------------------------------------------------------------
# FLAC
# ---------------------------------------------------------------------------

def _read_flac(path: str) -> str | None:
    from mutagen.flac import FLAC
    audio = FLAC(path)
    tag_key = TAG_IDENTIFIER.lower()
    if tag_key in audio:
        text = audio[tag_key][0]
        if parse_tag(text):
            return text
    if "comment" in audio:
        text = audio["comment"][0]
        if parse_tag(text):
            return text
    return None


def _write_flac(path: str, tag_string: str) -> None:
    from mutagen.flac import FLAC
    audio = FLAC(path)
    audio["comment"] = [tag_string]
    audio[TAG_IDENTIFIER.lower()] = [tag_string]
    audio.save()


# ---------------------------------------------------------------------------
# AIFF (ID3 tags)
# ---------------------------------------------------------------------------

def _read_aiff(path: str) -> str | None:
    from mutagen.aiff import AIFF
    audio = AIFF(path)
    if audio.tags is None:
        return None
    key = f"COMM:{TAG_IDENTIFIER}:eng"
    if key in audio.tags:
        text = str(audio.tags[key])
        if parse_tag(text):
            return text
    for k, v in audio.tags.items():
        if k.startswith("COMM:"):
            text = str(v)
            if parse_tag(text):
                return text
    return None


def _write_aiff(path: str, tag_string: str) -> None:
    from mutagen.aiff import AIFF
    from mutagen.id3 import COMM

    audio = AIFF(path)
    if audio.tags is None:
        audio.add_tags()
    audio.tags.add(
        COMM(encoding=0, lang="eng", desc=TAG_IDENTIFIER, text=tag_string)
    )
    audio.tags.add(
        COMM(encoding=0, lang="eng", desc="", text=tag_string)
    )
    audio.save()


# ---------------------------------------------------------------------------
# WAV (ID3 tags)
# ---------------------------------------------------------------------------

def _read_wav(path: str) -> str | None:
    from mutagen.wave import WAVE
    audio = WAVE(path)
    if audio.tags is None:
        return None
    key = f"COMM:{TAG_IDENTIFIER}:eng"
    if key in audio.tags:
        text = str(audio.tags[key])
        if parse_tag(text):
            return text
    for k, v in audio.tags.items():
        if k.startswith("COMM:"):
            text = str(v)
            if parse_tag(text):
                return text
    return None


def _write_wav(path: str, tag_string: str) -> None:
    from mutagen.wave import WAVE
    from mutagen.id3 import COMM

    audio = WAVE(path)
    if audio.tags is None:
        audio.add_tags()
    audio.tags.add(
        COMM(encoding=0, lang="eng", desc=TAG_IDENTIFIER, text=tag_string)
    )
    audio.tags.add(
        COMM(encoding=0, lang="eng", desc="", text=tag_string)
    )
    audio.save()


# ---------------------------------------------------------------------------
# M4A / MP4
# ---------------------------------------------------------------------------

def _read_m4a(path: str) -> str | None:
    from mutagen.mp4 import MP4
    audio = MP4(path)
    if "\xa9cmt" in audio:
        text = audio["\xa9cmt"][0]
        if parse_tag(text):
            return text
    return None


def _write_m4a(path: str, tag_string: str) -> None:
    from mutagen.mp4 import MP4
    audio = MP4(path)
    audio["\xa9cmt"] = [tag_string]
    audio.save()


# ---------------------------------------------------------------------------
# Descriptive metadata (Title / Artist / Album / Genre / Year / Label)
# ---------------------------------------------------------------------------

def write_track_metadata(
    path: str,
    *,
    title: str = "",
    artist: str = "",
    album: str = "",
    album_artist: str = "",
    genre: str = "",
    year: str = "",
    label: str = "",
) -> None:
    """Write standard descriptive tags so DJ software shows real metadata.

    Only non-empty fields are written, so this never blanks out an existing
    value. Intended for freshly downloaded files whose only "metadata" is the
    filename. Key/BPM are deliberately not written here — those are owned by the
    registry's canonical resolution.

    Supports ID3-backed formats (.aiff/.aif, .wav, .mp3), plus .flac and .m4a.
    """
    ext = Path(path).suffix.lower()
    fields = {
        "title": title, "artist": artist, "album": album,
        "album_artist": album_artist, "genre": genre, "year": year, "label": label,
    }
    if not any(v for v in fields.values()):
        return

    try:
        if ext in (".aiff", ".aif", ".wav", ".mp3"):
            _write_meta_id3(path, ext, fields)
        elif ext == ".flac":
            _write_meta_flac(path, fields)
        elif ext == ".m4a":
            _write_meta_m4a(path, fields)
        else:
            raise ValueError(f"Unsupported format for metadata: {ext}")
    except MutagenError:
        logger.error("Failed to write metadata to %s", path, exc_info=True)
        raise


def read_descriptive_tags(path: str) -> dict[str, str]:
    """Read the descriptive tags this module writes, as a plain dict.

    Returns keys ``title/artist/album_artist/album/genre/year/label`` with the
    values found in the file (missing frames omitted). Used to log a Soundeo
    file's genuine release tags. Best-effort: any read error yields ``{}``.
    """
    ext = Path(path).suffix.lower()
    try:
        if ext in (".aiff", ".aif", ".wav", ".mp3"):
            return _read_meta_id3(path, ext)
        if ext == ".flac":
            from mutagen.flac import FLAC
            a = FLAC(path)
            vorbis = {"title": "title", "artist": "artist", "album": "album",
                      "album_artist": "albumartist", "genre": "genre",
                      "year": "date", "label": "label"}
            return {k: a[v][0] for k, v in vorbis.items() if a.get(v)}
        if ext == ".m4a":
            from mutagen.mp4 import MP4
            a = MP4(path)
            atom = {"title": "\xa9nam", "artist": "\xa9ART", "album": "\xa9alb",
                    "album_artist": "aART", "genre": "\xa9gen", "year": "\xa9day"}
            return {k: str(a[v][0]) for k, v in atom.items() if a.get(v)}
    except Exception:
        logger.debug("Could not read descriptive tags from %s", path, exc_info=True)
    return {}


def _read_meta_id3(path: str, ext: str) -> dict[str, str]:
    if ext == ".mp3":
        from mutagen.mp3 import MP3
        audio = MP3(path)
    elif ext in (".aiff", ".aif"):
        from mutagen.aiff import AIFF
        audio = AIFF(path)
    else:  # .wav
        from mutagen.wave import WAVE
        audio = WAVE(path)
    tags = audio.tags
    if tags is None:
        return {}
    frame_key = {"title": "TIT2", "artist": "TPE1", "album_artist": "TPE2",
                 "album": "TALB", "genre": "TCON", "year": "TDRC", "label": "TPUB"}
    out: dict[str, str] = {}
    for name, key in frame_key.items():
        frame = tags.get(key)
        if frame is not None:
            text = "".join(str(t) for t in getattr(frame, "text", [])).strip()
            if text:
                out[name] = text
    return out


def _write_meta_id3(path: str, ext: str, fields: dict[str, str]) -> None:
    from mutagen.id3 import TALB, TCON, TDRC, TIT2, TPE1, TPE2, TPUB

    if ext == ".mp3":
        from mutagen.mp3 import MP3
        audio = MP3(path)
    elif ext in (".aiff", ".aif"):
        from mutagen.aiff import AIFF
        audio = AIFF(path)
    else:  # .wav
        from mutagen.wave import WAVE
        audio = WAVE(path)

    if audio.tags is None:
        audio.add_tags()
    tags = audio.tags

    frame_for = {
        "title": (TIT2, fields["title"]),
        "artist": (TPE1, fields["artist"]),
        "album_artist": (TPE2, fields["album_artist"]),
        "album": (TALB, fields["album"]),
        "genre": (TCON, fields["genre"]),
        "year": (TDRC, fields["year"]),
        "label": (TPUB, fields["label"]),
    }
    for frame_cls, value in frame_for.values():
        if value:
            tags.setall(frame_cls.__name__, [frame_cls(encoding=3, text=[value])])
    audio.save()


def _write_meta_flac(path: str, fields: dict[str, str]) -> None:
    from mutagen.flac import FLAC
    audio = FLAC(path)
    vorbis = {
        "title": "title", "artist": "artist", "album": "album",
        "album_artist": "albumartist", "genre": "genre", "year": "date",
        "label": "label",
    }
    for key, value in fields.items():
        if value:
            audio[vorbis[key]] = [value]
    audio.save()


def _write_meta_m4a(path: str, fields: dict[str, str]) -> None:
    from mutagen.mp4 import MP4
    audio = MP4(path)
    atom = {
        "title": "\xa9nam", "artist": "\xa9ART", "album": "\xa9alb",
        "album_artist": "aART", "genre": "\xa9gen", "year": "\xa9day",
    }
    for key, value in fields.items():
        if value and key in atom:
            audio[atom[key]] = [value]
    audio.save()

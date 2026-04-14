"""Write canonical key back to file tags (KEY field + COMMENT key portion)."""

from __future__ import annotations

import logging
from pathlib import Path

from ..key_utils import camelot_to_standard
from ..models import now_iso
from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)


def _write_tkey(path: str, camelot: str) -> None:
    """Write Camelot key to the standard TKEY/InitialKey tag field."""
    ext = Path(path).suffix.lower()

    if ext == ".mp3":
        from mutagen.mp3 import MP3
        from mutagen.id3 import TKEY

        audio = MP3(path)
        if audio.tags is None:
            audio.add_tags()
        audio.tags.add(TKEY(encoding=0, text=[camelot]))
        audio.save()

    elif ext in (".aiff", ".aif"):
        from mutagen.aiff import AIFF
        from mutagen.id3 import TKEY

        audio = AIFF(path)
        if audio.tags is None:
            audio.add_tags()
        audio.tags.add(TKEY(encoding=0, text=[camelot]))
        audio.save()


def _update_comment_key(path: str, new_camelot: str) -> bool:
    """Update the key portion of an existing dj-tagger COMMENT tag.

    Returns True if the COMMENT was updated, False if no existing tag found.
    """
    from dj_tagger.metadata import read_existing_tag, write_tag
    from dj_tagger.formats import parse_tag, format_tag

    existing = read_existing_tag(path)
    if not existing:
        return False

    parsed = parse_tag(existing)
    if not parsed:
        return False

    # Reconstruct tag with new key
    energy_str = parsed.get("energy", "?")
    energy = int(energy_str) if energy_str.isdigit() else None

    bpm_str = parsed.get("bpm", "???")
    bpm = int(bpm_str) if bpm_str.isdigit() else None

    vibe = parsed.get("vibe")
    if vibe == "??":
        vibe = None

    structure = parsed.get("structure")
    if structure == "??":
        structure = None

    vocal_str = parsed.get("vocal", "??")
    if vocal_str == "V":
        has_vocals = True
    elif vocal_str == "NV":
        has_vocals = False
    else:
        has_vocals = None

    group_id = parsed.get("group_id")

    new_tag = format_tag(
        energy=energy,
        camelot=new_camelot,
        bpm=bpm,
        structure=structure,
        vibe=vibe,
        has_vocals=has_vocals,
        group_id=group_id,
    )

    write_tag(path, new_tag, dry_run=False)
    return True


def sync_tags(
    store: CsvStore,
    *,
    dry_run: bool = True,
    only_changed: bool = True,
) -> tuple[int, int, int]:
    """Write canonical key to file tags.

    Returns (written, skipped, errors).
    """
    tracks = store.load_tracks()
    files = store.load_files()

    track_by_id = {t.track_id: t for t in tracks}

    written = 0
    skipped = 0
    errors = 0

    for frec in files:
        track = track_by_id.get(frec.track_id)
        if not track or not track.canonical_key_camelot:
            skipped += 1
            continue

        canonical_cam = track.canonical_key_camelot

        # Check if key already matches
        if only_changed and frec.embedded_key_camelot == canonical_cam:
            skipped += 1
            continue

        if dry_run:
            logger.debug(
                "Would write key %s to %s (was: %s)",
                canonical_cam, frec.file_name,
                frec.embedded_key_camelot or "none",
            )
            skipped += 1
            continue

        try:
            # Write standard KEY tag field
            _write_tkey(frec.path_abs, canonical_cam)

            # Update key in COMMENT tag (if exists)
            _update_comment_key(frec.path_abs, canonical_cam)

            frec.embedded_key_camelot = canonical_cam
            std = camelot_to_standard(canonical_cam)
            if std:
                frec.embedded_key_standard = std
            frec.tag_write_status = "ok"
            frec.tag_write_error = ""
            frec.last_tag_written_at = now_iso()
            written += 1

            logger.debug("Wrote key %s to %s", canonical_cam, frec.file_name)

        except Exception as e:
            frec.tag_write_status = "error"
            frec.tag_write_error = str(e)
            errors += 1
            logger.error("Failed to write key to %s: %s", frec.file_name, e)

    store.save_files(files)
    if written or errors:
        logger.info("Tags: %d written, %d errors", written, errors)
    else:
        logger.debug("Tags: nothing to write")
    return written, skipped, errors

"""Write canonical key/BPM and tagger features back to file tags."""

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


def _write_full_tag(path: str, track) -> bool:
    """Write a full dj-tagger COMMENT tag from LogicalTrack canonical + tagger fields.

    Returns True if written, False if insufficient data.
    """
    from dj_tagger.formats import format_tag
    from dj_tagger.metadata import write_tag

    # Need at least key or some tagger data to write a meaningful tag
    camelot = track.canonical_key_camelot
    if not camelot and not track.tagger_energy:
        return False

    # Parse energy
    energy = None
    if track.tagger_energy:
        try:
            energy = int(track.tagger_energy)
        except (ValueError, TypeError):
            pass

    # Parse BPM — prefer canonical (multi-source resolved), fall back to tagger
    bpm = None
    bpm_str = track.canonical_bpm or track.tagger_bpm
    if bpm_str:
        try:
            bpm = int(round(float(bpm_str)))
        except (ValueError, TypeError):
            pass

    # Key — use canonical (multi-source resolved)
    if not camelot:
        camelot = None

    # Vocal
    has_vocals = None
    if track.tagger_vocal == "V":
        has_vocals = True
    elif track.tagger_vocal == "NV":
        has_vocals = False

    tag_string = format_tag(
        energy=energy,
        camelot=camelot,
        bpm=bpm,
        structure=track.tagger_structure or None,
        vibe=track.tagger_vibe or None,
        has_vocals=has_vocals,
    )

    write_tag(path, tag_string, dry_run=False)
    return True


def sync_tags(
    store: CsvStore,
    *,
    dry_run: bool = True,
    only_changed: bool = True,
) -> tuple[int, int, int]:
    """Write canonical key/BPM and tagger features to file tags.

    Writes:
    1. TKEY tag — canonical Camelot key (for DJ software)
    2. COMMENT tag — full tag string (KEY_ENERGY_VIBE_STRUCT_VOC_BPM)

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
        if not track:
            skipped += 1
            continue

        # Need at least a canonical key or tagger features to write
        has_key = bool(track.canonical_key_camelot)
        has_features = bool(track.tagger_energy)
        if not has_key and not has_features:
            skipped += 1
            continue

        if dry_run:
            canonical_cam = track.canonical_key_camelot or "??"
            logger.debug(
                "Would write tag to %s (key=%s, energy=%s, vibe=%s)",
                frec.file_name, canonical_cam,
                track.tagger_energy, track.tagger_vibe,
            )
            skipped += 1
            continue

        try:
            # Write standard KEY tag field (TKEY)
            if track.canonical_key_camelot:
                _write_tkey(frec.path_abs, track.canonical_key_camelot)

            # Write full COMMENT tag
            _write_full_tag(frec.path_abs, track)

            # Update file record
            if track.canonical_key_camelot:
                frec.embedded_key_camelot = track.canonical_key_camelot
                std = camelot_to_standard(track.canonical_key_camelot)
                if std:
                    frec.embedded_key_standard = std
            frec.tag_write_status = "ok"
            frec.tag_write_error = ""
            frec.last_tag_written_at = now_iso()
            written += 1

            logger.debug("Wrote tag to %s", frec.file_name)

        except Exception as e:
            frec.tag_write_status = "error"
            frec.tag_write_error = str(e)
            errors += 1
            logger.error("Failed to write tag to %s: %s", frec.file_name, e)

    store.save_files(files)
    if written or errors:
        logger.info("Tags: %d written, %d errors", written, errors)
    else:
        logger.debug("Tags: nothing to write")
    return written, skipped, errors

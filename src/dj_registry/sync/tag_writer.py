"""Write canonical key/BPM and tagger features back to file tags."""

from __future__ import annotations

import logging
from pathlib import Path

from ..category_codes import compact_category_label
from ..key_utils import camelot_to_standard
from ..models import now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)


def _parse_energy(value: str | int | None) -> int | None:
    if value is None:
        return None
    raw = str(value).strip().upper()
    if raw.startswith("E"):
        raw = raw[1:]
    if not raw:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def _category_label_for_tag(track) -> str:
    """Use the internal DJ-taxonomy model label for COMMENT tags when available."""
    return (
        str(getattr(track, "dj_taxonomy_internal_label", "") or "").strip()
        or str(getattr(track, "dj_taxonomy_label", "") or "").strip()
    )


def _category_code_for_tag(track) -> str:
    label = _category_label_for_tag(track)
    if label:
        return compact_category_label(label)
    return ""


def _build_comment_tag(track) -> str | None:
    """Build the COMMENT tag from registry fields, including DJ category when present."""
    from dj_tagger.formats import format_tag

    camelot = track.canonical_key_camelot or None
    category = _category_code_for_tag(track) or None
    has_tagger_fields = any(
        bool(getattr(track, field, ""))
        for field in ("tagger_energy", "tagger_vibe", "tagger_vocal", "tagger_bpm")
    )
    if not camelot and not has_tagger_fields and not category:
        return None

    bpm = None
    bpm_str = track.canonical_bpm or track.tagger_bpm
    if bpm_str:
        try:
            bpm = int(round(float(bpm_str)))
        except (ValueError, TypeError):
            pass

    return format_tag(
        energy=_parse_energy(track.tagger_energy),
        camelot=camelot,
        bpm=bpm,
        vibe=track.tagger_vibe or None,
        vocal_profile=track.tagger_vocal or None,
        category=category,
    )


def _write_tkey(path: str, camelot: str) -> None:
    """Write Camelot key to the standard TKEY/InitialKey tag field."""
    ext = Path(path).suffix.lower()

    if ext == ".mp3":
        from mutagen.id3 import TKEY
        from mutagen.mp3 import MP3

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
    """Write a full dj-tagger COMMENT tag from LogicalTrack fields."""
    from dj_tagger.metadata import write_tag

    tag_string = _build_comment_tag(track)
    if not tag_string:
        return False

    write_tag(path, tag_string, dry_run=False)
    return True


def sync_tags(
    store: CsvStore,
    *,
    dry_run: bool = True,
    only_changed: bool = True,
    write_key_tag: bool = False,
    show_progress: bool = False,
) -> tuple[int, int, int]:
    """Write tagger features to file tags.

    Always writes:
      COMMENT tag - KEY_BPM_ENERGY_VIBE_VOCAL[_CATEGORY]

    Optionally writes (write_key_tag=True):
      TKEY/InitialKey - canonical Camelot key, for DJ software display.
      Off by default so the original embedded key is preserved as a
      validation signal for multi-source key resolution.

    Returns (written, skipped, errors).
    """
    tracks = store.load_tracks()
    files = store.load_files()

    track_by_id = {t.track_id: t for t in tracks}

    written = 0
    skipped = 0
    errors = 0
    progress = ProgressBar(len(files), label="Sync tags", enabled=show_progress)

    for index, frec in enumerate(files, start=1):
        track = track_by_id.get(frec.track_id)
        if not track:
            skipped += 1
            progress.update(index, frec.file_name, written=written, skipped=skipped, errors=errors)
            continue

        candidate_tag = _build_comment_tag(track)
        if not candidate_tag:
            skipped += 1
            progress.update(index, frec.file_name, written=written, skipped=skipped, errors=errors)
            continue

        if dry_run:
            logger.debug("Would write tag to %s: %s", frec.file_name, candidate_tag)
            skipped += 1
            progress.update(index, frec.file_name, written=written, skipped=skipped, errors=errors)
            continue

        try:
            if write_key_tag and track.canonical_key_camelot:
                _write_tkey(frec.path_abs, track.canonical_key_camelot)
                frec.embedded_key_camelot = track.canonical_key_camelot
                std = camelot_to_standard(track.canonical_key_camelot)
                if std:
                    frec.embedded_key_standard = std

            _write_full_tag(frec.path_abs, track)

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
        progress.update(index, frec.file_name, written=written, skipped=skipped, errors=errors)

    progress.finish()
    store.save_files(files)
    if written or errors:
        logger.info("Tags: %d written, %d errors", written, errors)
    else:
        logger.debug("Tags: nothing to write")
    return written, skipped, errors

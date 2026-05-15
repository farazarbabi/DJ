"""Write canonical key/BPM and tagger features back to file tags."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from ..category_codes import compact_category_label
from ..key_utils import camelot_to_standard
from ..models import SourceObservation, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)


def load_group_ids_by_file(groups_csv: str) -> dict[str, str]:
    """Load filename -> group ID from a grouper groups.csv file."""
    if not groups_csv or not Path(groups_csv).exists():
        return {}
    mapping: dict[str, str] = {}
    with open(groups_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            file_name = (row.get("file") or "").strip()
            group_id = (row.get("group_id") or "").strip()
            if file_name and group_id:
                mapping[file_name] = group_id
    return mapping


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


def _build_comment_tag(track, *, group_id: str | None = None) -> str | None:
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
        group_id=group_id,
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


def _write_full_tag(path: str, track, tag_string: str | None = None) -> bool:
    """Write a full dj-tagger COMMENT tag from LogicalTrack fields."""
    from dj_tagger.metadata import write_tag

    tag_string = tag_string or _build_comment_tag(track)
    if not tag_string:
        return False

    write_tag(path, tag_string, dry_run=False)
    return True


def _tag_observation_from_write(frec, track, tag_string: str) -> SourceObservation:
    """Represent a just-written COMMENT as the current raw tag observation."""
    return SourceObservation(
        observation_id=f"OBS-tag-{frec.file_id}",
        track_id=frec.track_id,
        file_id=frec.file_id,
        artist=frec.embedded_artist or getattr(track, "artist_canonical", ""),
        title=frec.embedded_title or getattr(track, "title_canonical", ""),
        source_system="tag",
        source_object_id=frec.embedded_isrc or getattr(track, "isrc_canonical", ""),
        key_standard=frec.embedded_key_standard,
        key_camelot=frec.embedded_key_camelot,
        key_confidence=1.0 if frec.embedded_key_camelot else 0.0,
        bpm=frec.embedded_bpm,
        genre=frec.embedded_genre,
        comments=tag_string,
        observed_at=now_iso(),
    )


def sync_tags(
    store: CsvStore,
    *,
    dry_run: bool = True,
    only_changed: bool = True,
    write_key_tag: bool = False,
    group_ids_by_file: dict[str, str] | None = None,
    show_progress: bool = False,
) -> tuple[int, int, int]:
    """Write tagger features to file tags.

    Always writes:
      COMMENT tag - KEY|ENERGY|VIBE|VOCAL|CATEGORY when category is available.

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
    missing = 0
    errors = 0
    obs_cache = None
    obs_cache_dirty = False
    progress = ProgressBar(len(files), label="Sync tags", enabled=show_progress)

    for index, frec in enumerate(files, start=1):
        track = track_by_id.get(frec.track_id)
        if not track:
            skipped += 1
            progress.update(index, frec.file_name, written=written, skipped=skipped, errors=errors)
            continue

        if not os.path.exists(frec.path_abs):
            missing += 1
            skipped += 1
            logger.debug("Skipping missing file: %s", frec.path_abs)
            progress.update(index, frec.file_name, written=written, skipped=skipped, missing=missing, errors=errors)
            continue

        group_id = _group_id_for_file(frec, group_ids_by_file)
        candidate_tag = _build_comment_tag(track, group_id=group_id)
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

            _write_full_tag(frec.path_abs, track, candidate_tag)

            frec.embedded_comment = candidate_tag
            frec.tag_write_status = "ok"
            frec.tag_write_error = ""
            frec.last_tag_written_at = now_iso()
            if obs_cache is None:
                from ..store.obs_cache import ObsCache

                obs_cache = ObsCache()
            obs_cache.put_by_file(
                frec.path_abs,
                frec.audio_duration_sec,
                "tag",
                _tag_observation_from_write(frec, track, candidate_tag),
            )
            obs_cache_dirty = True
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
    if obs_cache is not None and obs_cache_dirty:
        obs_cache.save()
    if written or errors or missing:
        if missing:
            logger.info("Tags: %d written, %d errors, %d missing files skipped", written, errors, missing)
        else:
            logger.info("Tags: %d written, %d errors", written, errors)
    else:
        logger.debug("Tags: nothing to write")
    return written, skipped, errors


def _group_id_for_file(frec, group_ids_by_file: dict[str, str] | None) -> str | None:
    if not group_ids_by_file:
        return None
    for key in (frec.path_abs, frec.file_name, Path(frec.path_abs).name):
        group_id = group_ids_by_file.get(key)
        if group_id:
            return group_id
    return None

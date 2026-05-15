"""File scanner — discovers audio files and extracts metadata."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import mutagen
import soundfile as sf

from ..config import RegistryConfig
from ..models import FileRecord, SourceObservation, PayloadIndexEntry, now_iso
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from ..store.obs_cache import ObsCache
from .tag_extractor import extract_tags

logger = logging.getLogger(__name__)


def _sha256(path: str) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _audio_info(path: str) -> dict:
    """Extract audio technical metadata via soundfile."""
    try:
        info = sf.info(path)
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
        }
    except Exception:
        # Fallback via mutagen
        try:
            m = mutagen.File(path)
            if m and m.info:
                return {
                    "duration": getattr(m.info, "length", 0.0),
                    "sample_rate": getattr(m.info, "sample_rate", 0),
                    "channels": getattr(m.info, "channels", 0),
                }
        except Exception:
            pass
    return {"duration": 0.0, "sample_rate": 0, "channels": 0}


def _bitrate(path: str) -> int:
    """Extract bitrate via mutagen."""
    try:
        m = mutagen.File(path)
        if m and m.info:
            return getattr(m.info, "bitrate", 0)
    except Exception:
        pass
    return 0


def find_audio_files(
    roots: list[str],
    extensions: list[str],
    recursive: bool = True,
) -> list[Path]:
    """Discover audio files matching supported extensions."""
    results: list[Path] = []
    ext_set = {e.lower() for e in extensions}
    for root in roots:
        p = Path(root)
        if p.is_file() and p.suffix.lower() in ext_set:
            results.append(p)
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(p.glob(pattern)):
                if child.is_file() and child.suffix.lower() in ext_set:
                    # Exclude outputs directory
                    try:
                        rel = child.relative_to(p)
                        if any(part.lower() == "outputs" for part in rel.parts):
                            continue
                    except ValueError:
                        pass
                    results.append(child)
    return results


def scan_files(
    config: RegistryConfig,
    store: CsvStore,
    obs_cache: ObsCache | None = None,
    *,
    show_progress: bool = False,
) -> list[FileRecord]:
    """Scan library, create/update FileRecords, extract tags as observations.

    Returns list of all FileRecords (new + existing).
    """
    paths = find_audio_files(
        config.library_roots, config.supported_extensions
    )
    logger.debug("Found %d audio files", len(paths))

    existing_files = {f.path_abs: f for f in store.load_files()}
    store.load_observations()
    store.load_payload_index()

    new_files: list[FileRecord] = []
    new_obs: list[SourceObservation] = []
    new_payloads: list[PayloadIndexEntry] = []
    updated = 0
    skipped = 0
    visited: set[str] = set()
    progress = ProgressBar(len(paths), label="Scan files", enabled=show_progress)

    for index, path in enumerate(paths, start=1):
        path_abs = str(path.resolve())
        visited.add(path_abs)
        stat = path.stat()
        mtime_str = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")

        existing = existing_files.get(path_abs)
        if existing and existing.mtime_utc == mtime_str:
            # File unchanged — still emit tag observation from cache or existing data
            duration = existing.audio_duration_sec
            cached_tag = obs_cache.get_by_file(path_abs, duration, "tag") if obs_cache else None
            if cached_tag:
                cached_tag.observation_id = f"OBS-tag-{existing.file_id}"
                cached_tag.file_id = existing.file_id
                new_obs.append(cached_tag)
            elif existing.embedded_key_camelot or existing.embedded_bpm or existing.embedded_genre:
                new_obs.append(SourceObservation(
                    observation_id=f"OBS-tag-{existing.file_id}",
                    file_id=existing.file_id,
                    artist=existing.embedded_artist,
                    title=existing.embedded_title,
                    source_system="tag",
                    source_object_id=existing.embedded_isrc,
                    key_standard=existing.embedded_key_standard,
                    key_camelot=existing.embedded_key_camelot,
                    key_confidence=1.0 if existing.embedded_key_camelot else 0.0,
                    bpm=existing.embedded_bpm,
                    genre=existing.embedded_genre,
                    comments=existing.embedded_comment,
                    observed_at=existing.last_scanned_at,
                ))
            skipped += 1
            progress.update(index, path.name, new=len(new_files), updated=updated, skipped=skipped)
            continue

        # Compute file metadata
        file_hash = _sha256(path_abs)
        audio = _audio_info(path_abs)
        br = _bitrate(path_abs)

        file_id = existing.file_id if existing else f"F{len(existing_files) + len(new_files) + 1:05d}"

        cached_tag = obs_cache.get_by_file(path_abs, audio["duration"], "tag") if obs_cache else None
        tags = _tags_from_cached_observation(cached_tag, existing) if cached_tag else extract_tags(path_abs)

        rec = FileRecord(
            file_id=file_id,
            path_abs=path_abs,
            path_rel=str(path),
            file_name=path.name,
            extension=path.suffix.lower(),
            size_bytes=stat.st_size,
            mtime_utc=mtime_str,
            sha256=file_hash,
            audio_duration_sec=audio["duration"],
            sample_rate=audio["sample_rate"],
            bitrate=br,
            channels=audio["channels"],
            embedded_title=tags["title"],
            embedded_artist=tags["artist"],
            embedded_album=tags["album"],
            embedded_genre=tags["genre"],
            embedded_bpm=tags["bpm"],
            embedded_key_standard=tags["key_standard"],
            embedded_key_camelot=tags["key_camelot"],
            embedded_comment=tags["comment"],
            embedded_isrc=tags["isrc"],
            tag_read_status="ok",
            last_scanned_at=now_iso(),
        )

        if existing:
            rec.track_id = existing.track_id
            rec.is_primary_file = existing.is_primary_file
            rec.match_method = existing.match_method
            rec.match_score = existing.match_score
            existing_files[path_abs] = rec
            updated += 1
        else:
            new_files.append(rec)

        # Create tag observation — check cache first
        duration = audio["duration"]
        if cached_tag:
            cached_tag.observation_id = f"OBS-tag-{file_id}"
            cached_tag.file_id = file_id
            new_obs.append(cached_tag)
        elif tags["key_camelot"] or tags["bpm"] or tags["genre"]:
            obs = SourceObservation(
                observation_id=f"OBS-tag-{file_id}",
                file_id=file_id,
                artist=tags["artist"],
                title=tags["title"],
                source_system="tag",
                source_object_id=tags["isrc"],
                key_standard=tags["key_standard"],
                key_camelot=tags["key_camelot"],
                key_confidence=1.0 if tags["key_camelot"] else 0.0,
                bpm=tags["bpm"],
                genre=tags.get("genre", ""),
                comments=tags["comment"],
                observed_at=now_iso(),
            )
            new_obs.append(obs)
            if obs_cache:
                obs_cache.put_by_file(path_abs, duration, "tag", obs)

        # Save tag snapshot
        raw_dir = os.path.join(config.raw_dir, "file_tag_snapshots")
        os.makedirs(raw_dir, exist_ok=True)
        snapshot_path = os.path.join(raw_dir, f"{file_id}.json")
        payload_ref = f"tag-snapshot-{file_id}"
        rec.file_tag_payload_ref = payload_ref

        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(tags, f, indent=2)

        new_payloads.append(PayloadIndexEntry(
            payload_ref=payload_ref,
            source_system="tag",
            source_type="json",
            file_id=file_id,
            payload_path=os.path.relpath(snapshot_path, config.output_dir),
            fetched_at=now_iso(),
        ))
        progress.update(index, path.name, new=len(new_files), updated=updated, skipped=skipped)

    progress.finish()

    # Prune FileRecords whose paths are no longer on disk. We only prune entries
    # not visited by this scan AND confirmed missing — entries outside the
    # current library_roots that still exist on disk are left alone.
    removed_file_ids: set[str] = set()
    for path, frec in list(existing_files.items()):
        if path in visited:
            continue
        if os.path.exists(path):
            continue
        removed_file_ids.add(frec.file_id)
        del existing_files[path]

    all_files = list(existing_files.values()) + new_files

    store.save_files(all_files)
    if new_obs:
        store.add_observations(new_obs)
    if new_payloads:
        all_payload = store.load_payload_index()
        all_payload.extend(new_payloads)
        store.save_payload_index(all_payload)

    if removed_file_ids:
        remaining_obs = [o for o in store.load_observations() if o.file_id not in removed_file_ids]
        store.save_observations(remaining_obs)
        remaining_payloads = [p for p in store.load_payload_index() if p.file_id not in removed_file_ids]
        store.save_payload_index(remaining_payloads)

    if removed_file_ids:
        logger.info(
            "Scan: %d files (%d new, %d updated, %d unchanged, %d removed)",
            len(all_files), len(new_files), updated, skipped, len(removed_file_ids),
        )
    else:
        logger.info("Scan: %d files (%d new, %d updated, %d unchanged)", len(all_files), len(new_files), updated, skipped)
    return all_files


def _tags_from_cached_observation(obs: SourceObservation, existing: FileRecord | None) -> dict[str, str]:
    """Rebuild a tag snapshot from cached raw tag data."""
    return {
        "title": obs.title or (existing.embedded_title if existing else ""),
        "artist": obs.artist or (existing.embedded_artist if existing else ""),
        "album": existing.embedded_album if existing else "",
        "genre": obs.genre or (existing.embedded_genre if existing else ""),
        "bpm": obs.bpm or (existing.embedded_bpm if existing else ""),
        "key_raw": existing.embedded_key_camelot if existing else "",
        "key_standard": obs.key_standard or (existing.embedded_key_standard if existing else ""),
        "key_camelot": obs.key_camelot or (existing.embedded_key_camelot if existing else ""),
        "comment": obs.comments or (existing.embedded_comment if existing else ""),
        "isrc": obs.source_object_id or (existing.embedded_isrc if existing else ""),
    }

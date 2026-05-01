"""Identity resolution — map files and external records to logical tracks."""

from __future__ import annotations

import logging
import uuid

from ..config import RegistryConfig
from ..models import LogicalTrack
from ..progress import ProgressBar
from ..store.csv_store import CsvStore
from .normalize import normalize_artist, normalize_title, normalize_mix, extract_mix_from_title

logger = logging.getLogger(__name__)


def _duration_close(a: float, b: float, tolerance: float) -> bool:
    """Check if two durations are within tolerance."""
    if a <= 0 or b <= 0:
        return True  # Can't compare, don't penalize
    return abs(a - b) <= tolerance


def _make_track_id() -> str:
    return f"T-{uuid.uuid4().hex[:12]}"


def _identity_key(artist: str, title: str, mix: str) -> str:
    """Create a normalized identity key for exact matching."""
    return f"{artist}|||{title}|||{mix}"


def link_files_to_tracks(
    config: RegistryConfig,
    store: CsvStore,
    *,
    show_progress: bool = False,
) -> None:
    """Resolve file-to-track identity mappings.

    Creates/updates LogicalTracks in tracks_master from FileRecords in files_master.
    Uses a cascade: SHA256 -> exact normalized -> title+duration.
    """
    files = store.load_files()
    tracks = store.load_tracks()

    # Build indexes
    track_by_id: dict[str, LogicalTrack] = {t.track_id: t for t in tracks}
    tracks_by_identity: dict[str, LogicalTrack] = {}
    tracks_by_sha: dict[str, LogicalTrack] = {}

    for t in tracks:
        ik = _identity_key(
            normalize_artist(t.artist_canonical),
            normalize_title(t.title_canonical),
            normalize_mix(t.mix_canonical),
        )
        if ik != "||||||":
            tracks_by_identity[ik] = t

    # Index existing file->track SHA links
    for f in files:
        if f.track_id and f.sha256:
            track = track_by_id.get(f.track_id)
            if track:
                tracks_by_sha[f.sha256] = track

    unlinked = [f for f in files if not f.track_id]
    linked_count = 0
    new_track_count = 0
    progress = ProgressBar(len(unlinked), label="Link files", enabled=show_progress)

    for index, frec in enumerate(unlinked, start=1):
        matched_track: LogicalTrack | None = None
        method = ""
        score = 0.0

        # Step 1: SHA256 match (exact file duplicate)
        if frec.sha256 and frec.sha256 in tracks_by_sha:
            matched_track = tracks_by_sha[frec.sha256]
            method = "sha256"
            score = 1.0

        # Step 2: Exact normalized match (artist + title + mix + duration)
        if not matched_track:
            artist_n = normalize_artist(frec.embedded_artist)
            title_raw = frec.embedded_title or frec.file_name
            clean_title, mix_from_title = extract_mix_from_title(title_raw)
            title_n = normalize_title(clean_title)
            mix_n = normalize_mix(mix_from_title)

            ik = _identity_key(artist_n, title_n, mix_n)
            candidate = tracks_by_identity.get(ik)
            if candidate and _duration_close(
                frec.audio_duration_sec, candidate.duration_sec_canonical,
                config.duration_tolerance_strong,
            ):
                matched_track = candidate
                method = "exact_normalized"
                score = 0.95

        # Step 3: Title + duration match (artist missing)
        if not matched_track and not artist_n:
            for t in tracks:
                t_title = normalize_title(t.title_canonical)
                if t_title and t_title == title_n and _duration_close(
                    frec.audio_duration_sec, t.duration_sec_canonical,
                    config.duration_tolerance_strong,
                ):
                    matched_track = t
                    method = "title_duration"
                    score = 0.80
                    break

        if matched_track:
            frec.track_id = matched_track.track_id
            frec.match_method = method
            frec.match_score = score
            matched_track.linked_file_count += 1
            linked_count += 1
        else:
            # Create new logical track
            artist_n = normalize_artist(frec.embedded_artist)
            title_raw = frec.embedded_title or frec.file_name
            clean_title, mix_from_title = extract_mix_from_title(title_raw)

            track = LogicalTrack(
                track_id=_make_track_id(),
                identity_status="provisional",
                artist_canonical=frec.embedded_artist,
                title_canonical=clean_title,
                mix_canonical=mix_from_title,
                album_canonical=frec.embedded_album,
                duration_sec_canonical=frec.audio_duration_sec,
                isrc_canonical=frec.embedded_isrc,
                primary_file_id=frec.file_id,
                linked_file_count=1,
            )

            frec.track_id = track.track_id
            frec.is_primary_file = True
            frec.match_method = "new"
            frec.match_score = 1.0

            tracks.append(track)
            track_by_id[track.track_id] = track

            # Index for future matches
            ik = _identity_key(
                normalize_artist(track.artist_canonical),
                normalize_title(track.title_canonical),
                normalize_mix(track.mix_canonical),
            )
            if ik != "||||||":
                tracks_by_identity[ik] = track
            if frec.sha256:
                tracks_by_sha[frec.sha256] = track

            new_track_count += 1
        progress.update(index, frec.file_name, matched=linked_count, new=new_track_count)

    # Assign track_ids to tag observations that were missing them
    obs = store.load_observations()
    file_to_track = {f.file_id: f.track_id for f in files}
    changed = False
    for o in obs:
        if not o.track_id and o.file_id:
            tid = file_to_track.get(o.file_id, "")
            if tid:
                o.track_id = tid
                changed = True

    already_linked = sum(1 for f in files if f.track_id) - linked_count - new_track_count

    store.save_files(files)
    store.save_tracks(tracks)
    if changed:
        store.save_observations(obs)

    progress.finish()
    logger.info("Link: %d tracks (%d new, %d matched, %d existing)", len(tracks), new_track_count, linked_count, already_linked)

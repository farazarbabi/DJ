"""Canonical BPM resolver — tag BPM is killer vote, otherwise majority wins."""

from __future__ import annotations

import logging
from collections import defaultdict

from ..config import RegistryConfig
from ..models import LogicalTrack, SourceObservation
from ..progress import ProgressBar
from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)

# Tie-break priority when vote counts are equal (lower = preferred)
_TIE_BREAK_PRIORITY = {"tag": 0, "rekordbox": 1, "songstats": 2, "analysis_librosa": 3, "analysis_essentia": 4}


def _best_source_rank(sources: list[str]) -> int:
    return min(_TIE_BREAK_PRIORITY.get(s, 99) for s in sources)


def resolve_track_bpm(
    track: LogicalTrack,
    observations: list[SourceObservation],
) -> None:
    """Resolve the canonical BPM for a single track. Mutates the track in place.

    Rules:
    1. Tag BPM is a killer vote — if present, it wins unconditionally.
    2. Otherwise, majority vote among sources. Ties broken by source priority.
    3. Confidence = fraction of sources that agree with the winner.
    """
    bpm_obs = [(o.source_system, o.bpm.strip()) for o in observations if o.bpm and o.bpm.strip()]

    if not bpm_obs:
        track.canonical_bpm = ""
        track.canonical_bpm_confidence = 0.0
        return

    # Normalize BPM values to rounded integers for comparison
    normalized: list[tuple[str, str]] = []  # (source, bpm_str)
    for source, raw in bpm_obs:
        try:
            bpm_int = str(round(float(raw)))
            normalized.append((source, bpm_int))
        except ValueError:
            continue

    if not normalized:
        track.canonical_bpm = ""
        track.canonical_bpm_confidence = 0.0
        return

    # Group by BPM value
    bpm_sources: dict[str, list[str]] = defaultdict(list)
    for src, bpm in normalized:
        bpm_sources[bpm].append(src)

    # Sort: most votes first, then tag as killer tiebreaker, then source priority
    def _sort_key(item: tuple[str, list[str]]) -> tuple[int, int, int]:
        bpm_val, sources = item
        vote_count = -len(sources)  # more votes = better (negative for descending)
        has_tag = 0 if "tag" in sources else 1  # tag present = killer tiebreaker
        source_rank = _best_source_rank(sources)
        return (vote_count, has_tag, source_rank)

    ranked = sorted(bpm_sources.items(), key=_sort_key)

    winner_bpm, winner_sources = ranked[0]
    total_sources = len(normalized)
    confidence = round(len(winner_sources) / total_sources, 3)

    track.canonical_bpm = winner_bpm
    track.canonical_bpm_confidence = confidence


def resolve_all_bpms(
    config: RegistryConfig,
    store: CsvStore,
    *,
    force: bool = False,
    show_progress: bool = False,
) -> tuple[int, int]:
    """Resolve canonical BPM for all tracks.

    Returns (resolved_count, missing_count).
    """
    tracks = store.load_tracks()
    observations = store.load_observations()

    # Group observations by track
    obs_by_track: dict[str, list[SourceObservation]] = defaultdict(list)
    for o in observations:
        if o.track_id:
            obs_by_track[o.track_id].append(o)

    resolved = 0
    missing = 0
    progress = ProgressBar(len(tracks), label="Resolve BPM", enabled=show_progress)

    for index, track in enumerate(tracks, start=1):
        if not force and track.canonical_bpm:
            resolved += 1
            progress.update(index, track.title_canonical, resolved=resolved, missing=missing)
            continue

        track_obs = obs_by_track.get(track.track_id, [])
        resolve_track_bpm(track, track_obs)

        if track.canonical_bpm:
            resolved += 1
        else:
            missing += 1
        progress.update(index, track.title_canonical, resolved=resolved, missing=missing)

    progress.finish()
    store.save_tracks(tracks)
    logger.info("BPM: %d resolved, %d missing", resolved, missing)
    return resolved, missing

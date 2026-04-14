"""Registry overview and audit report generation."""

from __future__ import annotations

import csv
import logging
import os
from collections import defaultdict

from ..store.csv_store import CsvStore

logger = logging.getLogger(__name__)

OVERVIEW_COLUMNS = [
    # Identity
    "artist",
    "title",
    "mix",
    "file_name",
    "duration_sec",
    # Key comparison (all sources + canonical)
    "tag_key",
    "rekordbox_key",
    "songstats_key",
    "librosa_key",
    "librosa_confidence",
    "canonical_key",
    "canonical_confidence",
    "canonical_source",
    "resolution",
    "needs_review",
    "review_reason",
    # BPM
    "tag_bpm",
    "rekordbox_bpm",
    "songstats_bpm",
    # Metadata
    "rekordbox_genre",
    "rekordbox_label",
    "songstats_genre",
    "songstats_genres_all",
    "songstats_label",
    "songstats_release_date",
    "is_remix",
    # Audio features (Songstats/Spotify-derived)
    "acousticness",
    "danceability",
    "ss_energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "valence",
    # Rekordbox DJ data
    "rekordbox_rating",
    "rekordbox_play_count",
    # IDs
    "isrc",
    "spotify_id",
    "beatport_id",
    "track_id",
]


def generate_overview(store: CsvStore, output_dir: str) -> str:
    """Generate registry_overview.csv — one row per track, all sources as columns.

    Returns the path to the generated file.
    """
    tracks = store.load_tracks()
    files = store.load_files()
    observations = store.load_observations()

    # Index files by track
    file_by_track: dict[str, object] = {}
    for f in files:
        if f.is_primary_file and f.track_id:
            file_by_track[f.track_id] = f

    # Index observations by (track_id, source_system)
    obs_index: dict[tuple[str, str], object] = {}
    for o in observations:
        if o.track_id:
            obs_index[(o.track_id, o.source_system)] = o

    path = os.path.join(output_dir, "registry_overview.csv")
    os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OVERVIEW_COLUMNS, extrasaction="ignore")
        w.writeheader()

        for t in sorted(tracks, key=lambda x: (x.artist_canonical, x.title_canonical)):
            frec = file_by_track.get(t.track_id)
            tag_obs = obs_index.get((t.track_id, "tag"))
            rb_obs = obs_index.get((t.track_id, "rekordbox"))
            ss_obs = obs_index.get((t.track_id, "songstats"))
            lr_obs = obs_index.get((t.track_id, "analysis_librosa"))

            row = {
                # Identity
                "artist": t.artist_canonical,
                "title": t.title_canonical,
                "mix": t.mix_canonical,
                "file_name": frec.file_name if frec else "",
                "duration_sec": round(t.duration_sec_canonical, 1) if t.duration_sec_canonical else "",
                # Key comparison
                "tag_key": tag_obs.key_camelot if tag_obs else "",
                "rekordbox_key": rb_obs.key_camelot if rb_obs else "",
                "songstats_key": ss_obs.key_camelot if ss_obs else "",
                "librosa_key": lr_obs.key_camelot if lr_obs else "",
                "librosa_confidence": round(lr_obs.key_confidence, 2) if lr_obs and lr_obs.key_confidence else "",
                # Canonical
                "canonical_key": t.canonical_key_camelot,
                "canonical_confidence": round(t.canonical_key_confidence, 2) if t.canonical_key_confidence else "",
                "canonical_source": t.canonical_key_source,
                "resolution": t.canonical_key_resolution_reason,
                "needs_review": "YES" if t.needs_manual_review else "",
                "review_reason": t.review_reason if t.needs_manual_review else "",
                # BPM
                "tag_bpm": tag_obs.bpm if tag_obs else "",
                "rekordbox_bpm": rb_obs.bpm if rb_obs else "",
                "songstats_bpm": ss_obs.bpm if ss_obs else "",
                # Metadata
                "rekordbox_genre": rb_obs.genre if rb_obs else "",
                "rekordbox_label": rb_obs.label if rb_obs else "",
                "songstats_genre": ss_obs.genre if ss_obs else "",
                "songstats_genres_all": ss_obs.genres_all if ss_obs else "",
                "songstats_label": ss_obs.label if ss_obs else "",
                "songstats_release_date": ss_obs.release_date if ss_obs else "",
                "is_remix": ss_obs.is_remix if ss_obs and ss_obs.is_remix else (rb_obs.is_remix if rb_obs and rb_obs.is_remix else ""),
                # Audio features
                "acousticness": ss_obs.acousticness if ss_obs else "",
                "danceability": ss_obs.danceability if ss_obs else "",
                "ss_energy": ss_obs.energy if ss_obs else "",
                "instrumentalness": ss_obs.instrumentalness if ss_obs else "",
                "liveness": ss_obs.liveness if ss_obs else "",
                "loudness": ss_obs.loudness if ss_obs else "",
                "speechiness": ss_obs.speechiness if ss_obs else "",
                "valence": ss_obs.valence if ss_obs else "",
                # Rekordbox DJ data
                "rekordbox_rating": rb_obs.rating if rb_obs else "",
                "rekordbox_play_count": rb_obs.play_count if rb_obs else "",
                # IDs
                "isrc": t.isrc_canonical,
                "spotify_id": ss_obs.spotify_id if ss_obs else "",
                "beatport_id": ss_obs.beatport_id if ss_obs else "",
                "track_id": t.track_id,
            }
            w.writerow(row)

    logger.info("Overview: %s (%d tracks)", path, len(tracks))
    return path


def generate_reports(store: CsvStore, reports_dir: str) -> None:
    """Generate all reports including the overview."""
    # Always generate the overview in the registry root
    output_dir = os.path.dirname(reports_dir) if reports_dir.endswith("reports") else reports_dir
    generate_overview(store, output_dir)

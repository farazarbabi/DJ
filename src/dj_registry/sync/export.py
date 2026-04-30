"""Registry overview and audit report generation."""

from __future__ import annotations

import csv
import logging
import os

from ..store.csv_store import CsvStore, _sanitize_cell

logger = logging.getLogger(__name__)

OVERVIEW_COLUMNS = [
    # Identity
    "file_name",
    "title",
    "artist",
    "mix",
    "duration_sec",
    # Key comparison (all sources + canonical)
    "tag_key",
    "rekordbox_key",
    "songstats_key",
    "librosa_key",
    "librosa_confidence",
    "canonical_key",
    "canonical_key_confidence",
    "canonical_key_source",
    "needs_review",
    # BPM
    "tag_bpm",
    "rekordbox_bpm",
    "songstats_bpm",
    "canonical_bpm",
    "bpm_confidence",
    # 3-level genre classification
    "genre_family",
    "genre",
    "subgenre",
    "genre_confidence",
    "genre_confidence_level",
    "genre_alternatives",
    "genre_evidence",
    "genre_warnings",
    "genre_taxonomy_version",
    # Legacy DJ-functional taxonomy
    "taxonomy_id",
    "taxonomy_label",
    "taxonomy_confidence",
    "taxonomy_alternatives",
    "taxonomy_evidence",
    "taxonomy_version",
    # Metadata
    "rekordbox_genre",
    "rekordbox_label",
    "songstats_genre",
    "songstats_genres_all",
    "songstats_label",
    "songstats_release_date",
    "is_remix",
    # Audio features (Songstats/Spotify-derived)
    "ss_acousticness",
    "ss_danceability",
    "ss_energy",
    "ss_instrumentalness",
    "ss_liveness",
    "ss_loudness",
    "ss_speechiness",
    "ss_valence",
    # Tagger analysis features
    "tagger_energy",
    "tagger_vibe",
    "tagger_vocal",
    "tagger_structure",
    "tagger_bpm",
    "tagger_vibe_scores",
    "tagger_confidences",
    "tagger_version",
    "tagger_raw_signature",
    "tagger_derived_signature",
    "tagger_key_signature",
    "tagger_audio_features_signature",
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
                "canonical_key_confidence": round(t.canonical_key_confidence, 2) if t.canonical_key_confidence else "",
                "canonical_key_source": t.canonical_key_source,
                "needs_review": "YES" if t.needs_manual_review else "NO",
                # BPM
                "tag_bpm": tag_obs.bpm if tag_obs else "",
                "rekordbox_bpm": rb_obs.bpm if rb_obs else "",
                "songstats_bpm": ss_obs.bpm if ss_obs else "",
                "canonical_bpm": t.canonical_bpm,
                "bpm_confidence": round(t.canonical_bpm_confidence, 2) if t.canonical_bpm_confidence else "",
                # 3-level genre classification
                "genre_family": t.genre_family,
                "genre": t.genre,
                "subgenre": t.subgenre,
                "genre_confidence": round(t.genre_confidence, 2) if t.genre_confidence else "",
                "genre_confidence_level": t.genre_confidence_level,
                "genre_alternatives": t.genre_alternatives,
                "genre_evidence": t.genre_evidence,
                "genre_warnings": t.genre_warnings,
                "genre_taxonomy_version": t.genre_taxonomy_version,
                # Legacy DJ-functional taxonomy
                "taxonomy_id": t.taxonomy_id,
                "taxonomy_label": t.taxonomy_label,
                "taxonomy_confidence": round(t.taxonomy_confidence, 2) if t.taxonomy_confidence else "",
                "taxonomy_alternatives": t.taxonomy_alternatives,
                "taxonomy_evidence": t.taxonomy_evidence,
                "taxonomy_version": t.taxonomy_version,
                # Metadata
                "rekordbox_genre": rb_obs.genre if rb_obs else "",
                "rekordbox_label": rb_obs.label if rb_obs else "",
                "songstats_genre": ss_obs.genre if ss_obs else "",
                "songstats_genres_all": ss_obs.genres_all if ss_obs else "",
                "songstats_label": ss_obs.label if ss_obs else "",
                "songstats_release_date": ss_obs.release_date if ss_obs else "",
                "is_remix": ss_obs.is_remix if ss_obs and ss_obs.is_remix else (rb_obs.is_remix if rb_obs and rb_obs.is_remix else ""),
                # Audio features
                "ss_acousticness": ss_obs.acousticness if ss_obs else "",
                "ss_danceability": ss_obs.danceability if ss_obs else "",
                "ss_energy": ss_obs.energy if ss_obs else "",
                "ss_instrumentalness": ss_obs.instrumentalness if ss_obs else "",
                "ss_liveness": ss_obs.liveness if ss_obs else "",
                "ss_loudness": ss_obs.loudness if ss_obs else "",
                "ss_speechiness": ss_obs.speechiness if ss_obs else "",
                "ss_valence": ss_obs.valence if ss_obs else "",
                # Tagger analysis features
                "tagger_energy": t.tagger_energy,
                "tagger_vibe": t.tagger_vibe,
                "tagger_vocal": t.tagger_vocal,
                "tagger_structure": t.tagger_structure,
                "tagger_bpm": t.tagger_bpm,
                "tagger_vibe_scores": t.tagger_vibe_scores,
                "tagger_confidences": t.tagger_confidences,
                "tagger_version": t.tagger_version,
                "tagger_raw_signature": t.tagger_raw_signature,
                "tagger_derived_signature": t.tagger_derived_signature,
                "tagger_key_signature": t.tagger_key_signature,
                "tagger_audio_features_signature": t.tagger_audio_features_signature,
                # IDs
                "isrc": t.isrc_canonical,
                "spotify_id": ss_obs.spotify_id if ss_obs else "",
                "beatport_id": ss_obs.beatport_id if ss_obs else "",
                "track_id": t.track_id,
            }
            w.writerow({k: _sanitize_cell(str(v)) for k, v in row.items()})

    logger.info("Overview: %s (%d tracks)", path, len(tracks))
    return path


def generate_reports(store: CsvStore, reports_dir: str) -> None:
    """Generate all reports including the overview.

    Writes registry_overview.csv both in the registry dir and in the
    top-level outputs/ dir for easy access.
    """
    # Always generate the overview in the registry root
    output_dir = os.path.dirname(reports_dir) if reports_dir.endswith("reports") else reports_dir
    overview_path = generate_overview(store, output_dir)

    # Copy to top-level outputs/ for easy access
    top_outputs = os.path.dirname(output_dir)
    if top_outputs and top_outputs != output_dir:
        import shutil
        dest = os.path.join(top_outputs, "registry_overview.csv")
        try:
            shutil.copy2(overview_path, dest)
        except PermissionError:
            logger.warning("Could not update %s because it is locked/open", dest)

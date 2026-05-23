"""Data models for the track registry."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class LogicalTrack:
    """One row in tracks_master.csv."""

    track_id: str = ""
    identity_status: str = "provisional"  # confirmed | provisional | ambiguous

    # Canonical identity
    artist_canonical: str = ""
    title_canonical: str = ""
    mix_canonical: str = ""
    album_canonical: str = ""
    label_canonical: str = ""
    release_date_canonical: str = ""
    duration_sec_canonical: float = 0.0
    isrc_canonical: str = ""

    # Canonical resolved key
    canonical_key_standard: str = ""
    canonical_key_camelot: str = ""
    canonical_key_confidence: float = 0.0
    canonical_key_source: str = ""
    canonical_key_resolution_reason: str = ""
    key_evidence_summary: str = ""

    # Future canonical fields
    canonical_bpm: str = ""
    canonical_bpm_confidence: float = 0.0
    canonical_genre: str = ""

    # 3-level genre classification (curated static vocabulary + classifier)
    genre_family: str = ""
    genre: str = ""
    subgenre: str = ""
    genre_confidence: float = 0.0
    genre_confidence_level: str = ""
    genre_alternatives: str = ""
    genre_evidence: str = ""
    genre_warnings: str = ""
    genre_taxonomy_version: str = ""

    # Legacy DJ-functional taxonomy fields kept for downstream compatibility
    taxonomy_id: str = ""
    taxonomy_label: str = ""
    taxonomy_confidence: float = 0.0
    taxonomy_alternatives: str = ""
    taxonomy_evidence: str = ""
    taxonomy_version: str = ""

    # Flat DJ-functional taxonomy ML fields from dj_taxonomy.json
    dj_taxonomy_id: str = ""
    dj_taxonomy_label: str = ""
    dj_taxonomy_family: str = ""
    dj_taxonomy_moods: str = ""
    dj_taxonomy_grooves: str = ""
    dj_taxonomy_set_roles: str = ""
    dj_taxonomy_bpm_range: str = ""
    dj_taxonomy_energy_range: str = ""
    dj_taxonomy_vocal_profiles: str = ""
    dj_taxonomy_source_genres: str = ""
    dj_taxonomy_keywords: str = ""
    dj_taxonomy_confidence: float = 0.0
    dj_taxonomy_confidence_level: str = ""  # high | medium-high | medium | low | unknown
    dj_taxonomy_source_model: str = ""
    dj_taxonomy_internal_id: str = ""
    dj_taxonomy_internal_label: str = ""
    dj_taxonomy_internal_confidence: float = 0.0
    dj_taxonomy_external_id: str = ""
    dj_taxonomy_external_label: str = ""
    dj_taxonomy_external_confidence: float = 0.0
    dj_taxonomy_models_agree: str = ""
    dj_taxonomy_external_evidence_available: str = ""
    dj_taxonomy_alternatives: str = ""
    dj_taxonomy_evidence: str = ""
    dj_taxonomy_version: str = ""

    # Tagger analysis features (single-source, stored directly)
    tagger_energy: str = ""
    tagger_vibe: str = ""
    tagger_vocal: str = ""
    tagger_structure: str = ""
    tagger_bpm: str = ""
    tagger_vibe_scores: str = ""
    tagger_vocal_scores: str = ""
    tagger_confidences: str = ""
    tagger_version: str = ""
    tagger_raw_signature: str = ""
    tagger_derived_signature: str = ""
    tagger_key_signature: str = ""
    tagger_audio_features_signature: str = ""

    # Resolution control
    needs_manual_review: bool = False
    review_reason: str = ""
    primary_file_id: str = ""
    linked_file_count: int = 0
    last_resolved_at: str = ""
    registry_notes: str = ""


@dataclass
class FileRecord:
    """One row in files_master.csv."""

    file_id: str = ""
    track_id: str = ""
    is_primary_file: bool = False
    match_method: str = ""  # sha256 | exact_normalized | fuzzy | manual
    match_score: float = 0.0

    # File system
    path_abs: str = ""
    path_rel: str = ""
    file_name: str = ""
    extension: str = ""
    size_bytes: int = 0
    mtime_utc: str = ""
    sha256: str = ""

    # Audio technical
    audio_duration_sec: float = 0.0
    sample_rate: int = 0
    bitrate: int = 0
    channels: int = 0
    bits_per_sample: int = 0

    # Embedded tags
    embedded_title: str = ""
    embedded_artist: str = ""
    embedded_album: str = ""
    embedded_genre: str = ""
    embedded_bpm: str = ""
    embedded_key_standard: str = ""
    embedded_key_camelot: str = ""
    embedded_comment: str = ""
    embedded_isrc: str = ""

    # Read/write traceability
    tag_read_status: str = ""  # ok | error | missing
    tag_read_error: str = ""
    tag_write_status: str = ""  # ok | error | pending | skipped
    tag_write_error: str = ""
    last_scanned_at: str = ""
    last_tag_written_at: str = ""

    # Payload
    file_tag_payload_ref: str = ""


@dataclass
class SourceObservation:
    """One row in source_observations.csv — wide format.

    One row per (track, source). All fields as columns.
    """

    observation_id: str = ""
    track_id: str = ""
    file_id: str = ""

    # Human-readable identity (denormalized for easy review)
    artist: str = ""
    title: str = ""

    source_system: str = ""  # tag | songstats | rekordbox | analysis_librosa | analysis_essentia | manual
    source_object_id: str = ""

    # Key
    key_standard: str = ""
    key_camelot: str = ""
    key_confidence: float = 0.0

    # Core metadata
    bpm: str = ""
    genre: str = ""
    genres_all: str = ""  # semicolon-separated full genre list
    label: str = ""
    release_date: str = ""
    year: str = ""
    is_remix: str = ""

    # Rekordbox-specific
    rating: str = ""
    play_count: str = ""
    date_added: str = ""
    comments: str = ""

    # Audio features (Songstats/Spotify-derived, 0.0-1.0 scale)
    acousticness: str = ""
    danceability: str = ""
    energy: str = ""
    instrumentalness: str = ""
    liveness: str = ""
    loudness: str = ""
    speechiness: str = ""
    valence: str = ""
    time_signature: str = ""

    # Spotify popularity (0-100, recency-weighted by Spotify)
    popularity: str = ""

    # Tagger analysis features
    tagger_energy: str = ""
    tagger_vibe: str = ""
    tagger_vocal: str = ""
    tagger_structure: str = ""
    tagger_vibe_scores: str = ""
    tagger_vocal_scores: str = ""
    tagger_confidences: str = ""

    # Platform IDs (for cross-referencing)
    spotify_id: str = ""
    beatport_id: str = ""
    apple_music_id: str = ""

    # Traceability
    payload_ref: str = ""
    observed_at: str = ""
    notes: str = ""


@dataclass
class ReviewItem:
    """One row in review_queue.csv."""

    review_id: str = ""
    track_id: str = ""
    priority: str = "medium"  # high | medium | low
    reason_code: str = ""
    reason_detail: str = ""

    candidate_key_standard_1: str = ""
    candidate_key_camelot_1: str = ""
    candidate_source_1: str = ""
    candidate_score_1: float = 0.0

    candidate_key_standard_2: str = ""
    candidate_key_camelot_2: str = ""
    candidate_source_2: str = ""
    candidate_score_2: float = 0.0

    candidate_key_standard_3: str = ""
    candidate_key_camelot_3: str = ""
    candidate_source_3: str = ""
    candidate_score_3: float = 0.0

    suggested_key_standard: str = ""
    suggested_key_camelot: str = ""
    primary_file_id: str = ""
    primary_file_path: str = ""
    key_evidence_summary: str = ""
    created_at: str = ""

    # User-editable
    reviewer_key_standard: str = ""
    reviewer_key_camelot: str = ""
    reviewer_decision: str = ""  # accept | override | skip
    reviewer_note: str = ""
    resolved_at: str = ""


@dataclass
class PayloadIndexEntry:
    """One row in source_payload_index.csv."""

    payload_ref: str = ""
    source_system: str = ""
    source_type: str = ""  # json | xml | pickle
    source_object_id: str = ""
    track_id: str = ""
    file_id: str = ""
    payload_path: str = ""
    fetched_at: str = ""
    notes: str = ""


def now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

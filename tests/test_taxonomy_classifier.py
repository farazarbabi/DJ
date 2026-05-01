"""Tests for 3-level genre classification."""

import csv
import json
from argparse import Namespace

from dj_registry.cli import cmd_taxonomy
from dj_registry.models import FileRecord, LogicalTrack, SourceObservation
from dj_registry.store.csv_store import CsvStore
from dj_registry.sync.export import generate_overview
from dj_registry.taxonomy.classifier import classify_all_taxonomies, classify_track, load_taxonomy


def test_taxonomy_file_is_three_level_reference():
    taxonomy = load_taxonomy()

    assert len(taxonomy.families()) >= 10
    assert "House" in taxonomy.families()
    assert "Tech House" in taxonomy.genres("House")
    assert "Rolling Tech House" in taxonomy.subgenres("House", "Tech House")
    assert taxonomy.validate_path("House", "Tech House", "Rolling Tech House")
    assert not taxonomy.validate_path("House", "Techno", "Rolling Hypnotic Techno")


def test_classify_clear_tech_house_to_rolling_subgenre():
    track = LogicalTrack(
        track_id="T-001",
        tagger_energy="E4",
        tagger_structure="16H",
        tagger_bpm="126",
    )
    observations = [
        SourceObservation(
            track_id="T-001",
            source_system="rekordbox",
            genre="Tech House",
        )
    ]
    file_record = FileRecord(
        track_id="T-001",
        is_primary_file=True,
        embedded_genre="Tech House",
    )

    result = classify_track(track, observations, file_record)

    assert result.family == "House"
    assert result.genre == "Tech House"
    assert result.subgenre == "Rolling Tech House"
    assert result.confidence_level in {"medium-high", "high"}


def test_classify_dark_hypnotic_techno():
    track = LogicalTrack(
        track_id="T-002",
        tagger_energy="E4",
        tagger_vibe="DRK,HYP",
        tagger_vocal="NV",
        tagger_structure="16H",
        tagger_bpm="132",
    )
    observations = [
        SourceObservation(
            track_id="T-002",
            source_system="songstats",
            genre="Techno",
            genres_all="Techno; Hypnotic Techno",
            energy="0.82",
            valence="0.21",
            instrumentalness="0.91",
        )
    ]

    result = classify_track(track, observations)

    assert result.family == "Techno"
    assert result.genre == "Minimal / Hypnotic Techno"
    assert result.subgenre == "Rolling Hypnotic Techno"


def test_classify_uk_garage_not_garage_rock():
    track = LogicalTrack(
        track_id="T-003",
        tagger_structure="BREAKS",
        tagger_bpm="134",
    )
    file_record = FileRecord(track_id="T-003", embedded_genre="Garage")
    observations = [
        SourceObservation(
            track_id="T-003",
            source_system="songstats",
            genres_all="UKG; 2 Step; Bassline",
        )
    ]

    result = classify_track(track, observations, file_record)

    assert result.family == "Garage / UK Bass / Breaks"
    assert result.genre == "UK Garage"
    assert result.subgenre == "2-Step Garage"


def test_classify_garage_rock_not_uk_garage():
    track = LogicalTrack(track_id="T-004")
    observations = [
        SourceObservation(
            track_id="T-004",
            source_system="songstats",
            genre="Garage Rock",
            genres_all="garage punk; psych rock",
            acousticness="0.74",
            energy="0.71",
        )
    ]

    result = classify_track(track, observations)

    assert result.family == "Rock / Metal / Guitar-Based"
    assert result.genre == "Garage / Punk / Post-Punk"
    assert result.subgenre == "Garage Rock"
    assert "garage house" not in result.evidence.get("source_genres_used", [])


def test_classify_organic_desert_house():
    track = LogicalTrack(
        track_id="T-005",
        tagger_energy="E2",
        tagger_vibe="ORG,MEL",
        tagger_bpm="120",
    )
    observations = [
        SourceObservation(
            track_id="T-005",
            source_system="songstats",
            genres_all="Organic House; Downtempo",
            comments="middle eastern percussion desert sunset vibe",
            instrumentalness="0.84",
        )
    ]

    result = classify_track(track, observations)

    assert result.family == "House"
    assert result.genre == "Organic / Afro / Tribal House"
    assert result.subgenre == "Desert House"


def test_generic_electronic_uses_tagger_and_audio_for_deep_downtempo():
    track = LogicalTrack(
        track_id="T-006",
        tagger_energy="E3",
        tagger_vibe="DEEP",
        tagger_vocal="V",
        tagger_structure="16H",
        tagger_bpm="109",
    )
    observations = [
        SourceObservation(
            track_id="T-006",
            source_system="songstats",
            genre="Electronic",
            genres_all="Electronic; Electronica",
            danceability="0.80",
            energy="0.52",
            instrumentalness="0.86",
            valence="0.25",
        )
    ]

    result = classify_track(track, observations)

    assert result.family == "Downtempo / Slow Electronic"
    assert result.genre == "Downtempo"
    assert result.subgenre == "Deep Downtempo"


def test_generic_dance_with_tribal_tagger_uses_ethnic_downtempo():
    track = LogicalTrack(
        track_id="T-007",
        tagger_energy="E3",
        tagger_vibe="TRIB",
        tagger_vocal="V",
        tagger_structure="16H",
        tagger_bpm="102",
    )
    observations = [
        SourceObservation(
            track_id="T-007",
            source_system="songstats",
            genre="Electronic",
            genres_all="Electronic; Dance",
            danceability="0.70",
            energy="0.62",
            instrumentalness="0.90",
            valence="0.20",
        )
    ]

    result = classify_track(track, observations)

    assert result.family == "Downtempo / Slow Electronic"
    assert result.genre == "Psychedelic / Ethnic Downtempo"
    assert result.subgenre == "Tribal Downtempo"


def test_provider_parenthetical_techno_maps_to_peak_time_techno():
    track = LogicalTrack(
        track_id="T-008",
        tagger_energy="E4",
        tagger_vibe="DRK",
        tagger_structure="32H",
        tagger_bpm="123",
    )
    file_record = FileRecord(
        track_id="T-008",
        is_primary_file=True,
        embedded_genre="Techno (Peak Time / Driving)",
    )

    result = classify_track(track, [], file_record)

    assert result.family == "Techno"
    assert result.genre == "Driving / Peak-Time Techno"
    assert result.subgenre in {"Peak-Time Techno", "Driving Techno", "Rolling Techno"}


def test_analysis_tagger_fields_used_when_track_fields_are_blank():
    track = LogicalTrack(track_id="T-009")
    observations = [
        SourceObservation(
            track_id="T-009",
            source_system="analysis_librosa",
            tagger_energy="2",
            tagger_vibe="DEEP",
            tagger_vocal="V",
            tagger_structure="16H",
            bpm="108",
        )
    ]

    result = classify_track(track, observations)

    assert result.family == "Downtempo / Slow Electronic"
    assert result.genre == "Downtempo"
    assert result.subgenre == "Deep Downtempo"


def test_weak_evidence_returns_unknown_path():
    track = LogicalTrack(
        track_id="T-010",
        artist_canonical="Unknown Artist",
        title_canonical="Untitled",
    )

    result = classify_track(track, [])

    assert result.family is None
    assert result.genre is None
    assert result.subgenre is None
    assert result.confidence_level == "unknown"
    assert result.warnings


def test_classify_all_persists_fields_and_overview(tmp_path):
    store = CsvStore(str(tmp_path))
    store.save_tracks([
        LogicalTrack(
            track_id="T-001",
            tagger_energy="E4",
            tagger_structure="16H",
            tagger_bpm="126",
        )
    ])
    store.save_files([
        FileRecord(
            file_id="F001",
            track_id="T-001",
            is_primary_file=True,
            embedded_genre="Tech House",
        )
    ])
    store.save_observations([
        SourceObservation(
            observation_id="OBS-001",
            track_id="T-001",
            file_id="F001",
            source_system="rekordbox",
            genre="Tech House",
        )
    ])

    count = classify_all_taxonomies(store)
    overview_path = generate_overview(store, str(tmp_path))

    loaded = store.load_tracks()[0]
    assert count == 1
    assert loaded.genre_family == "House"
    assert loaded.genre == "Tech House"
    assert loaded.subgenre == "Rolling Tech House"
    assert loaded.canonical_genre == "Rolling Tech House"
    assert loaded.taxonomy_label == "House > Tech House > Rolling Tech House"
    assert json.loads(loaded.genre_evidence)["tagger_signals"]

    with open(overview_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["genre_family"] == "House"
    assert rows[0]["genre"] == "Tech House"
    assert rows[0]["subgenre"] == "Rolling Tech House"
    assert "tagger_mood" in rows[0]
    assert "tagger_mood_scores" in rows[0]
    assert rows[0]["taxonomy_label"] == "House > Tech House > Rolling Tech House"


def test_taxonomy_cli_regenerates_registry_overview(tmp_path):
    store = CsvStore(str(tmp_path))
    store.save_tracks([
        LogicalTrack(
            track_id="T-001",
            tagger_energy="E4",
            tagger_structure="16H",
            tagger_bpm="126",
        )
    ])
    store.save_files([
        FileRecord(
            file_id="F001",
            track_id="T-001",
            is_primary_file=True,
            embedded_genre="Tech House",
        )
    ])
    store.save_observations([])

    rc = cmd_taxonomy(Namespace(output=str(tmp_path), taxonomy=None))

    assert rc == 0
    overview_path = tmp_path / "registry_overview.csv"
    with overview_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["genre_family"] == "House"
    assert rows[0]["genre"] == "Tech House"
    assert rows[0]["subgenre"] == "Rolling Tech House"

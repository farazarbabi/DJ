"""Tests for registry bridge: loading, matching, and upgrading."""

import csv
import os

import pytest

from dj_grouper.features.registry_bridge import (
    RegistryBridge,
    RegistryEnrichment,
    match_enrichments,
    apply_registry_upgrades,
)
from dj_grouper.scanner import TrackInfo


@pytest.fixture
def registry_csv(tmp_path):
    """Create a minimal registry_overview.csv."""
    overview = tmp_path / "registry_overview.csv"
    rows = [
        {
            "file_name": "track_a.aiff",
            "danceability": "0.72",
            "valence": "0.31",
            "songstats_genre": "Melodic Techno",
            "songstats_genres_all": "Melodic Techno;Techno",
            "rekordbox_genre": "Techno",
            "canonical_key": "9A",
            "canonical_bpm": "126",
        },
        {
            "file_name": "track_b.mp3",
            "danceability": "",
            "valence": "",
            "songstats_genre": "",
            "songstats_genres_all": "",
            "rekordbox_genre": "Deep House",
            "canonical_key": "4B",
            "canonical_bpm": "122",
        },
    ]
    fieldnames = list(rows[0].keys())
    with open(overview, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return tmp_path


def test_load_returns_enrichments(registry_csv):
    bridge = RegistryBridge(str(registry_csv))
    enrichments = bridge.load()
    assert len(enrichments) == 2
    assert "track_a.aiff" in enrichments
    assert "track_b.mp3" in enrichments


def test_songstats_features_parsed(registry_csv):
    bridge = RegistryBridge(str(registry_csv))
    enrichments = bridge.load()
    enr = enrichments["track_a.aiff"]
    assert enr.danceability == pytest.approx(0.72)
    assert enr.valence == pytest.approx(0.31)
    assert enr.has_songstats is True


def test_missing_songstats_features(registry_csv):
    bridge = RegistryBridge(str(registry_csv))
    enrichments = bridge.load()
    enr = enrichments["track_b.mp3"]
    assert enr.danceability is None
    assert enr.valence is None
    assert enr.has_songstats is False


def test_genre_parsing(registry_csv):
    bridge = RegistryBridge(str(registry_csv))
    enrichments = bridge.load()
    enr_a = enrichments["track_a.aiff"]
    assert enr_a.primary_genre == "Melodic Techno"
    assert enr_a.all_genres == ["Melodic Techno", "Techno"]

    enr_b = enrichments["track_b.mp3"]
    assert enr_b.primary_genre == "Deep House"
    assert enr_b.all_genres == ["Deep House"]


def test_canonical_key_bpm(registry_csv):
    bridge = RegistryBridge(str(registry_csv))
    enrichments = bridge.load()
    assert enrichments["track_a.aiff"].canonical_key == "9A"
    assert enrichments["track_a.aiff"].canonical_bpm == 126


def test_match_enrichments_by_filename(registry_csv):
    bridge = RegistryBridge(str(registry_csv))
    raw = bridge.load()
    paths = ["/music/track_a.aiff", "/music/track_b.mp3", "/music/track_c.wav"]
    matched = match_enrichments(paths, raw)
    assert "/music/track_a.aiff" in matched
    assert "/music/track_b.mp3" in matched
    assert "/music/track_c.wav" not in matched


def test_apply_registry_upgrades():
    infos = [
        TrackInfo(path="/music/track_a.aiff", key="??", bpm=120),
        TrackInfo(path="/music/track_b.mp3", key="1A", bpm=130),
    ]
    enrichments = {
        "/music/track_a.aiff": RegistryEnrichment(canonical_key="9A", canonical_bpm=126),
    }
    n = apply_registry_upgrades(infos, enrichments)
    assert n == 2  # key + bpm
    assert infos[0].key == "9A"
    assert infos[0].bpm == 126
    # Unmatched track unchanged
    assert infos[1].key == "1A"
    assert infos[1].bpm == 130


def test_load_missing_file():
    bridge = RegistryBridge("/nonexistent/path")
    enrichments = bridge.load()
    assert enrichments == {}

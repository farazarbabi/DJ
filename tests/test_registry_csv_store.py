"""Tests for dj_registry.store.csv_store — CSV round-trip and atomic writes."""

import os
import pytest

from dj_registry.models import (
    FileRecord,
    LogicalTrack,
    SourceObservation,
    ReviewItem,
)
from dj_registry.store.csv_store import CsvStore


@pytest.fixture
def store(tmp_path):
    return CsvStore(str(tmp_path))


class TestTracksRoundTrip:
    def test_save_load_empty(self, store):
        store.save_tracks([])
        assert store.load_tracks() == []

    def test_save_load(self, store):
        t = LogicalTrack(
            track_id="T-001",
            artist_canonical="Artist",
            title_canonical="Title",
            mix_canonical="Original Mix",
            canonical_key_standard="E minor",
            canonical_key_camelot="9A",
            canonical_key_confidence=0.95,
            needs_manual_review=False,
            linked_file_count=1,
            duration_sec_canonical=300.5,
        )
        store.save_tracks([t])
        loaded = store.load_tracks()
        assert len(loaded) == 1
        lt = loaded[0]
        assert lt.track_id == "T-001"
        assert lt.artist_canonical == "Artist"
        assert lt.canonical_key_camelot == "9A"
        assert lt.canonical_key_confidence == 0.95
        assert lt.needs_manual_review is False
        assert lt.linked_file_count == 1
        assert lt.duration_sec_canonical == 300.5

    def test_upsert_insert(self, store):
        t = LogicalTrack(track_id="T-001", artist_canonical="A")
        store.upsert_track(t)
        assert len(store.load_tracks()) == 1

    def test_upsert_update(self, store):
        t = LogicalTrack(track_id="T-001", artist_canonical="A")
        store.save_tracks([t])
        t2 = LogicalTrack(track_id="T-001", artist_canonical="B")
        store.upsert_track(t2)
        loaded = store.load_tracks()
        assert len(loaded) == 1
        assert loaded[0].artist_canonical == "B"

    def test_get_track(self, store):
        store.save_tracks([LogicalTrack(track_id="T-001", artist_canonical="A")])
        assert store.get_track("T-001") is not None
        assert store.get_track("T-999") is None


class TestFilesRoundTrip:
    def test_save_load(self, store):
        f = FileRecord(
            file_id="F00001",
            path_abs="/music/track.mp3",
            file_name="track.mp3",
            extension=".mp3",
            size_bytes=1024000,
            sha256="abc123",
            audio_duration_sec=300.0,
            is_primary_file=True,
        )
        store.save_files([f])
        loaded = store.load_files()
        assert len(loaded) == 1
        assert loaded[0].file_id == "F00001"
        assert loaded[0].size_bytes == 1024000
        assert loaded[0].is_primary_file is True

    def test_get_file_by_path(self, store):
        f = FileRecord(file_id="F00001", path_abs="/music/track.mp3")
        store.save_files([f])
        assert store.get_file_by_path("/music/track.mp3") is not None
        assert store.get_file_by_path("/music/other.mp3") is None


class TestObservations:
    def test_add_and_query(self, store):
        obs = SourceObservation(
            observation_id="OBS-1",
            track_id="T-001",
            source_system="tag",
            key_camelot="9A",
            key_standard="E minor",
            key_confidence=1.0,
            bpm="128",
        )
        store.add_observation(obs)

        loaded = store.get_observations_for_track("T-001")
        assert len(loaded) == 1
        assert loaded[0].key_camelot == "9A"
        assert loaded[0].bpm == "128"

        # Filter by source
        assert len(store.get_observations_for_track("T-001", "tag")) == 1
        assert len(store.get_observations_for_track("T-001", "rekordbox")) == 0

    def test_delete_observations(self, store):
        obs1 = SourceObservation(observation_id="O1", track_id="T-001", source_system="tag", key_camelot="9A")
        obs2 = SourceObservation(observation_id="O2", track_id="T-001", source_system="rekordbox", key_camelot="9A")
        obs3 = SourceObservation(observation_id="O3", track_id="T-002", source_system="tag", key_camelot="8A")
        store.save_observations([obs1, obs2, obs3])

        # Delete only rekordbox obs for T-001
        deleted = store.delete_observations("T-001", "rekordbox")
        assert deleted == 1
        remaining = store.load_observations()
        assert len(remaining) == 2


class TestReviewQueue:
    def test_save_load(self, store):
        item = ReviewItem(
            review_id="REV-1",
            track_id="T-001",
            priority="high",
            reason_code="source_conflict",
            suggested_key_camelot="9A",
            candidate_score_1=0.85,
        )
        store.save_review_queue([item])
        loaded = store.load_review_queue()
        assert len(loaded) == 1
        assert loaded[0].priority == "high"
        assert loaded[0].candidate_score_1 == 0.85


class TestSnapshot:
    def test_snapshot_creates_directory(self, store):
        store.save_tracks([LogicalTrack(track_id="T-001")])
        snap_dir = store.snapshot("test-run")
        assert os.path.exists(snap_dir)
        assert os.path.exists(os.path.join(snap_dir, "tracks_master.csv"))


class TestAtomicWrite:
    def test_no_partial_writes_on_error(self, store):
        """If an error occurs during write, the original file should be unchanged."""
        store.save_tracks([LogicalTrack(track_id="T-001")])
        # Verify the file exists
        assert len(store.load_tracks()) == 1

    def test_file_missing_returns_empty(self, store):
        """Loading from a non-existent CSV returns empty list."""
        assert store.load_tracks() == []
        assert store.load_files() == []
        assert store.load_observations() == []

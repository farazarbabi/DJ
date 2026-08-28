"""Tests for registry identity linking."""

from __future__ import annotations

from dj_registry.config import RegistryConfig
from dj_registry.identity.matcher import link_files_to_tracks
from dj_registry.models import FileRecord, LogicalTrack
from dj_registry.store.csv_store import CsvStore


def test_link_files_repairs_stale_primary_file_links(tmp_path):
    store = CsvStore(str(tmp_path / "registry"))
    tracks = [
        LogicalTrack(track_id="T1", artist_canonical="Slow", title_canonical="Song", primary_file_id="F2", linked_file_count=1),
        LogicalTrack(track_id="T2", artist_canonical="Fast", title_canonical="Song", primary_file_id="F2", linked_file_count=1),
    ]
    files = [
        FileRecord(file_id="F1", track_id="T1", path_abs="/music/slow.aiff", file_name="slow.aiff", is_primary_file=False),
        FileRecord(file_id="F2", track_id="T2", path_abs="/music/fast.aiff", file_name="fast.aiff", is_primary_file=True),
    ]
    store.save_tracks(tracks)
    store.save_files(files)

    link_files_to_tracks(RegistryConfig(output_dir=str(tmp_path / "registry")), store)

    loaded_tracks = {t.track_id: t for t in store.load_tracks()}
    loaded_files = {f.file_id: f for f in store.load_files()}
    assert loaded_tracks["T1"].primary_file_id == "F1"
    assert loaded_tracks["T1"].linked_file_count == 1
    assert loaded_tracks["T2"].primary_file_id == "F2"
    assert loaded_tracks["T2"].linked_file_count == 1
    assert loaded_files["F1"].is_primary_file is True
    assert loaded_files["F2"].is_primary_file is True

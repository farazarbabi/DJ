"""Tests for library scanner tag normalization."""

from pathlib import Path

from dj_grouper import scanner


def test_scan_library_normalizes_unknown_key_placeholder(monkeypatch, tmp_path):
    track = tmp_path / "track.mp3"
    track.write_bytes(b"")

    monkeypatch.setattr(scanner, "find_audio_files", lambda paths, recursive, exclude_dirs: [track])
    monkeypatch.setattr(scanner, "read_existing_tag", lambda path: "??_124_E3_DRK_INST")

    tracks = scanner.scan_library([str(tmp_path)], recursive=True)

    assert len(tracks) == 1
    assert tracks[0].path == str(Path(track))
    assert tracks[0].key is None
    assert tracks[0].energy == 3

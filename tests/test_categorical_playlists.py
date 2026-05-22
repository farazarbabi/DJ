"""Tests for categorical playlist generation (by_key / by_subgenre)."""

from __future__ import annotations

from dj_registry.models import FileRecord, LogicalTrack
from dj_registry.sync.playlists import (
    _camelot_padded,
    _safe_filename,
    generate_categorical_playlists,
)


def _track(
    track_id: str,
    *,
    file_id: str,
    artist: str = "Art",
    title: str = "Title",
    key: str = "",
    bpm: str = "",
    subgenre: str = "",
) -> LogicalTrack:
    return LogicalTrack(
        track_id=track_id,
        primary_file_id=file_id,
        artist_canonical=artist,
        title_canonical=title,
        canonical_key_camelot=key,
        canonical_bpm=bpm,
        dj_taxonomy_label=subgenre,
    )


def _file(file_id: str, path: str) -> FileRecord:
    return FileRecord(file_id=file_id, path_abs=path)


def _read(path):
    return path.read_text(encoding="utf-8").splitlines()


# ── helpers ────────────────────────────────────────────────────────────────


def test_camelot_padded():
    assert _camelot_padded("1A") == "01A"
    assert _camelot_padded("3a") == "03A"
    assert _camelot_padded("12B") == "12B"
    assert _camelot_padded("10A") == "10A"
    assert _camelot_padded("") is None
    assert _camelot_padded("C#minor") is None
    assert _camelot_padded("13A") is None  # out of Camelot range
    assert _camelot_padded("1C") is None  # only A/B allowed


def test_safe_filename_underscores():
    assert _safe_filename("Organic House Builder") == "Organic_House_Builder"
    assert _safe_filename("Deep/House:Soul") == "Deep_House_Soul"


# ── by_key ─────────────────────────────────────────────────────────────────


def test_by_key_zero_padded_filenames(tmp_path):
    tracks = [_track("t1", file_id="f1", key="3A", bpm="120")]
    files = [_file("f1", "/music/a.aiff")]
    counts = generate_categorical_playlists(tracks, files, str(tmp_path))
    assert counts["by_key"] == 1
    assert (tmp_path / "by_key" / "03A.m3u8").exists()
    assert not (tmp_path / "by_key" / "3A.m3u8").exists()


def test_by_key_skips_missing_key(tmp_path):
    tracks = [_track("t1", file_id="f1", key="", bpm="120")]
    files = [_file("f1", "/music/a.aiff")]
    counts = generate_categorical_playlists(tracks, files, str(tmp_path))
    assert counts["by_key"] == 0
    by_key_dir = tmp_path / "by_key"
    if by_key_dir.exists():
        assert list(by_key_dir.glob("*.m3u8")) == []


# ── by_subgenre ────────────────────────────────────────────────────────────


def test_by_subgenre_underscore_filename(tmp_path):
    tracks = [
        _track("t1", file_id="f1", subgenre="Organic House Builder", bpm="120"),
    ]
    files = [_file("f1", "/music/a.aiff")]
    counts = generate_categorical_playlists(tracks, files, str(tmp_path))
    assert counts["by_subgenre"] == 1
    assert (tmp_path / "by_subgenre" / "Organic_House_Builder.m3u8").exists()


def test_by_subgenre_empty_label_skipped(tmp_path):
    tracks = [_track("t1", file_id="f1", subgenre="", bpm="120")]
    files = [_file("f1", "/music/a.aiff")]
    counts = generate_categorical_playlists(tracks, files, str(tmp_path))
    assert counts["by_subgenre"] == 0


# ── sort + format ──────────────────────────────────────────────────────────


def test_bpm_sort_order_within_playlist(tmp_path):
    tracks = [
        _track("t_124", file_id="f124", key="7A", bpm="124"),
        _track("t_120", file_id="f120", key="7A", bpm="120"),
        _track("t_blank", file_id="fblank", key="7A", bpm=""),
    ]
    files = [
        _file("f124", "/music/124.aiff"),
        _file("f120", "/music/120.aiff"),
        _file("fblank", "/music/blank.aiff"),
    ]
    generate_categorical_playlists(tracks, files, str(tmp_path))
    lines = _read(tmp_path / "by_key" / "07A.m3u8")
    # Lines: [#EXTM3U, #EXTINF t_120, /music/120.aiff, #EXTINF t_124, /music/124.aiff, #EXTINF blank, /music/blank.aiff]
    paths_in_order = [lines[i] for i in (2, 4, 6)]
    assert paths_in_order == [
        "/music/120.aiff",
        "/music/124.aiff",
        "/music/blank.aiff",
    ]


def test_m3u8_format_uses_artist_title_label(tmp_path):
    tracks = [
        _track(
            "t1",
            file_id="f1",
            artist="Sasha",
            title="Xpander",
            key="7A",
            bpm="120",
        ),
    ]
    files = [_file("f1", "/music/xp.aiff")]
    generate_categorical_playlists(tracks, files, str(tmp_path))
    lines = _read(tmp_path / "by_key" / "07A.m3u8")
    assert lines[0] == "#EXTM3U"
    assert lines[1] == "#EXTINF:-1,Sasha - Xpander"
    assert lines[2] == "/music/xp.aiff"


def test_m3u8_falls_back_to_stem_when_no_artist_title(tmp_path):
    tracks = [_track("t1", file_id="f1", artist="", title="", key="7A", bpm="120")]
    files = [_file("f1", "/music/no_meta.aiff")]
    generate_categorical_playlists(tracks, files, str(tmp_path))
    lines = _read(tmp_path / "by_key" / "07A.m3u8")
    assert lines[1] == "#EXTINF:-1,no_meta"


def test_empty_bucket_not_written(tmp_path):
    """Track exists for one sub-genre — the others produce no file."""
    tracks = [_track("t1", file_id="f1", subgenre="Melodic Techno Driver", bpm="120")]
    files = [_file("f1", "/music/a.aiff")]
    generate_categorical_playlists(tracks, files, str(tmp_path))
    sg_dir = tmp_path / "by_subgenre"
    files_written = sorted(p.name for p in sg_dir.glob("*.m3u8"))
    assert files_written == ["Melodic_Techno_Driver.m3u8"]


def test_track_missing_path_is_skipped(tmp_path):
    """A track whose primary_file_id has no matching FileRecord is silently skipped."""
    tracks = [
        _track("t1", file_id="f_missing", key="7A", bpm="120"),
        _track("t2", file_id="f2", key="7A", bpm="121"),
    ]
    files = [_file("f2", "/music/has_file.aiff")]  # f_missing has no record
    counts = generate_categorical_playlists(tracks, files, str(tmp_path))
    assert counts["by_key"] == 1
    lines = _read(tmp_path / "by_key" / "07A.m3u8")
    assert "/music/has_file.aiff" in lines
    assert all("f_missing" not in line for line in lines)

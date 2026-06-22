"""Tests for categorical playlist generation (by_key / by_subgenre)."""

from __future__ import annotations

from dj_registry.models import FileRecord, LogicalTrack
from dj_registry.sync.playlists import (
    _camelot_padded,
    _coarse_key_bucket,
    _coarse_subgenre_bucket,
    _compute_family_first_words,
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
    # utf-8-sig strips the BOM that Rekordbox requires on .m3u8 files.
    return path.read_text(encoding="utf-8-sig").splitlines()


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


# ── coarse helpers ─────────────────────────────────────────────────────────


def test_coarse_key_bucket_groups_four_keys():
    # {1A,1B,2A,2B} → 01A-02B
    assert _coarse_key_bucket("01A") == "01A-02B"
    assert _coarse_key_bucket("01B") == "01A-02B"
    assert _coarse_key_bucket("02A") == "01A-02B"
    assert _coarse_key_bucket("02B") == "01A-02B"
    # {3A,3B,4A,4B} → 03A-04B
    assert _coarse_key_bucket("03A") == "03A-04B"
    assert _coarse_key_bucket("04B") == "03A-04B"
    # {11A,11B,12A,12B} → 11A-12B
    assert _coarse_key_bucket("11A") == "11A-12B"
    assert _coarse_key_bucket("12B") == "11A-12B"


def test_coarse_key_bucket_rejects_invalid():
    assert _coarse_key_bucket("") is None
    assert _coarse_key_bucket("13A") is None
    assert _coarse_key_bucket("XX") is None


def test_family_first_words_threshold():
    labels = [
        "Acid Breaks",
        "Acid House",
        "Acid Tech-House",
        "Indie Dance / Acid",
        "Indie Dance / Cosmic Disco",
        "Indie Dance",
        "Techno",
    ]
    fams = _compute_family_first_words(labels)
    # "Acid" appears in 3 heads (Acid Breaks, Acid House, Acid Tech-House).
    assert "Acid" in fams
    # "Indie" appears only once at first-token position because all three
    # "Indie Dance ..." labels share the same head "Indie Dance" after the
    # split-on-slash collapse — first-word count is 1 distinct head, so it
    # is NOT promoted as a family first-word. Bucketing relies on the
    # slash-merge alone here, which is the intended behaviour.
    assert "Indie" not in fams
    # "Techno" is a single-word head, ignored by the rule.
    assert "Techno" not in fams


def test_coarse_subgenre_collapses_to_first_word():
    labels = [
        "Acid Breaks",
        "Acid House",
        "Acid Tech-House",
        "Indie Dance / Acid",
        "Indie Dance / Electro",
    ]
    fams = _compute_family_first_words(labels)
    assert _coarse_subgenre_bucket("Acid Breaks", fams) == "Acid"
    assert _coarse_subgenre_bucket("Acid House", fams) == "Acid"
    # Slash-head only (no first-word collapse) since "Indie" is unique.
    assert _coarse_subgenre_bucket("Indie Dance / Acid", fams) == "Indie Dance"
    assert _coarse_subgenre_bucket("Indie Dance / Electro", fams) == "Indie Dance"


# ── coarse playlists ───────────────────────────────────────────────────────


def test_coarse_flag_writes_parallel_dirs(tmp_path):
    tracks = [
        _track("t1", file_id="f1", key="1A", bpm="120", subgenre="Acid House"),
        _track("t2", file_id="f2", key="2B", bpm="121", subgenre="Acid Techno"),
        _track("t3", file_id="f3", key="7A", bpm="122", subgenre="Acid Tech-House"),
    ]
    files = [
        _file("f1", "/music/a.aiff"),
        _file("f2", "/music/b.aiff"),
        _file("f3", "/music/c.aiff"),
    ]
    counts = generate_categorical_playlists(tracks, files, str(tmp_path), coarse=True)

    # Fine-grained still written.
    assert (tmp_path / "by_key" / "01A.m3u8").exists()
    assert (tmp_path / "by_key" / "02B.m3u8").exists()
    assert (tmp_path / "by_key" / "07A.m3u8").exists()
    # Coarse buckets: t1+t2 → 01A-02B, t3 → 07A-08B.
    assert (tmp_path / "by_key_coarse" / "01A-02B.m3u8").exists()
    assert (tmp_path / "by_key_coarse" / "07A-08B.m3u8").exists()
    assert counts["by_key_coarse"] == 2

    # Acid* (3 distinct heads) collapse to "Acid" — one coarse subgenre file.
    coarse_sg = sorted(p.name for p in (tmp_path / "by_subgenre_coarse").glob("*.m3u8"))
    assert coarse_sg == ["Acid.m3u8"]
    assert counts["by_subgenre_coarse"] == 1


def test_coarse_only_skips_fine_dirs(tmp_path):
    tracks = [
        _track("t1", file_id="f1", key="1A", bpm="120", subgenre="Acid House"),
        _track("t2", file_id="f2", key="2B", bpm="121", subgenre="Acid Techno"),
    ]
    files = [_file("f1", "/music/a.aiff"), _file("f2", "/music/b.aiff")]
    counts = generate_categorical_playlists(
        tracks, files, str(tmp_path), fine=False, coarse=True
    )

    # Coarse dirs written, fine dirs skipped entirely.
    assert (tmp_path / "by_key_coarse" / "01A-02B.m3u8").exists()
    assert not (tmp_path / "by_key").exists()
    assert not (tmp_path / "by_subgenre").exists()
    assert "by_key" not in counts
    assert "by_subgenre" not in counts
    assert counts["by_key_coarse"] == 1


def test_coarse_flag_off_writes_no_coarse_dirs(tmp_path):
    tracks = [_track("t1", file_id="f1", key="1A", bpm="120", subgenre="Acid House")]
    files = [_file("f1", "/music/a.aiff")]
    counts = generate_categorical_playlists(tracks, files, str(tmp_path))

    assert "by_key_coarse" not in counts
    assert "by_subgenre_coarse" not in counts
    assert not (tmp_path / "by_key_coarse").exists()
    assert not (tmp_path / "by_subgenre_coarse").exists()


def test_coarse_key_bucket_aggregates_all_keys_into_six(tmp_path):
    """Every Camelot key lands in exactly one of the six coarse buckets."""
    tracks = []
    files = []
    for n in range(1, 13):
        for letter in ("A", "B"):
            tid = f"t{n}{letter}"
            fid = f"f{n}{letter}"
            tracks.append(_track(tid, file_id=fid, key=f"{n}{letter}", bpm="120"))
            files.append(_file(fid, f"/music/{tid}.aiff"))

    counts = generate_categorical_playlists(tracks, files, str(tmp_path), coarse=True)
    assert counts["by_key"] == 24
    assert counts["by_key_coarse"] == 6
    expected = {
        "01A-02B.m3u8",
        "03A-04B.m3u8",
        "05A-06B.m3u8",
        "07A-08B.m3u8",
        "09A-10B.m3u8",
        "11A-12B.m3u8",
    }
    written = {p.name for p in (tmp_path / "by_key_coarse").glob("*.m3u8")}
    assert written == expected


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

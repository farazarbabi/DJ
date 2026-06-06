"""Tests for the Spotify playlist fetch-missing feature."""

from dj_tools.spotify_fetch import (
    PlaylistTrack,
    best_match,
    build_ytdlp_command,
    classify_tracks,
    clean_track_name,
    collect_playlists,
    duration_mismatch,
    fetch_missing,
    load_unique_tracks,
    parse_playlist_csv,
    resolve_library_dir,
    sanitize_filename,
    scan_library,
    tokens,
)

CSV_HEADER = (
    "Track URI,Track Name,Album Name,Artist Name(s),Release Date,"
    "Duration (ms),Popularity,Explicit,Added By,Added At,Genres,Record Label\n"
)


def _row(uri, name, artists, ms):
    return (
        f"spotify:track:{uri},\"{name}\",\"Album\",\"{artists}\","
        f"2020-01-01,{ms},50,false,user,2020-01-01T00:00:00Z,\"\",\"Label\"\n"
    )


def _write_csv(path, rows):
    path.write_text(CSV_HEADER + "".join(rows), encoding="utf-8-sig")
    return str(path)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def test_clean_track_name_parenthesizes_remix():
    assert clean_track_name("Kryptonite - Mateo! Remix") == "Kryptonite (Mateo! Remix)"
    assert clean_track_name("Skyfall") == "Skyfall"


def test_sanitize_filename_strips_illegal_chars():
    assert sanitize_filename('A/B: C?"<>|*') == "AB C"
    assert sanitize_filename("trailing dots...") == "trailing dots"


def test_tokens_drops_stopwords_and_accents():
    assert tokens("The Extended Mix") == set()  # all stopwords
    assert "merci" in tokens("Merci Éclair")
    assert "eclair" in tokens("Merci Éclair")


# --------------------------------------------------------------------------- #
# Playlist parsing
# --------------------------------------------------------------------------- #
def test_parse_playlist_csv(tmp_path):
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("aaa", "Let Go", "Oliver Koletzki;Temple Haze", 254845),
    ])
    tracks = parse_playlist_csv(csv_path)
    assert len(tracks) == 1
    t = tracks[0]
    assert t.name == "Let Go"
    assert t.artists == ["Oliver Koletzki", "Temple Haze"]
    assert t.primary_artist == "Oliver Koletzki"
    assert abs(t.duration_sec - 254.845) < 1e-6


def test_target_basename_and_query():
    t = PlaylistTrack(name="Starlings - Henry Saiz Remix", artists=["NTO"])
    assert t.target_basename() == "NTO - Starlings (Henry Saiz Remix)"
    assert t.search_query() == "NTO Starlings - Henry Saiz Remix"


def test_load_unique_tracks_dedupes_by_uri(tmp_path):
    a = _write_csv(tmp_path / "a.csv", [_row("dup", "Song", "Artist", 200000)])
    b = _write_csv(tmp_path / "b.csv", [
        _row("dup", "Song", "Artist", 200000),
        _row("other", "Other", "Artist", 200000),
    ])
    tracks = load_unique_tracks([a, b])
    assert len(tracks) == 2


def test_collect_playlists_expands_directory(tmp_path):
    _write_csv(tmp_path / "one.csv", [_row("x", "X", "A", 1000)])
    _write_csv(tmp_path / "two.csv", [_row("y", "Y", "A", 1000)])
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    found = collect_playlists([str(tmp_path)])
    assert len(found) == 2
    assert all(p.endswith(".csv") for p in found)


# --------------------------------------------------------------------------- #
# Library matching
# --------------------------------------------------------------------------- #
def _make_library(tmp_path, names):
    for n in names:
        (tmp_path / n).write_bytes(b"\x00")
    return scan_library(tmp_path)


def test_best_match_present_track(tmp_path):
    index = _make_library(tmp_path, ["Adele - Skyfall (Original Mix).aiff"])
    path, score = best_match(PlaylistTrack(name="Skyfall", artists=["Adele"]), index)
    assert score >= 0.9
    assert path.endswith("Adele - Skyfall (Original Mix).aiff")


def test_best_match_rejects_same_title_different_artist(tmp_path):
    """Two unrelated 'Kryptonite' tracks must not match on title alone."""
    index = _make_library(
        tmp_path, ["Cristhian Valencia - Kryptonite (Original Mix).aiff"]
    )
    _path, score = best_match(
        PlaylistTrack(name="Kryptonite", artists=["3 Doors Down"]), index
    )
    assert score < 0.62


def test_best_match_alternate_remix_below_threshold(tmp_path):
    """Owning the Original Mix should not count as having a specific remix."""
    index = _make_library(tmp_path, ["NTO - Starlings (Original Mix).aiff"])
    _path, score = best_match(
        PlaylistTrack(name="Starlings - Henry Saiz Remix", artists=["NTO"]), index
    )
    assert score < 0.62


def test_best_match_artistless_filename_distinctive_title(tmp_path):
    """A 3+ token title can match a file that has no artist prefix."""
    index = _make_library(tmp_path, ["To The Sun (Deep House).wav"])
    _path, score = best_match(
        PlaylistTrack(name="To The Sun - Deep House", artists=["fluffysnaff"]), index
    )
    assert score >= 0.62


def test_classify_tracks_splits_present_and_missing(tmp_path):
    index = _make_library(tmp_path, ["Adele - Skyfall (Original Mix).aiff"])
    tracks = [
        PlaylistTrack(name="Skyfall", artists=["Adele"]),
        PlaylistTrack(name="Free Babe", artists=["Elodie Gervaise"]),
    ]
    present, missing = classify_tracks(tracks, index)
    assert [r.track.name for r in present] == ["Skyfall"]
    assert [r.track.name for r in missing] == ["Free Babe"]


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def test_build_ytdlp_command():
    cmd = build_ytdlp_command("NTO Starlings", "/out/%(ext)s",
                              min_duration=30, max_duration=900)
    assert cmd[0] == "yt-dlp"
    assert "ytsearch5:NTO Starlings" in cmd
    assert "duration < 900 & duration > 30" in cmd
    assert "wav" in cmd


def test_duration_mismatch_flags_large_gap():
    # 408s vs 223s -> flagged.
    assert duration_mismatch(223, 408) is not None
    # 234s vs 238s -> within tolerance.
    assert duration_mismatch(234, 238) is None
    # Short absolute gap on a short track stays unflagged.
    assert duration_mismatch(200, 215) is None
    # Missing data -> no judgement.
    assert duration_mismatch(None, 300) is None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def test_resolve_library_dir(tmp_path):
    f = tmp_path / "track.aiff"
    f.write_bytes(b"\x00")
    assert resolve_library_dir(str(tmp_path)) == str(tmp_path)
    assert resolve_library_dir(str(f)) == str(tmp_path)


def test_fetch_missing_dry_run_writes_reports(tmp_path):
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Adele - Skyfall (Original Mix).aiff").write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Skyfall", "Adele", 286000),
        _row("b", "Free Babe", "Elodie Gervaise", 200000),
    ])

    summary = fetch_missing([csv_path], str(library), dry_run=True)

    assert summary["total"] == 2
    assert summary["present"] == 1
    assert summary["missing"] == 1
    assert summary["downloaded"] == 0  # dry run downloads nothing
    report_dir = tmp_path / "lib" / "outputs" / "fetch"
    assert (report_dir / "missing_report.csv").exists()
    assert (report_dir / "matched_report.csv").exists()
    # No network tools invoked, so no download log on a dry run.
    assert not (report_dir / "download_log.csv").exists()


def test_fetch_missing_bad_library_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        fetch_missing([], str(tmp_path / "nope"))

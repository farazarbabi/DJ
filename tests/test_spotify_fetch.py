"""Tests for the Spotify playlist fetch-missing feature."""

import os

from dj_tools.spotify_fetch import (
    PlaylistTrack,
    best_match,
    build_candidate_command,
    build_download_command,
    classify_tracks,
    clean_track_name,
    collect_playlists,
    default_playlists_dir,
    duration_mismatch,
    fetch_missing,
    generate_spotify_playlists,
    load_unique_tracks,
    parse_candidate_lines,
    parse_playlist_csv,
    prune_superseded_downloads,
    resolve_library_dir,
    sanitize_filename,
    scan_library,
    select_candidates,
    tokens,
    tool_file_for,
    version_key,
)
from dj_tools import spotify_fetch as sf
from dj_tools import soundeo as so

CSV_HEADER = (
    "Track URI,Track Name,Album Name,Artist Name(s),Release Date,"
    "Duration (ms),Popularity,Explicit,Added By,Added At,Genres,Record Label\n"
)


def _row(uri, name, artists, ms, added_at="2020-01-01T00:00:00Z"):
    return (
        f"spotify:track:{uri},\"{name}\",\"Album\",\"{artists}\","
        f"2020-01-01,{ms},50,false,user,{added_at},\"\",\"Label\"\n"
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
    # Source markers are appended verbatim after the clean stem.
    assert t.target_basename("[U]") == "NTO - Starlings (Henry Saiz Remix)[U]"
    assert t.target_basename("[W]") == "NTO - Starlings (Henry Saiz Remix)[W]"
    assert t.search_query() == "NTO Starlings - Henry Saiz Remix"


def test_parse_playlist_csv_captures_metadata(tmp_path):
    row = (
        'spotify:track:m,"Enigma","Pure Bliss EP","Leo Janeiro;Hauy",'
        '2018-06-01,422000,40,false,u,2020-01-01T00:00:00Z,'
        '"jazz house,deep house","Get Physical Music"\n'
    )
    tracks = parse_playlist_csv(_write_csv(tmp_path / "p.csv", [row]))
    t = tracks[0]
    assert t.album == "Pure Bliss EP"
    assert t.year == "2018"
    assert t.genres == ["jazz house", "deep house"]
    assert t.primary_genre == "jazz house"
    assert t.label == "Get Physical Music"
    assert t.title_tag() == "Enigma"


def test_load_unique_tracks_dedupes_by_uri(tmp_path):
    a = _write_csv(tmp_path / "a.csv", [_row("dup", "Song", "Artist", 200000)])
    b = _write_csv(tmp_path / "b.csv", [
        _row("dup", "Song", "Artist", 200000),
        _row("other", "Other", "Artist", 200000),
    ])
    tracks = load_unique_tracks([a, b])
    assert len(tracks) == 2


def test_load_unique_tracks_orders_newest_added_first(tmp_path):
    # Fetch order is "Added At" descending so the most recently added tracks
    # download first (and get Soundeo quota priority); undated rows sort last.
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Old", "Artist", 200000, added_at="2019-01-01T00:00:00Z"),
        _row("b", "New", "Artist", 200000, added_at="2024-05-01T00:00:00Z"),
        _row("c", "Mid", "Artist", 200000, added_at="2021-09-01T00:00:00Z"),
        _row("d", "Undated", "Artist", 200000, added_at=""),
    ])
    names = [t.name for t in load_unique_tracks([csv_path])]
    assert names == ["New", "Mid", "Old", "Undated"]


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
def test_build_candidate_command():
    cmd = build_candidate_command("NTO Starlings", count=5,
                                  min_duration=30, max_duration=900)
    assert cmd[0] == "yt-dlp"
    assert "--skip-download" in cmd
    assert "ytsearch5:NTO Starlings" in cmd
    assert "duration < 900 & duration > 30" in cmd


def test_build_download_command_targets_specific_video():
    cmd = build_download_command("abc123", "/out/%(ext)s")
    assert "https://www.youtube.com/watch?v=abc123" in cmd
    assert "wav" in cmd
    assert "/out/%(ext)s" in cmd


def test_parse_candidate_lines_handles_missing_duration():
    text = "id1\t210.0\tTitle One\nid2\tNA\tTitle Two\nid3\t300\tThree\n"
    rows = parse_candidate_lines(text)
    assert rows[0] == ("id1", 210.0, "Title One")
    assert rows[1] == ("id2", None, "Title Two")
    assert rows[2] == ("id3", 300.0, "Three")


def test_select_candidates_keeps_within_tolerance_closest_first():
    cands = [
        ("far", 250.0, "x"),     # 30s off
        ("close", 222.0, "x"),   # 2s off  -> within 3s
        ("none", None, "x"),     # unknown duration -> excluded
        ("edge", 217.0, "x"),    # 3s off  -> within 3s (boundary)
    ]
    assert select_candidates(cands, expected=220.0, tolerance=3.0) == ["close", "edge"]


def test_select_candidates_unknown_expected_keeps_order():
    cands = [("a", 100.0, "x"), ("b", None, "x")]
    assert select_candidates(cands, expected=None) == ["a", "b"]


def test_select_candidates_none_within_tolerance():
    cands = [("a", 300.0, "x"), ("b", 100.0, "x")]
    assert select_candidates(cands, expected=220.0, tolerance=3.0) == []


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


def test_fetch_missing_treats_extended_variant_as_present(tmp_path):
    # Spotify lists the untagged title; library only has the Extended Mix.
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Iorie - Matter of Fact (Extended Mix).aiff").write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Matter of Fact", "Iorie", 357000),
    ])

    summary = fetch_missing([csv_path], str(library), dry_run=True)

    assert summary["present"] == 1
    assert summary["missing"] == 0
    assert summary["downloaded"] == 0


def test_default_playlists_dir():
    assert default_playlists_dir(os.path.join("D:", "Music")) == os.path.join(
        "D:", "Music", "spotify-playlists"
    )


def test_fetch_missing_bad_library_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        fetch_missing([], str(tmp_path / "nope"))


def test_tool_file_for_matches_only_marked_naming(tmp_path):
    track = PlaylistTrack(name="Starlings - Henry Saiz Remix", artists=["NTO"])
    # A differently-named user file is ignored.
    (tmp_path / "NTO - Starlings (Original Mix).aiff").write_bytes(b"\x00")
    assert tool_file_for(track, str(tmp_path)) is None
    # An unmarked file at the exact target name is the user's curated original
    # (standard library naming), NOT a tool download -> must be ignored, or
    # re-verify would re-download it to the [U] path and leave a duplicate.
    (tmp_path / "NTO - Starlings (Henry Saiz Remix).aiff").write_bytes(b"\x00")
    assert tool_file_for(track, str(tmp_path)) is None
    # Marked names count as tool downloads.
    tool = tmp_path / "NTO - Starlings (Henry Saiz Remix)[W].aiff"
    tool.write_bytes(b"\x00")
    assert tool_file_for(track, str(tmp_path)) == str(tool)


def test_tool_file_for_prefers_marked_name(tmp_path):
    track = PlaylistTrack(name="Skyfall", artists=["Adele"])
    marked = tmp_path / "Adele - Skyfall[U].aiff"
    marked.write_bytes(b"\x00")
    assert tool_file_for(track, str(tmp_path)) == str(marked)


def test_prune_superseded_downloads_removes_marked_when_original_exists(tmp_path):
    orig = tmp_path / "Elodie Gervaise - Free Babe.aiff"
    marked = tmp_path / "Elodie Gervaise - Free Babe[U].aiff"
    orig.write_bytes(b"\x00")
    marked.write_bytes(b"\x00")
    removed = prune_superseded_downloads(str(tmp_path))
    assert removed == [str(marked)]
    assert not marked.exists()
    assert orig.exists()  # the curated original is kept


def test_prune_superseded_downloads_removes_soundeo_wav_marker(tmp_path):
    orig = tmp_path / "Artist - Song.aiff"
    marked = tmp_path / "Artist - Song[W].aiff"
    orig.write_bytes(b"\x00")
    marked.write_bytes(b"\x00")
    assert prune_superseded_downloads(str(tmp_path)) == [str(marked)]
    assert not marked.exists()


def test_prune_superseded_downloads_matches_across_formats(tmp_path):
    orig = tmp_path / "Artist - Song.aiff"
    marked = tmp_path / "Artist - Song[U].wav"  # different ext still superseded
    orig.write_bytes(b"\x00")
    marked.write_bytes(b"\x00")
    removed = prune_superseded_downloads(str(tmp_path))
    assert removed == [str(marked)]
    assert not marked.exists()


def test_version_key_ignores_original_extended_suffixes():
    base = version_key("Elodie Gervaise - Free Babe")
    assert version_key("Elodie Gervaise - Free Babe (Original Mix)") == base
    assert version_key("Elodie Gervaise - Free Babe (Original Version)") == base
    assert version_key("Elodie Gervaise - Free Babe (Extended Mix)") == base
    assert version_key("Elodie Gervaise - Free Babe (Extended Version)") == base
    assert version_key("Elodie Gervaise - Free Babe [Extended]") == base
    # A true remix is a distinct track and keeps its own identity.
    assert version_key("Elodie Gervaise - Free Babe (Henry Saiz Remix)") != base
    # The artist/title separator dash is never stripped.
    assert version_key("Artist - Some Title") == "artist - some title"


def test_prune_superseded_downloads_removes_when_variant_original_added(tmp_path):
    # User fetched a [U] copy, later added a curated "(Extended Mix)" original.
    marked = tmp_path / "Iorie - Matter of Fact[U].aiff"
    variant = tmp_path / "Iorie - Matter of Fact (Extended Mix).aiff"
    marked.write_bytes(b"\x00")
    variant.write_bytes(b"\x00")
    assert prune_superseded_downloads(str(tmp_path)) == [str(marked)]
    assert not marked.exists()
    assert variant.exists()


def test_prune_superseded_downloads_ignores_unrelated_remix(tmp_path):
    marked = tmp_path / "Iorie - Matter of Fact[U].aiff"
    remix = tmp_path / "Iorie - Matter of Fact (Some Remix).aiff"
    marked.write_bytes(b"\x00")
    remix.write_bytes(b"\x00")
    # A remix is a different track, so the [U] copy is NOT superseded.
    assert prune_superseded_downloads(str(tmp_path)) == []
    assert marked.exists()


def test_prune_superseded_downloads_removes_extra_remixer_artist_credit(tmp_path):
    marked = tmp_path / (
        "Elias Dor" + "\u00e9" + ", Sarah Mon" + "\u00ed"
        + ", Sydka - Disappear (Sydka Remix)[U].aiff"
    )
    original = tmp_path / (
        "Elias Dore, Sarah Mon" + "\u00ed"
        + " - Disappear (Sydka Remix).aiff"
    )
    marked.write_bytes(b"\x00")
    original.write_bytes(b"\x00")

    assert prune_superseded_downloads(str(tmp_path)) == [str(marked)]
    assert not marked.exists()
    assert original.exists()


def test_prune_superseded_downloads_keeps_extra_artist_not_in_remix_title(tmp_path):
    marked = tmp_path / "Artist, Different Artist - Song (Known Remix)[U].aiff"
    original = tmp_path / "Artist - Song (Known Remix).aiff"
    marked.write_bytes(b"\x00")
    original.write_bytes(b"\x00")

    assert prune_superseded_downloads(str(tmp_path)) == []
    assert marked.exists()


def test_prune_superseded_downloads_keeps_marked_without_original(tmp_path):
    marked = tmp_path / "Artist - Song[U].aiff"
    marked.write_bytes(b"\x00")
    assert prune_superseded_downloads(str(tmp_path)) == []
    assert marked.exists()


def test_prune_superseded_downloads_dry_run_reports_without_deleting(tmp_path):
    orig = tmp_path / "Artist - Song.aiff"
    marked = tmp_path / "Artist - Song[U].aiff"
    orig.write_bytes(b"\x00")
    marked.write_bytes(b"\x00")
    removed = prune_superseded_downloads(str(tmp_path), dry_run=True)
    assert removed == [str(marked)]
    assert marked.exists()  # dry run leaves files in place


# --------------------------------------------------------------------------- #
# Representative per-playlist M3U8s
# --------------------------------------------------------------------------- #
def _read_playlist(path):
    text = path.read_text(encoding="utf-8-sig")  # transparently strips the BOM
    return text.splitlines()


def test_generate_spotify_playlists_writes_one_per_csv_in_order(tmp_path):
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Adele - Skyfall (Original Mix).aiff").write_bytes(b"\x00")
    (library / "Iorie - Matter of Fact (Extended Mix).aiff").write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "Sunset Set.csv", [
        _row("a", "Matter of Fact", "Iorie", 357000),
        _row("b", "Skyfall", "Adele", 286000),
    ])

    summary = generate_spotify_playlists([csv_path], str(library))

    out = library / "outputs" / "playlists" / "spotify" / "Sunset Set.m3u8"
    assert out.exists()
    assert summary == {"playlists_written": 1, "tracks_added": 2, "tracks_skipped": 0}

    lines = _read_playlist(out)
    assert lines[0] == "#EXTM3U"
    paths = [ln for ln in lines if not ln.startswith("#")]
    # CSV order preserved: Matter of Fact before Skyfall.
    assert paths[0].endswith("Iorie - Matter of Fact (Extended Mix).aiff")
    assert paths[1].endswith("Adele - Skyfall (Original Mix).aiff")
    assert all(os.path.isabs(p) for p in paths)
    assert lines.count("#EXTM3U") == 1
    assert sum(1 for ln in lines if ln.startswith("#EXTINF")) == 2


def test_generate_spotify_playlists_has_utf8_bom(tmp_path):
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Adele - Skyfall (Original Mix).aiff").write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Skyfall", "Adele", 286000)])

    generate_spotify_playlists([csv_path], str(library))

    raw = (library / "outputs" / "playlists" / "spotify" / "p.m3u8").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # Rekordbox requires the UTF-8 BOM


def test_generate_spotify_playlists_skips_unresolvable(tmp_path):
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Adele - Skyfall (Original Mix).aiff").write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Skyfall", "Adele", 286000),
        _row("b", "Free Babe", "Elodie Gervaise", 200000),  # not in library
    ])

    summary = generate_spotify_playlists([csv_path], str(library))

    assert summary["tracks_added"] == 1
    assert summary["tracks_skipped"] == 1
    paths = [ln for ln in _read_playlist(
        library / "outputs" / "playlists" / "spotify" / "p.m3u8"
    ) if not ln.startswith("#")]
    assert len(paths) == 1
    assert paths[0].endswith("Adele - Skyfall (Original Mix).aiff")


def test_generate_spotify_playlists_resolves_downloaded_marked_file(tmp_path):
    # A just-downloaded [U] file resolves into its playlist (marker is matcher-invisible).
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Elodie Gervaise - Free Babe[U].aiff").write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Free Babe", "Elodie Gervaise", 200000),
    ])

    summary = generate_spotify_playlists([csv_path], str(library))

    assert summary["tracks_added"] == 1
    paths = [ln for ln in _read_playlist(
        library / "outputs" / "playlists" / "spotify" / "p.m3u8"
    ) if not ln.startswith("#")]
    assert paths[0].endswith("Elodie Gervaise - Free Babe[U].aiff")


def test_generate_spotify_playlists_dedupes_repeated_tracks(tmp_path):
    # A Spotify export can list the same track twice, and two rows (an alternate
    # title/version) can resolve to the same library file — each file must
    # appear at most once per playlist so Rekordbox has no duplicate entries.
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Adele - Skyfall (Original Mix).aiff").write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Skyfall", "Adele", 286000),
        _row("b", "Skyfall", "Adele", 286000),          # exact duplicate row
        _row("c", "Skyfall (Extended Mix)", "Adele", 286000),  # variant → same file
    ])

    summary = generate_spotify_playlists([csv_path], str(library))

    assert summary["tracks_added"] == 1
    assert summary["tracks_skipped"] == 0  # the extra rows are dups, not misses
    paths = [ln for ln in _read_playlist(
        library / "outputs" / "playlists" / "spotify" / "p.m3u8"
    ) if not ln.startswith("#")]
    assert paths == [p for p in paths if p.endswith("Adele - Skyfall (Original Mix).aiff")]
    assert len(paths) == 1


def test_generate_spotify_playlists_sorts_by_bpm_with_key_tiebreaker(tmp_path):
    # Playlists sort by BPM (ascending), with Camelot key as tiebreaker.
    # Files without tags sort by infinity, maintaining CSV order within that tier.
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Adele - Skyfall (Original Mix).aiff").write_bytes(b"\x00")
    (library / "Iorie - Matter of Fact (Extended Mix).aiff").write_bytes(b"\x00")
    (library / "Baime - Satara (Original Mix).aiff").write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Skyfall", "Adele", 286000, added_at="2021-06-01T00:00:00Z"),
        _row("b", "Satara", "Baime", 300000, added_at="2023-01-15T00:00:00Z"),
        _row("c", "Matter of Fact", "Iorie", 357000, added_at="2019-03-10T00:00:00Z"),
    ])

    generate_spotify_playlists([csv_path], str(library))

    paths = [ln for ln in _read_playlist(
        library / "outputs" / "playlists" / "spotify" / "p.m3u8"
    ) if not ln.startswith("#")]
    # No tags on test files, so all sort by BPM=infinity; CSV order preserved
    assert paths[0].endswith("Adele - Skyfall (Original Mix).aiff")
    assert paths[1].endswith("Baime - Satara (Original Mix).aiff")
    assert paths[2].endswith("Iorie - Matter of Fact (Extended Mix).aiff")


def test_generate_spotify_playlists_removes_tracks_when_csv_updated(tmp_path):
    # When a track is removed from the CSV, it must be removed from the playlist.
    # CSV v1: Tracks A, B, C → Playlist has A, B, C
    # CSV v2: Tracks A, B (C removed) → Playlist must have only A, B
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Adele - Skyfall (Original Mix).aiff").write_bytes(b"\x00")
    (library / "Iorie - Matter of Fact (Extended Mix).aiff").write_bytes(b"\x00")
    (library / "Baime - Satara (Original Mix).aiff").write_bytes(b"\x00")

    # Initial CSV with 3 tracks
    csv_path = tmp_path / "p.csv"
    _write_csv(csv_path, [
        _row("a", "Skyfall", "Adele", 286000),
        _row("b", "Matter of Fact", "Iorie", 357000),
        _row("c", "Satara", "Baime", 300000),
    ])

    generate_spotify_playlists([csv_path], str(library))
    playlist_path = library / "outputs" / "playlists" / "spotify" / "p.m3u8"
    initial_paths = [ln for ln in _read_playlist(playlist_path) if not ln.startswith("#")]
    assert len(initial_paths) == 3

    # Updated CSV: remove Satara (Baime track)
    _write_csv(csv_path, [
        _row("a", "Skyfall", "Adele", 286000),
        _row("b", "Matter of Fact", "Iorie", 357000),
    ])

    generate_spotify_playlists([csv_path], str(library))
    updated_paths = [ln for ln in _read_playlist(playlist_path) if not ln.startswith("#")]

    # Playlist must now have only 2 tracks; Satara removed
    assert len(updated_paths) == 2
    assert all(not p.endswith("Baime - Satara (Original Mix).aiff") for p in updated_paths)
    assert any(p.endswith("Adele - Skyfall (Original Mix).aiff") for p in updated_paths)
    assert any(p.endswith("Iorie - Matter of Fact (Extended Mix).aiff") for p in updated_paths)


def test_playlist_stem_drops_csv_and_sanitizes():
    from dj_tools.spotify_fetch import _playlist_stem

    # ".csv" extension dropped, spaces kept.
    assert _playlist_stem("Sunset Set.csv") == "Sunset Set"
    # Illegal Windows filename chars stripped (these can't appear in a real CSV
    # filename on Windows, but a name may arrive sanitization-needing).
    assert _playlist_stem('House/Techno.csv') == "HouseTechno"
    assert _playlist_stem("") == "playlist"


def test_fetch_missing_dry_run_generates_playlists(tmp_path):
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Adele - Skyfall (Original Mix).aiff").write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Skyfall", "Adele", 286000),
        _row("b", "Free Babe", "Elodie Gervaise", 200000),
    ])

    summary = fetch_missing([csv_path], str(library), dry_run=True)

    assert summary["playlists_written"] == 1
    out = library / "outputs" / "playlists" / "spotify" / "p.m3u8"
    assert out.exists()
    paths = [ln for ln in _read_playlist(out) if not ln.startswith("#")]
    assert paths == [p for p in paths if p.endswith("Adele - Skyfall (Original Mix).aiff")]


def test_embed_metadata_writes_id3_tags(tmp_path):
    import numpy as np
    import soundfile as sf
    from mutagen.aiff import AIFF

    from dj_tools.spotify_fetch import _embed_metadata

    aiff = tmp_path / "track.aiff"
    sf.write(str(aiff), np.zeros(22050, dtype="float32"), 22050)

    track = PlaylistTrack(
        name="Every You - Erly Tepshi Remix",
        artists=["Rafael Cerato", "Jager"],
        album="Every You (Remixes)", year="2023",
        genres=["melodic techno"], label="Get Physical Music",
    )
    _embed_metadata(str(aiff), track)

    tags = AIFF(str(aiff)).tags
    assert str(tags.get("TIT2")) == "Every You (Erly Tepshi Remix)"
    assert str(tags.get("TPE1")) == "Rafael Cerato, Jager"
    assert str(tags.get("TPE2")) == "Rafael Cerato"
    assert str(tags.get("TALB")) == "Every You (Remixes)"
    assert str(tags.get("TCON")) == "melodic techno"
    assert str(tags.get("TDRC")) == "2023"
    assert str(tags.get("TPUB")) == "Get Physical Music"


# --------------------------------------------------------------------------- #
# Source routing: Soundeo (primary) -> YouTube (fallback)
# --------------------------------------------------------------------------- #
class _FakeSoundeo:
    """Stand-in for SoundeoClient: serves a single optional result."""

    audio_format = "aiff"

    def __init__(self, result=None, quota_exhausted=False):
        self.result = result
        self.quota_exhausted = quota_exhausted
        self.download_calls = 0
        self.closed = False

    def login(self):
        pass

    def close(self):
        self.closed = True

    def search(self, track):
        return [self.result] if self.result else []

    def pick(self, track, results):
        return results[0] if results else None

    def download(self, result, dest, **kw):
        self.download_calls += 1
        if self.quota_exhausted:
            raise so.SoundeoQuotaExceeded("quota")
        with open(dest, "wb") as f:
            f.write(b"AIFFdata")


def _yt_stub(monkeypatch):
    def fake_dl(track, dest_dir, **kw):
        p = os.path.join(dest_dir, track.target_basename(sf.SOURCE_MARKER) + ".aiff")
        with open(p, "wb") as f:
            f.write(b"YT")
        return sf.DownloadOutcome("ok", p, "youtube-stub", "youtube")
    monkeypatch.setattr(sf, "download_track", fake_dl)


def test_acquire_prefers_soundeo_unmarked(tmp_path, monkeypatch):
    _yt_stub(monkeypatch)
    track = PlaylistTrack(name="Tune", artists=["Artist"], duration_sec=200)
    fake = _FakeSoundeo(so.SoundeoResult("1", "Artist", "Tune", 200, formats=["aiff"]))
    out = sf.acquire_track(track, str(tmp_path), soundeo=fake, quota=sf._QuotaState())
    assert out.status == "ok" and out.source == "soundeo"
    assert (tmp_path / "Artist - Tune.aiff").exists()       # unmarked
    assert not (tmp_path / "Artist - Tune[U].aiff").exists()


def test_acquire_soundeo_keeps_soundeo_filename(tmp_path, monkeypatch):
    # Spotify titles the track "- Radio Edit"; Soundeo lists "(Original Mix)".
    # The downloaded file must keep Soundeo's own name, not the Spotify one.
    _yt_stub(monkeypatch)
    track = PlaylistTrack(name="Diclofél - Radio Edit", artists=["Julian Schraven"],
                          duration_sec=200)
    fake = _FakeSoundeo(so.SoundeoResult(
        "1", "Julian Schraven", "Diclofél (Original Mix)", 420, formats=["aiff"]))
    out = sf.acquire_track(track, str(tmp_path), soundeo=fake, quota=sf._QuotaState())
    assert out.status == "ok" and out.source == "soundeo"
    assert (tmp_path / "Julian Schraven - Diclofél (Original Mix).aiff").exists()
    assert not (tmp_path / "Julian Schraven - Diclofél (Radio Edit).aiff").exists()


def test_acquire_uses_soundeo_wav_before_youtube(tmp_path, monkeypatch):
    yt_calls = []
    monkeypatch.setattr(sf, "download_track", lambda *a, **k: yt_calls.append(a))

    def fake_convert(src, dest):
        with open(dest, "wb") as f:
            f.write(b"AIFF-from-wav")
        return True

    monkeypatch.setattr(sf, "_audio_to_aiff", fake_convert)
    track = PlaylistTrack(name="NahNah - Remix", artists=["Ka:lu", "Sydka"], duration_sec=470)
    fake = _FakeSoundeo(so.SoundeoResult("1", "Ka:lu", "NahNah (Remix)", 469,
                                         formats=["mp3", "wav"]))
    out = sf.acquire_track(track, str(tmp_path), soundeo=fake, quota=sf._QuotaState())

    assert out.status == "ok" and out.source == "soundeo"
    assert out.detail == "soundeo:1:wav"
    assert (tmp_path / "Kalu - NahNah (Remix)[W].aiff").exists()
    assert not (tmp_path / "Kalu - NahNah (Remix)[W].__soundeo__.wav").exists()
    assert yt_calls == []


def test_acquire_falls_back_to_youtube_when_absent(tmp_path, monkeypatch):
    _yt_stub(monkeypatch)
    track = PlaylistTrack(name="Tune", artists=["Artist"], duration_sec=200)
    fake = _FakeSoundeo(result=None)  # not on Soundeo
    out = sf.acquire_track(track, str(tmp_path), soundeo=fake, quota=sf._QuotaState())
    assert out.source == "youtube"
    assert (tmp_path / "Artist - Tune[U].aiff").exists()    # marked


def test_acquire_quota_skip_and_no_further_attempts(tmp_path, monkeypatch):
    _yt_stub(monkeypatch)
    quota = sf._QuotaState()
    fake = _FakeSoundeo(
        so.SoundeoResult("1", "Artist", "Tune", 200, formats=["aiff"]),
        quota_exhausted=True,
    )
    t1 = PlaylistTrack(name="Tune", artists=["Artist"], duration_sec=200)
    out1 = sf.acquire_track(t1, str(tmp_path), soundeo=fake, quota=quota)
    assert out1.status == "quota_skip" and quota.exhausted
    # Second on-Soundeo track: already exhausted -> immediate skip, no download.
    t2 = PlaylistTrack(name="Other", artists=["Artist"], duration_sec=200)
    fake.result = so.SoundeoResult("2", "Artist", "Other", 200, formats=["aiff"])
    out2 = sf.acquire_track(t2, str(tmp_path), soundeo=fake, quota=quota)
    assert out2.status == "quota_skip"
    assert fake.download_calls == 1  # not attempted again after exhaustion


def test_acquire_without_soundeo_uses_youtube(tmp_path, monkeypatch):
    _yt_stub(monkeypatch)
    track = PlaylistTrack(name="Tune", artists=["Artist"], duration_sec=200)
    out = sf.acquire_track(track, str(tmp_path), soundeo=None, quota=sf._QuotaState())
    assert out.source == "youtube"


def test_fetch_missing_routes_soundeo_then_youtube(tmp_path, monkeypatch):
    _yt_stub(monkeypatch)
    library = tmp_path / "lib"
    library.mkdir()
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "On Soundeo", "Artist A", 200000),
        _row("b", "Only Youtube", "Artist B", 180000),
    ])

    def fake_from_env(cls, **kw):
        return _FakeSoundeo(
            so.SoundeoResult("s1", "Artist A", "On Soundeo", 200, formats=["aiff"]))

    # Serve a Soundeo hit only for the "On Soundeo" track.
    fake = _FakeSoundeo(
        so.SoundeoResult("s1", "Artist A", "On Soundeo", 200, formats=["aiff"]))
    orig_search = fake.search
    fake.search = lambda track: orig_search(track) if "soundeo" in track.name.lower() else []
    monkeypatch.setattr(so.SoundeoClient, "from_env", classmethod(lambda cls, **kw: fake))

    summary = fetch_missing([csv_path], str(library), audio_format="aiff")

    assert summary["soundeo"] == 1
    assert summary["youtube"] == 1
    assert (library / "Artist A - On Soundeo.aiff").exists()        # unmarked Soundeo
    assert (library / "Artist B - Only Youtube[U].aiff").exists()   # YouTube fallback
    assert fake.closed
    log = (library / "outputs" / "fetch" / "download_log.csv").read_text(encoding="utf-8")
    assert "soundeo" in log and "youtube" in log


def test_fetch_missing_aborts_on_soundeo_login_failure(tmp_path, monkeypatch):
    import pytest
    _yt_stub(monkeypatch)  # present only to prove it is NOT used
    library = tmp_path / "lib"
    library.mkdir()
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Tune", "Artist", 200000)])

    class FailClient:
        def login(self):
            raise so.SoundeoAuthError("bad creds")

        def close(self):
            pass

    monkeypatch.setattr(so.SoundeoClient, "from_env",
                        classmethod(lambda cls, **kw: FailClient()))

    with pytest.raises(RuntimeError, match="Soundeo login failed"):
        fetch_missing([csv_path], str(library), audio_format="aiff")
    # No automatic YouTube fallback on login failure.
    assert not (library / "Artist - Tune[U].aiff").exists()


def test_fetch_missing_stops_on_quota_without_youtube(tmp_path, monkeypatch):
    _yt_stub(monkeypatch)  # would create [U] files if (wrongly) used
    library = tmp_path / "lib"
    library.mkdir()
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "First", "Artist A", 200000),
        _row("b", "Second", "Artist B", 200000),
        _row("c", "Third", "Artist C", 200000),
    ])

    class QuotaClient:
        audio_format = "aiff"

        def login(self):
            pass

        def close(self):
            pass

        def search(self, track):
            return [so.SoundeoResult("x", track.primary_artist, track.name, 200,
                                     formats=["aiff"])]

        def pick(self, track, results):
            return results[0]

        def download(self, result, dest):
            raise so.SoundeoQuotaExceeded("daily limit reached")

    monkeypatch.setattr(so.SoundeoClient, "from_env",
                        classmethod(lambda cls, **kw: QuotaClient()))

    summary = fetch_missing([csv_path], str(library), audio_format="aiff")

    assert summary["quota_skipped"] == 1   # the track that hit the limit
    assert summary["deferred"] == 2        # the rest, stopped (not YouTubed)
    assert summary["youtube"] == 0
    assert not any(p.name.endswith("[U].aiff") for p in library.iterdir())


def test_acquire_soundeo_download_error_falls_back_to_youtube(tmp_path, monkeypatch):
    _yt_stub(monkeypatch)
    track = PlaylistTrack(name="Tune", artists=["Artist"], duration_sec=200)

    class ErrClient:  # found on Soundeo, but the download endpoint 404s
        def search(self, t):
            return [so.SoundeoResult("1", "Artist", "Tune", 200, formats=["aiff"])]

        def pick(self, t, results):
            return results[0]

        def download(self, result, dest):
            raise so.SoundeoError("not available on Soundeo (HTTP 404)")

    out = sf.acquire_track(track, str(tmp_path), soundeo=ErrClient(), quota=sf._QuotaState())
    assert out.source == "youtube"                       # per-track backfall
    assert (tmp_path / "Artist - Tune[U].aiff").exists()


# --------------------------------------------------------------------------- #
# Not-found cache (skip re-searching tracks absent from both sources)
# --------------------------------------------------------------------------- #
def _no_match_yt(monkeypatch):
    # download_track always reports "not found on YouTube".
    monkeypatch.setattr(sf, "download_track",
                        lambda track, dest_dir, **kw: sf.DownloadOutcome(
                            "no_match", "", "no result", "youtube"))


def test_fetch_missing_logs_work_queue_composition(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(so.SoundeoClient, "from_env", classmethod(lambda cls, **kw: None))
    monkeypatch.setattr(sf, "tools_available", lambda: (True, True))
    library = tmp_path / "lib"
    library.mkdir()
    (library / "Adele - Skyfall (Original Mix).aiff").write_bytes(b"\x00")
    youtube_file = library / "UNKLE - Hold My Hand[U].aiff"
    youtube_file.write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Skyfall", "Adele", 286000),
        _row("b", "Hold My Hand", "UNKLE", 300000),
        _row("c", "Ghost", "Nobody", 200000),
    ])
    _seed_cache(str(library / "outputs" / "fetch"), {
        "spotify:track:c": {"artists": "Nobody", "name": "Ghost"},
    })
    monkeypatch.setattr(sf, "download_track", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("present marked tracks are not checked by default")))

    caplog.set_level("INFO", logger="dj_tools.spotify_fetch")
    summary = fetch_missing([csv_path], str(library), audio_format="aiff")

    assert summary["missing"] == 1
    assert summary["cached_skipped"] == 1
    assert summary["reverify"] == 0
    log_text = caplog.text
    assert "fetch queue: 0 to fetch, 1 existing marked download(s) not checked" in log_text
    assert "use --check-marked-upgrades" in log_text
    assert "1 cached not-found skipped, 1 already present as originals/curated" in log_text
    assert "CHECK marked download" not in log_text


def test_fetch_missing_check_marked_upgrades_reverifies_marked_files(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(so.SoundeoClient, "from_env", classmethod(lambda cls, **kw: None))
    monkeypatch.setattr(sf, "tools_available", lambda: (True, True))
    library = tmp_path / "lib"
    library.mkdir()
    youtube_file = library / "UNKLE - Hold My Hand[U].aiff"
    youtube_file.write_bytes(b"\x00")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("b", "Hold My Hand", "UNKLE", 300000),
    ])
    monkeypatch.setattr(sf, "download_track", lambda *a, **k: sf.DownloadOutcome(
        "skip", str(youtube_file), "already exists, duration ok (300s)", "youtube"))

    caplog.set_level("INFO", logger="dj_tools.spotify_fetch")
    summary = fetch_missing(
        [csv_path], str(library), audio_format="aiff", check_marked_upgrades=True,
    )

    assert summary["reverify"] == 1
    log_text = caplog.text
    assert "fetch queue: 0 to fetch, 1 existing marked download(s) queued" in log_text
    assert "CHECK marked download for Soundeo upgrade/duration: UNKLE - Hold My Hand" in log_text
    assert "YOUTUBE OK (existing marked download; duration ok): UNKLE - Hold My Hand" in log_text


def test_fetch_missing_caches_not_found_and_skips_next_run(tmp_path, monkeypatch):
    _no_match_yt(monkeypatch)  # YouTube-only (no Soundeo creds), nothing found
    monkeypatch.setattr(so.SoundeoClient, "from_env", classmethod(lambda cls, **kw: None))
    library = tmp_path / "lib"
    library.mkdir()
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Ghost", "Nobody", 200000)])

    s1 = fetch_missing([csv_path], str(library), audio_format="aiff")
    assert s1["unmatched"] == 1 and s1["cached_skipped"] == 0
    cache = sf.load_not_found_cache(str(library / "outputs" / "fetch"))
    assert "spotify:track:a" in cache  # remembered

    # Second run: the search is skipped, download_track must NOT be called.
    monkeypatch.setattr(sf, "download_track", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not search a cached not-found track")))
    s2 = fetch_missing([csv_path], str(library), audio_format="aiff")
    assert s2["cached_skipped"] == 1
    assert s2["unmatched"] == 0


def test_fetch_missing_force_lookup_retries_cached(tmp_path, monkeypatch):
    _no_match_yt(monkeypatch)
    monkeypatch.setattr(so.SoundeoClient, "from_env", classmethod(lambda cls, **kw: None))
    library = tmp_path / "lib"
    library.mkdir()
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Ghost", "Nobody", 200000)])
    fetch_missing([csv_path], str(library), audio_format="aiff")  # seeds the cache

    calls = []
    monkeypatch.setattr(sf, "download_track",
                        lambda track, dest_dir, **kw: calls.append(track)
                        or sf.DownloadOutcome("no_match", "", "no result", "youtube"))
    s = fetch_missing([csv_path], str(library), audio_format="aiff", force_lookup=True)
    assert s["cached_skipped"] == 0
    assert len(calls) == 1  # re-searched despite being cached


def test_fetch_missing_clears_cache_when_found(tmp_path, monkeypatch):
    monkeypatch.setattr(so.SoundeoClient, "from_env", classmethod(lambda cls, **kw: None))
    library = tmp_path / "lib"
    library.mkdir()
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Ghost", "Nobody", 200000)])
    _no_match_yt(monkeypatch)
    fetch_missing([csv_path], str(library), audio_format="aiff")  # cached not-found

    # Now YouTube "finds" it -> cache entry must be cleared.
    _yt_stub(monkeypatch)
    fetch_missing([csv_path], str(library), audio_format="aiff", force_lookup=True)
    cache = sf.load_not_found_cache(str(library / "outputs" / "fetch"))
    assert "spotify:track:a" not in cache


def test_fetch_missing_does_not_cache_when_soundeo_had_match(tmp_path, monkeypatch):
    # Soundeo lists the track but the download fails; YouTube also misses.
    # It must NOT be cached as not-found (it IS on Soundeo).
    _no_match_yt(monkeypatch)
    library = tmp_path / "lib"
    library.mkdir()
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Aerial", "Azzecca", 200000)])

    class ListedButFailsClient:
        def login(self): pass
        def close(self): pass
        def search(self, track):
            return [so.SoundeoResult("1", "Azzecca", "Aerial", 200, formats=["aiff"])]
        def pick(self, track, results):
            return results[0]
        def download(self, result, dest):
            raise so.SoundeoError("not available on Soundeo (HTTP 404)")

    monkeypatch.setattr(so.SoundeoClient, "from_env",
                        classmethod(lambda cls, **kw: ListedButFailsClient()))

    s = fetch_missing([csv_path], str(library), audio_format="aiff")
    assert s["unmatched"] == 1
    cache = sf.load_not_found_cache(str(library / "outputs" / "fetch"))
    assert "spotify:track:a" not in cache  # on Soundeo -> not cached as not-found


def test_download_track_keeps_existing_file_when_replacement_not_found(tmp_path, monkeypatch):
    existing = tmp_path / "Artist - Tune[U].aiff"
    existing.write_bytes(b"old-copy")

    def fake_probe(path):
        return 250.0 if str(path) == str(existing) else None

    def fake_run(cmd, **kw):
        class R:
            pass
        r = R()
        r.stdout = "vid1\t210\tWrong Duration\n"
        r.stderr = ""
        return r

    monkeypatch.setattr(sf, "probe_duration", fake_probe)
    monkeypatch.setattr(sf.subprocess, "run", fake_run)
    t = PlaylistTrack(name="Tune", artists=["Artist"], duration_sec=200)

    out = sf.download_track(t, str(tmp_path), tolerance=3)

    assert out.status == "no_match"
    assert existing.exists()
    assert existing.read_bytes() == b"old-copy"


def test_download_track_transient_403_is_fail_not_no_match(tmp_path, monkeypatch):
    # A YouTube download that 403s on every attempt is transient -> "fail"
    # (retried next run, never cached as not-found), not "no_match".
    def fake_run(cmd, **kw):
        class R:
            pass
        r = R()
        if any("ytsearch" in a for a in cmd):          # candidate listing
            r.stdout, r.stderr = "vid1\t200\tTitle\n", ""
        else:                                           # the download attempt
            r.stdout = ""
            r.stderr = "ERROR: unable to download video data: HTTP Error 403: Forbidden"
        return r

    monkeypatch.setattr(sf.subprocess, "run", fake_run)
    monkeypatch.setattr(sf.os.path, "exists", lambda p: False)  # no wav ever produced
    t = PlaylistTrack(name="RIZZ", artists=["AYYBO"], duration_sec=200)
    out = sf.download_track(t, str(tmp_path))
    assert out.status == "fail"
    assert "403" in out.detail


# --------------------------------------------------------------------------- #
# forget_not_found (drop cached not-found entries)
# --------------------------------------------------------------------------- #
def _seed_cache(report_dir, entries):
    import os as _os
    _os.makedirs(report_dir, exist_ok=True)
    sf.save_not_found_cache(report_dir, entries)


def test_forget_not_found_matches_substring(tmp_path):
    rd = str(tmp_path / "outputs" / "fetch")
    _seed_cache(rd, {
        "spotify:track:a": {"artists": "bawab, Sydka", "name": "Malevolence"},
        "spotify:track:b": {"artists": "AYYBO", "name": "RIZZ"},
    })
    removed = sf.forget_not_found(str(tmp_path), ["Malevolence"])
    assert removed == ["bawab, Sydka - Malevolence"]
    remaining = sf.load_not_found_cache(rd)
    assert "spotify:track:a" not in remaining and "spotify:track:b" in remaining


def test_forget_not_found_all_clears(tmp_path):
    rd = str(tmp_path / "outputs" / "fetch")
    _seed_cache(rd, {"spotify:track:a": {"artists": "X", "name": "Y"},
                     "spotify:track:b": {"artists": "P", "name": "Q"}})
    removed = sf.forget_not_found(str(tmp_path), ["all"])
    assert len(removed) == 2
    assert sf.load_not_found_cache(rd) == {}


def test_forget_not_found_dry_run_keeps_entries(tmp_path):
    rd = str(tmp_path / "outputs" / "fetch")
    _seed_cache(rd, {"spotify:track:a": {"artists": "X", "name": "Y"}})
    removed = sf.forget_not_found(str(tmp_path), ["X - Y"], dry_run=True)
    assert removed == ["X - Y"]
    assert "spotify:track:a" in sf.load_not_found_cache(rd)  # dry run leaves it


def test_cli_fetch_missing_default_library():
    from dj_tools.cli import DEFAULT_FETCH_LIBRARY, _build_parser
    args = _build_parser().parse_args(["fetch-missing", "playlist.csv"])
    assert args.library == DEFAULT_FETCH_LIBRARY == r"D:\Music"


def test_cli_forget_cached_removes_entry(tmp_path):
    from dj_tools.cli import _build_parser, _run_fetch_missing
    rd = str(tmp_path / "lib" / "outputs" / "fetch")
    _seed_cache(rd, {"spotify:track:a": {"artists": "Spada", "name": "I Lose My Mind"}})
    args = _build_parser().parse_args(
        ["fetch-missing", "--library", str(tmp_path / "lib"), "--forget-cached", "Lose My Mind"])
    assert _run_fetch_missing(args) == 0
    assert sf.load_not_found_cache(rd) == {}


# --------------------------------------------------------------------------- #
# Soundeo tag storage (store as-is) + tag repair
# --------------------------------------------------------------------------- #
def _write_tagged_aiff(path, *, title="", artist=""):
    import numpy as np
    import soundfile as sf_
    sf_.write(str(path), np.zeros(2205, dtype="float32"), 22050)
    if title or artist:
        from dj_tagger.metadata import write_track_metadata
        write_track_metadata(str(path), title=title, artist=artist)


def test_read_descriptive_tags_roundtrip(tmp_path):
    from dj_tagger.metadata import read_descriptive_tags
    p = tmp_path / "x.aiff"
    _write_tagged_aiff(p, title="Madama Firefly (Original Mix)", artist="Gab Rhome")
    tags = read_descriptive_tags(str(p))
    assert tags["title"] == "Madama Firefly (Original Mix)"
    assert tags["artist"] == "Gab Rhome"


def test_acquire_soundeo_stores_as_is_no_embed(tmp_path, monkeypatch):
    # Soundeo path must NOT overwrite the file's native tags with Spotify ones.
    _yt_stub(monkeypatch)
    calls = []
    monkeypatch.setattr(sf, "_embed_metadata", lambda *a, **k: calls.append(a))
    track = PlaylistTrack(name="Tune", artists=["Artist"], album="SpotifyAlbum")
    fake = _FakeSoundeo(so.SoundeoResult("1", "Artist", "Tune", 200, formats=["aiff"]))
    out = sf.acquire_track(track, str(tmp_path), soundeo=fake, quota=sf._QuotaState())
    assert out.source == "soundeo" and out.status == "ok"
    assert calls == []  # never embedded on the Soundeo path


class _RefixFake:
    """Fake Soundeo client for refix: serves owned/not-owned results per title."""
    audio_format = "aiff"

    def __init__(self, owned_titles):
        self.owned_titles = set(owned_titles)  # titles the account "owns"
        self.downloaded_ids = []

    def login(self): pass
    def close(self): pass

    def search(self, track):
        owned = track.name in self.owned_titles
        return [so.SoundeoResult("sid-" + track.name, track.primary_artist, track.name,
                                 200, formats=["aiff"], downloaded=owned)]

    def pick_owned(self, track, results, *, target_duration=None):
        cand = [r for r in results if r.downloaded and "aiff" in r.formats]
        return cand[0] if cand else None

    def download(self, result, dest, *, assume_free=False):
        self.downloaded_ids.append((result.id, assume_free))
        _write_tagged_aiff(dest, title="NATIVE " + result.title, artist=result.artist)


def test_refix_fixes_owned_skips_curated(tmp_path, monkeypatch):
    lib = tmp_path
    owned = lib / "Gab Rhome - Madama Firefly.aiff"
    curated = lib / "Some Artist - My Own Master.aiff"
    _write_tagged_aiff(owned, title="Spotify Title", artist="Gab Rhome")
    _write_tagged_aiff(curated, title="Curated", artist="Some Artist")

    fake = _RefixFake(owned_titles=["Madama Firefly"])
    monkeypatch.setattr(so.SoundeoClient, "from_env", classmethod(lambda cls, **kw: fake))

    summary = sf.refix_soundeo_tags(str(lib))
    assert summary["fixed"] == 1 and summary["not_owned"] == 1
    # Owned file re-downloaded (assume_free) with native tags; curated untouched.
    assert fake.downloaded_ids == [("sid-Madama Firefly", True)]
    from dj_tagger.metadata import read_descriptive_tags
    assert read_descriptive_tags(str(owned))["title"] == "NATIVE Madama Firefly"
    assert read_descriptive_tags(str(curated))["title"] == "Curated"  # not touched
    # Logged so a second pass skips it.
    log = sf.load_soundeo_tags_log(str(lib / "outputs" / "fetch"))
    assert sf._soundeo_log_key(str(owned)) in log


def test_refix_upgrades_nonaiff_owned_to_aiff(tmp_path, monkeypatch):
    # A curated .mp3 the account owns on Soundeo is upgraded to AIFF; the
    # inferior .mp3 is removed (AIFF is superior).
    lib = tmp_path
    mp3 = lib / "Bob Moses - Winter's Song (Original Mix).mp3"
    tmp_aiff = lib / "Bob Moses - Winter's Song (Original Mix).aiff"
    _write_tagged_aiff(tmp_aiff, title="tmp")  # valid audio, then move to .mp3 name
    tmp_aiff.replace(mp3)
    fake = _RefixFake(owned_titles=["Winter's Song (Original Mix)"])
    monkeypatch.setattr(so.SoundeoClient, "from_env", classmethod(lambda cls, **kw: fake))

    summary = sf.refix_soundeo_tags(str(lib))
    assert summary["fixed"] == 1 and summary["upgraded"] == 1
    assert (lib / "Bob Moses - Winter's Song (Original Mix).aiff").exists()
    assert not mp3.exists()  # inferior lossy original removed
    assert fake.downloaded_ids == [("sid-Winter's Song (Original Mix)", True)]


def test_refix_skips_already_logged(tmp_path, monkeypatch):
    lib = tmp_path
    owned = lib / "Gab Rhome - Madama Firefly.aiff"
    _write_tagged_aiff(owned, title="X", artist="Gab Rhome")
    rd = lib / "outputs" / "fetch"
    rd.mkdir(parents=True)
    sf.save_soundeo_tags_log(str(rd), {sf._soundeo_log_key(str(owned)):
                                       {"soundeo_id": "old", "tags": {}}})
    fake = _RefixFake(owned_titles=["Madama Firefly"])
    monkeypatch.setattr(so.SoundeoClient, "from_env", classmethod(lambda cls, **kw: fake))
    summary = sf.refix_soundeo_tags(str(lib))
    assert summary["candidates"] == 0 and fake.downloaded_ids == []


def test_refix_ignores_marked_youtube_files(tmp_path, monkeypatch):
    lib = tmp_path
    yt = lib / "Artist - Tune[U].aiff"
    _write_tagged_aiff(yt, title="Spotify", artist="Artist")
    fake = _RefixFake(owned_titles=["Tune"])
    monkeypatch.setattr(so.SoundeoClient, "from_env", classmethod(lambda cls, **kw: fake))
    summary = sf.refix_soundeo_tags(str(lib))
    assert summary["candidates"] == 0 and fake.downloaded_ids == []

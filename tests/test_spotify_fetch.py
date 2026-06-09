"""Tests for the Spotify playlist fetch-missing feature."""

from dj_tools.spotify_fetch import (
    PlaylistTrack,
    best_match,
    build_candidate_command,
    build_download_command,
    classify_tracks,
    clean_track_name,
    collect_playlists,
    duration_mismatch,
    fetch_missing,
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
    # The [U] source marker is appended verbatim after the clean stem.
    assert t.target_basename("[U]") == "NTO - Starlings (Henry Saiz Remix)[U]"
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
    # Only the [U]-marked name counts as a tool download.
    tool = tmp_path / "NTO - Starlings (Henry Saiz Remix)[U].aiff"
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

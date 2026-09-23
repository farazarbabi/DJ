"""Tests for the Soundeo upgrade pass (``dj fetch-missing --upgrade-soundeo``)."""

import csv

from dj_tools import spotify_fetch as sf
from dj_tools.soundeo import SoundeoError, SoundeoQuotaExceeded, SoundeoResult
from dj_tools.soundeo_upgrade import (
    TIER_EXTENDED,
    TIER_MP3,
    TIER_ORIGINAL,
    TIER_RADIO,
    TIER_WAV,
    TIER_YOUTUBE,
    is_upgrade,
    quality_tier,
    retire_superseded,
    upgrade_soundeo,
)

CSV_HEADER = (
    "Track URI,Track Name,Album Name,Artist Name(s),Release Date,"
    "Duration (ms),Popularity,Explicit,Added By,Added At,Genres,Record Label\n"
)


def _row(uri, name, artists, ms=300000):
    return (
        f"spotify:track:{uri},\"{name}\",\"Album\",\"{artists}\","
        f"2020-01-01,{ms},50,false,user,2020-01-01T00:00:00Z,\"\",\"Label\"\n"
    )


def _write_csv(path, rows):
    path.write_text(CSV_HEADER + "".join(rows), encoding="utf-8-sig")
    return str(path)


def _result(rid, artist, title, formats=("aiff",), duration=420):
    return SoundeoResult(rid, artist, title, duration, formats=list(formats))


class FakeSoundeo:
    """Answers searches from a name->results map; download writes a stub file."""

    audio_format = "aiff"

    def __init__(self, results_by_name=None, *, download_error=None):
        self.results = results_by_name or {}
        self.download_error = download_error
        self.searched: list[str] = []
        self.downloaded: list[str] = []

    def login(self):
        pass

    def close(self):
        pass

    def search(self, track):
        self.searched.append(track.name)
        return list(self.results.get(track.name, []))

    def pick(self, track, results):
        return results[0] if results else None

    def download(self, result, dest, audio_format="aiff"):
        if self.download_error:
            raise self.download_error
        self.downloaded.append(result.id)
        with open(dest, "wb") as f:
            f.write(b"\x00")


def _library(tmp_path, *names):
    lib = tmp_path / "lib"
    lib.mkdir()
    for name in names:
        (lib / name).write_bytes(b"\x00")
    return lib


def _no_youtube(monkeypatch):
    monkeypatch.setattr(sf, "download_track", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("YouTube must never be used by the upgrade pass")))


def _report(lib):
    with open(lib / "outputs" / "fetch" / "soundeo_upgrade.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Tiers
# --------------------------------------------------------------------------- #
def test_quality_tier_ranks_markers_below_every_unmarked_cut():
    assert quality_tier("Artist - Song (Extended Mix)") == TIER_EXTENDED
    assert quality_tier("Artist - Song") == TIER_ORIGINAL
    assert quality_tier("Artist - Song (Original Mix)") == TIER_ORIGINAL
    assert quality_tier("Artist - Song (Avocado Remix)") == TIER_ORIGINAL
    assert quality_tier("Artist - Song (Radio Edit)") == TIER_RADIO
    assert quality_tier("Artist - Song (Extended Mix)[W]") == TIER_WAV
    assert quality_tier("Artist - Song[M]") == TIER_MP3
    assert quality_tier("Artist - Song[U]") == TIER_YOUTUBE
    assert quality_tier("Radiohead - Creep") == TIER_ORIGINAL  # artist is not a version


def test_is_upgrade_requires_a_strictly_better_tier():
    ext = _result("e", "Artist", "Song (Extended Mix)")
    orig = _result("o", "Artist", "Song (Original Mix)")
    radio = _result("r", "Artist", "Song (Radio Edit)")
    remix = _result("x", "Artist", "Song (Avocado Remix)")
    mp3 = _result("m", "Artist", "Song", formats=("mp3",))

    assert is_upgrade("Artist - Song (Original Mix)", "Artist - Song (Extended Mix)", ext)
    assert is_upgrade("Artist - Song (Radio Edit)", "Artist - Song (Original Mix)", orig)
    assert is_upgrade("Artist - Song[U]", "Artist - Song (Original Mix)", orig)
    assert is_upgrade("Artist - Song[U]", "Artist - Song[M]", mp3)

    assert not is_upgrade("Artist - Song (Avocado Remix)", "Artist - Song (Avocado Remix)", remix)
    assert not is_upgrade("Artist - Song", "Artist - Song (Original Mix)", orig)
    assert not is_upgrade("Artist - Song (Extended Mix)[W]", "Artist - Song (Extended Mix)[W]", ext)
    assert not is_upgrade("Artist - Song[U]", "Artist - Song (Radio Edit)", radio)


# --------------------------------------------------------------------------- #
# upgrade_soundeo
# --------------------------------------------------------------------------- #
def test_dry_run_reports_upgrade_without_downloading(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - Song (Original Mix).aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Song", "Artist")])
    client = FakeSoundeo({"Song": [_result("e", "Artist", "Song (Extended Mix)")]})

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client, dry_run=True)

    assert s["would_upgrade"] == 1 and s["upgraded"] == 0
    assert client.downloaded == []
    assert (lib / "Artist - Song (Original Mix).aiff").exists()
    assert not (lib / "Artist - Song (Extended Mix).aiff").exists()
    (row,) = _report(lib)
    assert (row["current"], row["soundeo"], row["action"]) == ("Original", "Extended", "would download")


def test_upgrade_replaces_youtube_copy_and_records_provenance(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - Song[U].aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Song", "Artist")])
    client = FakeSoundeo({"Song": [_result("o", "Artist", "Song (Original Mix)")]})

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client)

    assert s["upgraded"] == 1 and client.downloaded == ["o"]
    assert not (lib / "Artist - Song[U].aiff").exists()
    new = lib / "Artist - Song (Original Mix).aiff"
    assert new.exists()
    log = sf.load_soundeo_tags_log(str(lib / "outputs" / "fetch"))
    assert [entry["basename"] for entry in log.values()] == [new.name]
    (row,) = _report(lib)
    assert row["current"] == "YouTube [U]" and row["action"].startswith("upgraded; removed ")


def test_upgrade_moves_unmarked_original_to_replaced_dir(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - Song (Original Mix).aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Song", "Artist")])
    client = FakeSoundeo({"Song": [_result("e", "Artist", "Song (Extended Mix)")]})

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client)

    assert s["upgraded"] == 1
    assert (lib / "Artist - Song (Extended Mix).aiff").exists()
    assert not (lib / "Artist - Song (Original Mix).aiff").exists()
    assert (lib / "outputs" / "fetch" / "replaced" / "Artist - Song (Original Mix).aiff").exists()


def test_extended_in_library_is_not_even_searched(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - Song (Extended Mix).aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Song", "Artist")])
    client = FakeSoundeo({"Song": [_result("e", "Artist", "Song (Extended Mix)")]})

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client)

    assert s["no_upgrade"] == 1 and client.searched == []


def test_same_tier_remix_is_not_redownloaded(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - Song (Avocado Remix).aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Song - Avocado Remix", "Artist")])
    client = FakeSoundeo({"Song - Avocado Remix": [_result("x", "Artist", "Song (Avocado Remix)")]})

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client)

    assert s["no_upgrade"] == 1 and client.downloaded == []
    (row,) = _report(lib)
    assert (row["current"], row["soundeo"], row["action"]) == ("Original", "Original", "skip")


def test_quota_hit_stops_the_pass(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - One[U].aiff", "Artist - Two[U].aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "One", "Artist"), _row("b", "Two", "Artist")])
    client = FakeSoundeo(
        {"One": [_result("1", "Artist", "One")], "Two": [_result("2", "Artist", "Two")]},
        download_error=SoundeoQuotaExceeded("daily download limit reached"),
    )

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client)

    assert s["upgraded"] == 0 and s["deferred"] == 2
    assert len(client.searched) == 1
    assert (lib / "Artist - One[U].aiff").exists() and (lib / "Artist - Two[U].aiff").exists()


def test_failed_download_is_reported_and_leaves_library_untouched(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - Song[U].aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Song", "Artist")])
    client = FakeSoundeo({"Song": [_result("o", "Artist", "Song")]},
                         download_error=SoundeoError("cdn 500"))

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client)

    assert s["failed"] == 1 and (lib / "Artist - Song[U].aiff").exists()
    (row,) = _report(lib)
    assert row["action"] == "failed: cdn 500"


def test_missing_tracks_are_skipped_not_downloaded(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path)
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Ghost", "Nobody")])
    client = FakeSoundeo({"Ghost": [_result("g", "Nobody", "Ghost")]})

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client)

    assert s["missing"] == 1 and s["checked"] == 0 and client.searched == []


def test_retire_superseded_keeps_a_file_overwritten_in_place(tmp_path):
    f = tmp_path / "Artist - Song.aiff"
    f.write_bytes(b"\x00")
    assert retire_superseded(str(f), str(tmp_path), str(f)) == "overwritten in place"
    assert f.exists()


# --------------------------------------------------------------------------- #
# Resilience and resume
# --------------------------------------------------------------------------- #
def test_duplicate_rows_for_one_file_do_not_abort_the_pass(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - Song[U].aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Song", "Artist"), _row("b", "Song - Original Mix", "Artist"),
    ])
    pick = _result("o", "Artist", "Song (Original Mix)")
    client = FakeSoundeo({"Song": [pick], "Song - Original Mix": [pick]})

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client)

    assert s["upgraded"] == 1 and s["failed"] == 0 and client.downloaded == ["o"]
    rows = _report(lib)
    assert rows[1]["action"] == "skip: file already replaced"


def test_pick_of_a_different_cut_never_replaces_the_file(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - Far Away Place.aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [
        _row("a", "Far Away Place - Rampa Remix", "Artist, Rampa"),
    ])
    client = FakeSoundeo({"Far Away Place - Rampa Remix": [
        _result("r", "Artist", "Far Away Place (Rampa Extended Remix)")]})

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client, threshold=0.5)

    assert s["checked"] == 1 and client.downloaded == []
    assert (lib / "Artist - Far Away Place.aiff").exists()
    (row,) = _report(lib)
    assert row["action"] == "skip: different cut"


def test_one_failing_track_does_not_abort_the_pass(tmp_path, monkeypatch):
    import dj_tools.soundeo_upgrade as su

    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - One[U].aiff", "Artist - Two[U].aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "One", "Artist"), _row("b", "Two", "Artist")])
    client = FakeSoundeo({"One": [_result("1", "Artist", "One")], "Two": [_result("2", "Artist", "Two")]})
    real_retire = su.retire_superseded

    def flaky(current, library, new_path):
        if "One" in current:
            raise OSError("disk hiccup")
        return real_retire(current, library, new_path)

    monkeypatch.setattr(su, "retire_superseded", flaky)

    s = upgrade_soundeo([csv_path], str(lib), soundeo=client)

    assert s["failed"] == 1 and s["upgraded"] == 1 and client.downloaded == ["1", "2"]
    rows = _report(lib)
    assert rows[0]["action"] == "error: disk hiccup" and rows[1]["action"].startswith("upgraded")


def test_checked_tracks_are_skipped_on_rerun_unless_forced_or_file_changed(tmp_path, monkeypatch):
    _no_youtube(monkeypatch)
    lib = _library(tmp_path, "Artist - Song (Original Mix).aiff")
    csv_path = _write_csv(tmp_path / "p.csv", [_row("a", "Song", "Artist")])
    client = FakeSoundeo({"Song": [_result("o", "Artist", "Song (Original Mix)")]})

    upgrade_soundeo([csv_path], str(lib), soundeo=client)
    assert client.searched == ["Song"]

    s2 = upgrade_soundeo([csv_path], str(lib), soundeo=client)
    assert client.searched == ["Song"] and s2["previously_checked"] == 1
    assert (lib / "outputs" / "fetch" / "soundeo_upgrade_checked.json").exists()

    upgrade_soundeo([csv_path], str(lib), soundeo=client, force_lookup=True)
    assert client.searched == ["Song", "Song"]

    (lib / "Artist - Song (Original Mix).aiff").rename(lib / "Artist - Song (Radio Edit).aiff")
    s4 = upgrade_soundeo([csv_path], str(lib), soundeo=client)
    assert client.searched == ["Song", "Song", "Song"] and s4["upgraded"] == 1

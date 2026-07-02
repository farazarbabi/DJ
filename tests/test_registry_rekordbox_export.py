"""Tests for the Rekordbox XML collection export."""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from dj_registry.adapters.rekordbox_xml import ingest_rekordbox
from dj_registry.config import RegistryConfig
from dj_registry.models import CuePoint, FileRecord, LogicalTrack
from dj_registry.store.csv_store import CsvStore
from dj_registry.sync.rekordbox_export import generate_rekordbox_collection


def _track(track_id: str, file_id: str, **kw) -> LogicalTrack:
    defaults = dict(
        track_id=track_id,
        primary_file_id=file_id,
        artist_canonical="Artist",
        title_canonical=f"Title {track_id}",
        canonical_bpm="128",
        canonical_key_standard="Am",
        canonical_key_camelot="8A",
        duration_sec_canonical=300.0,
    )
    defaults.update(kw)
    return LogicalTrack(**defaults)


def _file(file_id: str, track_id: str, path: Path) -> FileRecord:
    return FileRecord(
        file_id=file_id,
        track_id=track_id,
        path_abs=str(path),
        file_name=path.name,
        size_bytes=max(path.stat().st_size, 1) if path.exists() else 123,
        audio_duration_sec=300.0,
        is_primary_file=True,
    )


def _hot_cue(cue_id: str, track_id: str, file_id: str, num: int, name: str, t: float) -> CuePoint:
    return CuePoint(
        cue_id=cue_id,
        track_id=track_id,
        file_id=file_id,
        cue_kind="hot",
        cue_name=name,
        cue_time_sec=t,
        rekordbox_num=num,
        rekordbox_type="0",
        red=255,
        green=55,
        blue=111,
    )


def _fixture(tmp_path: Path):
    """Two tracks with real files, one m3u8 playlist, a hot cue on track 1."""
    a = tmp_path / "Artist - One.aiff"
    b = tmp_path / "Artist - Two.aiff"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    tracks = [
        _track("T1", "F1", tagger_first_beat_sec="0.123"),
        _track("T2", "F2"),
    ]
    files = [_file("F1", "T1", a), _file("F2", "T2", b)]
    cues = [_hot_cue("C1", "T1", "F1", 0, "MIX IN", 8.0)]

    # A playlist directory mirroring outputs/playlists/, with one folder.
    playlists_root = tmp_path / "playlists"
    (playlists_root / "by_key").mkdir(parents=True)
    m3u8 = playlists_root / "by_key" / "08A.m3u8"
    # Duplicate the same path plus a path not in the library — both should be
    # collapsed/skipped in the resulting playlist node.
    m3u8.write_text(
        "\n".join([
            "#EXTM3U",
            f"#EXTINF:-1,Artist - One",
            str(a),
            str(a),  # duplicate -> deduped
            str(tmp_path / "not-in-library.aiff"),  # unresolved -> skipped
        ]),
        encoding="utf-8-sig",
    )
    return tracks, files, cues, playlists_root, a, b


def test_collection_has_tracks_tempo_and_cues(tmp_path):
    tracks, files, cues, playlists_root, _a, _b = _fixture(tmp_path)
    out = tmp_path / "collection.xml"

    res = generate_rekordbox_collection(tracks, files, cues, str(out), playlists_root=str(playlists_root))

    assert res["tracks"] == 2
    root = etree.parse(str(out)).getroot()
    collection = root.find("COLLECTION")
    assert collection.get("Entries") == "2"

    track_els = collection.findall("TRACK")
    assert len(track_els) == 2
    first = track_els[0]
    assert first.get("Artist") == "Artist"
    assert first.get("AverageBpm") == "128.00"
    assert first.get("Tonality") == "Am"
    assert first.get("Location").startswith("file://localhost/")

    tempo = first.find("TEMPO")
    assert tempo is not None
    assert tempo.get("Bpm") == "128.00"
    assert tempo.get("Inizio") == "0.123"  # from tagger_first_beat_sec
    assert tempo.get("Metro") == "4/4"

    marks = first.findall("POSITION_MARK")
    assert len(marks) == 1
    assert marks[0].get("Name") == "MIX IN"
    assert marks[0].get("Num") == "0"


def test_collection_beatgrid_falls_back_to_zero_without_first_beat(tmp_path):
    tracks, files, cues, playlists_root, _a, _b = _fixture(tmp_path)
    out = tmp_path / "collection.xml"

    generate_rekordbox_collection(tracks, files, cues, str(out), playlists_root=str(playlists_root))

    root = etree.parse(str(out)).getroot()
    # T2 has no tagger_first_beat_sec -> Inizio 0.000
    second = root.find("COLLECTION").findall("TRACK")[1]
    assert second.find("TEMPO").get("Inizio") == "0.000"


def test_playlist_tree_mirrors_m3u8_and_dedupes(tmp_path):
    tracks, files, cues, playlists_root, _a, _b = _fixture(tmp_path)
    out = tmp_path / "collection.xml"

    res = generate_rekordbox_collection(tracks, files, cues, str(out), playlists_root=str(playlists_root))

    assert res["playlists"] == 1
    assert res["entries"] == 1  # duplicate collapsed, unresolved skipped
    assert res["unresolved"] == 1

    root = etree.parse(str(out)).getroot()
    root_node = root.find("PLAYLISTS").find("NODE")
    assert root_node.get("Name") == "ROOT"

    folder = root_node.find("NODE")
    assert folder.get("Type") == "0"
    assert folder.get("Name") == "by_key"

    playlist = folder.find("NODE")
    assert playlist.get("Type") == "1"
    assert playlist.get("Name") == "08A"
    assert playlist.get("Entries") == "1"

    entries = playlist.findall("TRACK")
    assert len(entries) == 1
    # References the collection TrackID of the first track.
    t1_id = root.find("COLLECTION").findall("TRACK")[0].get("TrackID")
    assert entries[0].get("Key") == t1_id


def test_generated_xml_round_trips_through_ingest(tmp_path):
    tracks, files, cues, playlists_root, _a, _b = _fixture(tmp_path)
    out = tmp_path / "collection.xml"
    generate_rekordbox_collection(tracks, files, cues, str(out), playlists_root=str(playlists_root))

    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks(tracks)
    store.save_files(files)
    config = RegistryConfig(output_dir=str(tmp_path / "registry"))
    config.rekordbox_xml_path = str(out)

    matched = ingest_rekordbox(config, store)
    assert matched == 2

    obs = [o for o in store.load_observations() if o.source_system == "rekordbox"]
    assert len(obs) == 2
    assert all(o.bpm == "128" for o in obs)
    assert all(o.key_camelot for o in obs)


def test_tracks_without_primary_file_are_skipped(tmp_path):
    a = tmp_path / "Artist - One.aiff"
    a.write_bytes(b"a")
    tracks = [
        _track("T1", "F1"),
        _track("T2", "MISSING"),  # no matching FileRecord
    ]
    files = [_file("F1", "T1", a)]
    out = tmp_path / "collection.xml"

    res = generate_rekordbox_collection(tracks, files, [], str(out))

    assert res["tracks"] == 1
    root = etree.parse(str(out)).getroot()
    assert len(root.find("COLLECTION").findall("TRACK")) == 1

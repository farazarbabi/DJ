"""Tests for registry file scanner cache behavior."""

from dj_registry.adapters import file_scanner
from dj_registry.config import RegistryConfig
from dj_registry.models import FileRecord, PayloadIndexEntry, SourceObservation
from dj_registry.store.csv_store import CsvStore
from dj_registry.store.obs_cache import ObsCache
from dj_tagger.universal_cache import reset_cache


def test_scan_files_reuses_cached_tag_observation_without_extracting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reset_cache()

    track = tmp_path / "Track.mp3"
    track.write_bytes(b"fake")
    path_abs = str(track.resolve())

    config = RegistryConfig(
        library_roots=[str(tmp_path)],
        supported_extensions=[".mp3"],
        output_dir=str(tmp_path / "registry"),
    )
    store = CsvStore(config.output_dir)
    obs_cache = ObsCache("cache/registry_cache.pkl")
    obs_cache.put_by_file(
        path_abs,
        180.0,
        "tag",
        SourceObservation(
            source_system="tag",
            artist="Artist",
            title="Title",
            source_object_id="USRC12345",
            key_standard="A minor",
            key_camelot="9A",
            bpm="126",
            genre="Tech House",
            comments="9A|E3|MEL|INST|DRK.TECH.HOUS.DRV",
        ),
    )
    obs_cache.save()

    monkeypatch.setattr(file_scanner, "_audio_info", lambda path: {"duration": 180.0, "sample_rate": 44100, "channels": 2})
    monkeypatch.setattr(file_scanner, "_bitrate", lambda path: 320000)
    monkeypatch.setattr(file_scanner, "_sha256", lambda path: "sha")
    monkeypatch.setattr(
        file_scanner,
        "extract_tags",
        lambda path: (_ for _ in ()).throw(AssertionError("tag extraction should not run")),
    )

    files = file_scanner.scan_files(config, store, obs_cache=obs_cache)

    assert len(files) == 1
    assert files[0].embedded_title == "Title"
    assert files[0].embedded_artist == "Artist"
    assert files[0].embedded_key_camelot == "9A"
    assert files[0].embedded_comment == "9A|E3|MEL|INST|DRK.TECH.HOUS.DRV"


def test_scan_files_prunes_records_for_files_removed_from_disk(tmp_path, monkeypatch):
    """A FileRecord whose path no longer exists on disk should be pruned, along with its observations."""
    monkeypatch.chdir(tmp_path)
    reset_cache()

    present = tmp_path / "Present.mp3"
    present.write_bytes(b"fake")
    gone_path = str(tmp_path / "Gone.mp3")

    config = RegistryConfig(
        library_roots=[str(tmp_path)],
        supported_extensions=[".mp3"],
        output_dir=str(tmp_path / "registry"),
    )
    store = CsvStore(config.output_dir)
    store.save_files([
        FileRecord(
            file_id="F-present",
            path_abs=str(present.resolve()),
            file_name="Present.mp3",
            audio_duration_sec=180.0,
        ),
        FileRecord(
            file_id="F-gone",
            path_abs=gone_path,
            file_name="Gone.mp3",
            audio_duration_sec=200.0,
        ),
    ])
    store.add_observations([
        SourceObservation(observation_id="O-gone", file_id="F-gone", source_system="tag"),
        SourceObservation(observation_id="O-keep", file_id="F-present", source_system="tag"),
    ])
    store.save_payload_index([
        PayloadIndexEntry(payload_ref="P-gone", file_id="F-gone", source_system="tag", source_type="json"),
        PayloadIndexEntry(payload_ref="P-keep", file_id="F-present", source_system="tag", source_type="json"),
    ])

    monkeypatch.setattr(file_scanner, "_audio_info", lambda path: {"duration": 180.0, "sample_rate": 44100, "channels": 2})
    monkeypatch.setattr(file_scanner, "_bitrate", lambda path: 320000)
    monkeypatch.setattr(file_scanner, "_sha256", lambda path: "sha")
    monkeypatch.setattr(file_scanner, "extract_tags", lambda path: {
        "title": "", "artist": "", "album": "", "genre": "", "bpm": "",
        "key_raw": "", "key_standard": "", "key_camelot": "", "comment": "", "isrc": "",
    })

    files = file_scanner.scan_files(config, store)

    remaining_paths = {f.path_abs for f in files}
    assert str(present.resolve()) in remaining_paths
    assert gone_path not in remaining_paths

    obs_file_ids = {o.file_id for o in store.load_observations()}
    assert "F-gone" not in obs_file_ids
    assert "F-present" in obs_file_ids

    payload_file_ids = {p.file_id for p in store.load_payload_index()}
    assert "F-gone" not in payload_file_ids
    assert "F-present" in payload_file_ids

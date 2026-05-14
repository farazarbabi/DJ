"""Tests for registry file scanner cache behavior."""

from dj_registry.adapters import file_scanner
from dj_registry.config import RegistryConfig
from dj_registry.models import SourceObservation
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
            comments="9A_126_E3_MEL_INST_DRK.TECH.HOUS.DRV",
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
    assert files[0].embedded_comment == "9A_126_E3_MEL_INST_DRK.TECH.HOUS.DRV"

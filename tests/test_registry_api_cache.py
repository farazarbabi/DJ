"""Tests for API lookup caches."""

from dj_registry.adapters import spotify_isrc, songstats
from dj_registry.config import RegistryConfig
from dj_registry.models import FileRecord, LogicalTrack
from dj_registry.store.csv_store import CsvStore
from dj_tagger.universal_cache import get_cache, reset_cache


def test_spotify_cached_not_found_skips_api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")
    reset_cache()

    class FailingSpotifyClient:
        def __init__(self, client_id: str, client_secret: str) -> None:
            pass

        def search_track(self, artist: str, title: str, duration_sec: float = 0.0) -> dict | None:
            raise AssertionError("Spotify API should not be called")

    monkeypatch.setattr(spotify_isrc, "SpotifyClient", FailingSpotifyClient)

    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks([
        LogicalTrack(track_id="T1", artist_canonical="Artist", title_canonical="Title"),
    ])
    path = str(tmp_path / "Track.mp3")
    store.save_files([
        FileRecord(file_id="F1", track_id="T1", is_primary_file=True, path_abs=path, audio_duration_sec=180.0),
    ])
    cache = get_cache("cache/raw_cache.pkl")
    cache.put_track(path, 180.0, "spotify", {"isrc": "", "status": "not_found"})
    cache.save()

    assert spotify_isrc.enrich_isrcs(store) == 0
    assert store.load_tracks()[0].isrc_canonical == ""


def test_songstats_cached_not_found_skips_api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reset_cache()

    class FailingSongstatsClient:
        def __init__(self, config: RegistryConfig) -> None:
            pass

        def fetch_track_by_isrc(self, isrc: str) -> dict | None:
            raise AssertionError("Songstats API should not be called")

    monkeypatch.setattr(songstats, "SongstatsClient", FailingSongstatsClient)

    config = RegistryConfig(output_dir=str(tmp_path / "registry"), songstats_api_key="api-key")
    store = CsvStore(config.output_dir)
    store.save_tracks([
        LogicalTrack(track_id="T1", title_canonical="Title", isrc_canonical="USRC12345"),
    ])
    cache = get_cache("cache/raw_cache.pkl")
    cache.put_isrc("USRC12345", "songstats_lookup", {"status": "not_found"})
    cache.save()

    stats = songstats.ingest_songstats(config, store, obs_cache=None)

    assert stats["total"] == 0
    assert stats["cached"] == 0
    assert stats["fetched"] == 0

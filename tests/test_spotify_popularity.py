"""Tests for the Spotify popularity adapter."""

from dj_registry.adapters import spotify_popularity
from dj_registry.adapters.spotify_popularity import ingest_spotify_popularity
from dj_registry.models import LogicalTrack, SourceObservation
from dj_registry.store.csv_store import CsvStore
from dj_registry.store.obs_cache import ObsCache
from dj_tagger.universal_cache import reset_cache


class _FakeClient:
    def __init__(self, *_args, popularity_by_id=None, isrc_to_id=None) -> None:
        self.popularity_by_id = popularity_by_id or {}
        self.isrc_to_id = isrc_to_id or {}
        self.get_track_calls: list[str] = []
        self.find_calls: list[str] = []

    def get_track(self, spotify_id: str) -> dict | None:
        self.get_track_calls.append(spotify_id)
        if spotify_id in self.popularity_by_id:
            return {"id": spotify_id, "popularity": self.popularity_by_id[spotify_id]}
        return None

    def find_id_by_isrc(self, isrc: str) -> str:
        self.find_calls.append(isrc)
        return self.isrc_to_id.get(isrc, "")


def _set_creds(monkeypatch) -> None:
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "secret")


def test_uses_existing_songstats_spotify_id(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _set_creds(monkeypatch)
    reset_cache()

    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks([
        LogicalTrack(track_id="T1", artist_canonical="A", title_canonical="T", isrc_canonical="USRC12345"),
    ])
    store.save_observations([
        SourceObservation(
            observation_id="OBS-ss-USRC12345",
            track_id="T1",
            source_system="songstats",
            spotify_id="SPOT_ABC",
        )
    ])

    fake = _FakeClient(popularity_by_id={"SPOT_ABC": 73})
    monkeypatch.setattr(spotify_popularity, "SpotifyClient", lambda *_a, **_k: fake)

    stats = ingest_spotify_popularity(store, obs_cache=ObsCache())

    assert stats["fetched"] == 1
    assert stats["skipped"] == 0
    assert fake.find_calls == [], "should not need ISRC search when songstats already has spotify_id"
    assert fake.get_track_calls == ["SPOT_ABC"]

    obs = [o for o in store.load_observations() if o.source_system == "spotify"]
    assert len(obs) == 1
    assert obs[0].popularity == "73"
    assert obs[0].spotify_id == "SPOT_ABC"


def test_falls_back_to_isrc_search(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _set_creds(monkeypatch)
    reset_cache()

    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks([
        LogicalTrack(track_id="T1", artist_canonical="A", title_canonical="T", isrc_canonical="USRC99999"),
    ])

    fake = _FakeClient(
        popularity_by_id={"SPOT_XYZ": 12},
        isrc_to_id={"USRC99999": "SPOT_XYZ"},
    )
    monkeypatch.setattr(spotify_popularity, "SpotifyClient", lambda *_a, **_k: fake)

    stats = ingest_spotify_popularity(store, obs_cache=ObsCache())

    assert stats["fetched"] == 1
    assert fake.find_calls == ["USRC99999"]
    assert fake.get_track_calls == ["SPOT_XYZ"]

    obs = [o for o in store.load_observations() if o.source_system == "spotify"]
    assert obs and obs[0].popularity == "12"


def test_skips_when_no_credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("SPOTIFY_CLIENT_SECRET", raising=False)
    reset_cache()

    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks([
        LogicalTrack(track_id="T1", isrc_canonical="USRC0"),
    ])

    def boom(*_a, **_k):
        raise AssertionError("client should not be constructed when creds missing")

    monkeypatch.setattr(spotify_popularity, "SpotifyClient", boom)
    stats = ingest_spotify_popularity(store)
    assert stats == {"total": 0, "cached": 0, "fetched": 0, "skipped": 0, "candidates": 0}


def test_cache_hit_skips_api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _set_creds(monkeypatch)
    reset_cache()

    store = CsvStore(str(tmp_path / "registry"))
    store.save_tracks([
        LogicalTrack(track_id="T1", isrc_canonical="USRC42"),
    ])
    obs_cache = ObsCache()
    obs_cache.put_by_isrc("USRC42", "spotify", SourceObservation(
        observation_id="OBS-spop-USRC42",
        track_id="T1",
        source_system="spotify",
        spotify_id="SPOT_OLD",
        popularity="55",
    ))

    fake = _FakeClient()  # any API call would be a miss
    monkeypatch.setattr(spotify_popularity, "SpotifyClient", lambda *_a, **_k: fake)

    stats = ingest_spotify_popularity(store, obs_cache=obs_cache)

    assert stats["cached"] == 1
    assert stats["fetched"] == 0
    assert fake.get_track_calls == []
    assert fake.find_calls == []

    obs = [o for o in store.load_observations() if o.source_system == "spotify"]
    assert obs and obs[0].popularity == "55"

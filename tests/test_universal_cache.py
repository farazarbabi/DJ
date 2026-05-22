"""Tests for the universal cache."""

import os

import numpy as np
import pytest

import dj_tagger.universal_cache as universal_cache
from dj_tagger.universal_cache import (
    UniversalCache, CacheEntry, LAYER_VERSIONS, reset_cache,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton between tests."""
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def cache(tmp_path):
    path = str(tmp_path / "universal_cache.pkl")
    return UniversalCache(path)


def test_put_and_get(cache):
    cache.put("mykey", {"hello": "world"}, version="1")
    assert cache.get("mykey") == {"hello": "world"}


def test_get_missing_returns_none(cache):
    assert cache.get("nonexistent") is None


def test_version_mismatch_returns_none(cache):
    cache.put("mykey", {"hello": "world"}, version="1")
    assert cache.get("mykey", version="2") is None
    assert cache.get("mykey", version="1") == {"hello": "world"}


def test_track_key_with_duration():
    key = UniversalCache.track_key("track.aiff", 180.5, "tagger")
    assert key == "track.aiff|180.5|tagger"


def test_track_key_without_duration():
    key = UniversalCache.track_key("track.aiff", None, "dsp")
    assert key == "track.aiff|dsp"


def test_isrc_key():
    key = UniversalCache.isrc_key("USRC12345", "songstats")
    assert key == "isrc:USRC12345|songstats"


def test_put_track_with_duration_writes_single_key(cache):
    """put_track with duration should write only the duration-keyed entry."""
    cache.put_track("track.aiff", 180.5, "dsp", {"rms": 0.3})
    # Should be findable with duration
    assert cache.get_track("track.aiff", 180.5, "dsp") == {"rms": 0.3}
    # Name-only key should NOT exist (no double-write)
    assert cache.get("track.aiff|dsp") is None


def test_put_track_without_duration_writes_name_only(cache):
    """put_track without duration falls back to name-only key."""
    cache.put_track("track.aiff", None, "dsp", {"rms": 0.3})
    assert cache.get_track("track.aiff", None, "dsp") == {"rms": 0.3}


def test_get_track_requires_duration_match(cache):
    """get_track should not find entries stored with a different duration."""
    cache.put_track("track.aiff", 180.5, "dsp", {"rms": 0.3})
    assert cache.get_track("track.aiff", 180.5, "dsp") == {"rms": 0.3}
    assert cache.get_track("track.aiff", 200.0, "dsp") is None


def test_isrc_put_and_get(cache):
    cache.put_isrc("USRC12345", "songstats", {"danceability": 0.7})
    result = cache.get_isrc("USRC12345", "songstats")
    assert result == {"danceability": 0.7}
    assert cache.has_isrc("USRC12345", "songstats")
    assert not cache.has_isrc("USRC99999", "songstats")


def test_save_and_reload(tmp_path):
    path = str(tmp_path / "test_cache.pkl")
    cache = UniversalCache(path)
    cache.put_track("track.aiff", 180.5, "tagger", {"energy": 3})
    cache.put_track("track.aiff", 180.5, "dsp", {"rms": 0.3})
    cache.save()

    cache2 = UniversalCache(path)
    assert cache2.get_track("track.aiff", 180.5, "tagger") == {"energy": 3}
    assert cache2.get_track("track.aiff", 180.5, "dsp") == {"rms": 0.3}


def test_cross_module_sharing(tmp_path):
    """Tagger analysis should be visible to grouper lookups."""
    path = str(tmp_path / "shared_cache.pkl")
    cache = UniversalCache(path)

    tagger_result = {
        "file": "track.aiff",
        "tag": "9A|E3|HYPN|INST",
        "energy": 3,
        "camelot": "9A",
        "vibe": "HYPN",
        "vocal": "INST",
        "bpm": 126,
    }
    cache.put_track("track.aiff", 180.5, "tagger", tagger_result)
    cache.save()

    cache2 = UniversalCache(path)
    result = cache2.get_track("track.aiff", 180.5, "tagger")
    assert result is not None
    assert result["energy"] == 3
    assert result["camelot"] == "9A"


def test_ndarray_storage(cache):
    """CLAP embeddings (numpy arrays) should survive serialization."""
    embedding = np.random.rand(512).astype(np.float32)
    cache.put_track("track.aiff", 180.5, "clap", embedding)
    cache.force_save()

    cache2 = UniversalCache(cache.path)
    result = cache2.get_track("track.aiff", 180.5, "clap")
    np.testing.assert_array_almost_equal(result, embedding)


def test_empty_cache(tmp_path):
    path = str(tmp_path / "empty_cache.pkl")
    cache = UniversalCache(path)
    assert len(cache) == 0
    assert cache.get("anything") is None


def test_layers_isolated(cache):
    """Different layers for the same track should be independent."""
    cache.put_track("track.aiff", 180.5, "tagger", {"energy": 3})
    cache.put_track("track.aiff", 180.5, "dsp", {"rms": 0.3})
    cache.put_track("track.aiff", 180.5, "clap", [0.1, 0.2])

    assert cache.get_track("track.aiff", 180.5, "tagger") == {"energy": 3}
    assert cache.get_track("track.aiff", 180.5, "dsp") == {"rms": 0.3}
    assert cache.get_track("track.aiff", 180.5, "clap") == [0.1, 0.2]


def test_layer_version_auto_check(cache):
    """get_track should auto-check LAYER_VERSIONS for known layers."""
    # Store with correct tagger version
    cache.put_track("track.aiff", 180.5, "tagger", {"energy": 3})
    assert cache.get_track("track.aiff", 180.5, "tagger") == {"energy": 3}

    # Store with wrong version directly
    key = cache.track_key("track2.aiff", 180.5, "tagger")
    cache.put(key, {"energy": 99}, version="old_version")
    # Should return None because version doesn't match LAYER_VERSIONS["tagger"]
    assert cache.get_track("track2.aiff", 180.5, "tagger") is None


def test_layer_version_auto_set(cache):
    """put_track should auto-set version from LAYER_VERSIONS."""
    cache.put_track("track.aiff", 180.5, "tagger", {"energy": 3})
    key = cache.track_key("track.aiff", 180.5, "tagger")
    entry = cache._entries[key]
    assert entry.version == LAYER_VERSIONS["tagger"]


def test_raw_layers_ignore_version_mismatch(cache):
    """Raw/data-collection layers are keyed only by filename + duration + layer."""
    key = cache.track_key("track.aiff", 180.5, "raw_analysis")
    cache.put(key, {"tempo": 128.0}, version="old_raw_version")
    assert cache.get_track("track.aiff", 180.5, "raw_analysis") == {"tempo": 128.0}


def test_raw_api_layers_ignore_version_mismatch(cache):
    """ISRC-based API observations are also raw/data-collection layers."""
    key = cache.isrc_key("USRC12345", "songstats_lookup")
    cache.put(key, {"status": "not_found"}, version="old_api_version")
    assert cache.get_isrc("USRC12345", "songstats_lookup") == {"status": "not_found"}


def test_raw_layers_are_not_signature_versioned(monkeypatch):
    """Downstream signature churn must not invalidate collected raw layers."""
    import dj_tagger.settings as settings

    monkeypatch.setattr(settings, "dsp_version", lambda: "dsp-v")
    monkeypatch.setattr(settings, "section_dsp_version", lambda: "section-v")
    monkeypatch.setattr(settings, "raw_analysis_version", lambda: "analysis-v")

    versions = universal_cache.refresh_layer_versions()

    assert versions["dsp"] == "1"
    assert versions["section_dsp"] == "1"
    assert versions["raw_analysis"] == "1"


def test_direct_tagger_cache_rederives_from_loaded_raw_cache(tmp_path, monkeypatch):
    """Stale direct tagger cache entries should use raw cache, not audio."""
    import dj_tagger.derive as derive
    from dj_tagger.cache import TaggerCacheEntry, get_cached, load_cache

    cache_path = str(tmp_path / "tagger_cache.pkl")
    cache = load_cache(cache_path)
    raw_cache = universal_cache.get_cache(str(tmp_path / "raw_cache.pkl"))
    raw_cache.put_track("Track.mp3", 123.4, "dsp", {"raw": "dsp"})
    raw_cache.put_track("Track.mp3", 123.4, "raw_analysis", {"raw": "analysis"})

    monkeypatch.setattr(derive, "derive_all", lambda dsp, raw: {"energy": 4, "vibe": "MEL"})
    cache["Track.mp3|123.4"] = TaggerCacheEntry(
        mtime=1.0,
        version="old-version",
        result={"energy": 2, "camelot": "9A"},
    )

    result = get_cached(cache, str(tmp_path / "Track.mp3"), 99.0, duration=123.4)

    assert result is not None
    assert result["energy"] == 4
    assert result["vibe"] == "MEL"
    assert result["camelot"] == "9A"


def test_atomic_save_creates_files(tmp_path):
    """Atomic save should create raw and/or derived files."""
    path = str(tmp_path / "raw_cache.pkl")
    cache = UniversalCache(path)
    cache.put_track("track.aiff", 180.0, "dsp", {"rms": 0.3})  # raw
    cache.put_track("track.aiff", 180.0, "tagger", {"energy": 3})  # derived
    cache.save()
    assert os.path.exists(str(tmp_path / "raw_cache.pkl"))
    assert os.path.exists(str(tmp_path / "derived_cache.pkl"))


def test_atomic_save_no_temp_files(tmp_path):
    """After save, no .pkl.tmp files should remain."""
    path = str(tmp_path / "raw_cache.pkl")
    cache = UniversalCache(path)
    cache.put_track("track.aiff", 180.0, "dsp", {"rms": 0.3})
    cache.save()
    tmp_files = list(tmp_path.glob("*.pkl.tmp"))
    assert len(tmp_files) == 0


def test_unknown_layer_no_version_check(cache):
    """Layers not in LAYER_VERSIONS should not be version-checked."""
    cache.put_track("track.aiff", 180.5, "custom_layer", {"x": 1}, version="any")
    # Should return the data since "custom_layer" has no version in LAYER_VERSIONS
    assert cache.get_track("track.aiff", 180.5, "custom_layer") == {"x": 1}

"""Phase 1 — cache-backed dense feature wiring + popularity scalar."""

from __future__ import annotations

import os
import sys

import pytest

from dj_registry.models import FileRecord, LogicalTrack, SourceObservation
from dj_registry.taxonomy.features import (
    _cache_key_from_file_record,
    _popularity_band,
    build_track_features,
)


class _FakeCache:
    """Minimal stand-in for UniversalCache.get_track."""

    def __init__(self, payloads: dict[tuple[str, str], object]) -> None:
        self._payloads = payloads

    def get_track(self, filename, duration, layer):
        # Ignore duration for test simplicity; key by (filename, layer).
        return self._payloads.get((filename, layer))


def _track_and_file(filename: str = "track.mp3"):
    return (
        LogicalTrack(track_id="T1", title_canonical="X", tagger_energy="E4"),
        FileRecord(
            track_id="T1",
            file_name=filename,
            audio_duration_sec=180.0,
        ),
    )


def test_dsp_features_present_when_cache_layer_populated():
    track, fr = _track_and_file()
    cache = _FakeCache({
        ("track.mp3", "dsp"): {
            "onset_density": 2.4,
            "centroid_mean": 1700.0,
            "rms_mean": 0.18,
            "mfcc_1_mean": -12.3,
            "tonnetz_0_mean": 0.04,
        },
    })

    features = build_track_features(track, [], fr, ucache=cache)

    assert features.get("num:dsp:onset_density") == pytest.approx(2.4)
    assert features.get("num:dsp:centroid_mean") == pytest.approx(1700.0)
    assert features.get("num:dsp:rms_mean") == pytest.approx(0.18)
    assert features.get("num:dsp:mfcc_1_mean") == pytest.approx(-12.3)
    assert features.get("num:dsp:tonnetz_0_mean") == pytest.approx(0.04)


def test_dsp_features_absent_when_no_cache():
    track, fr = _track_and_file()
    features = build_track_features(track, [], fr)
    assert not any(k.startswith("num:dsp:") for k in features)


def test_dsp_features_absent_on_cache_miss():
    track, fr = _track_and_file()
    empty_cache = _FakeCache({})
    features = build_track_features(track, [], fr, ucache=empty_cache)
    assert not any(k.startswith("num:dsp:") for k in features)


def test_dsp_features_skipped_in_internal_feature_mode():
    """DSP scalars are part of the external feature set; internal mode must
    not pick them up so the dual-model comparison stays apples-to-apples."""
    track, fr = _track_and_file()
    cache = _FakeCache({
        ("track.mp3", "dsp"): {"onset_density": 2.4, "centroid_mean": 1700.0},
    })
    features = build_track_features(track, [], fr, feature_mode="internal", ucache=cache)
    assert not any(k.startswith("num:dsp:") for k in features)


def test_popularity_feature_emitted_when_observation_has_popularity():
    track, fr = _track_and_file()
    obs = SourceObservation(
        track_id="T1",
        source_system="spotify",
        popularity="72",
    )
    features = build_track_features(track, [obs], fr)

    assert features.get("num:spotify:popularity") == pytest.approx(0.72)
    assert features.get("popularity_band:spotify:hi") == 1.0


def test_popularity_feature_skipped_when_blank():
    track, fr = _track_and_file()
    obs = SourceObservation(track_id="T1", source_system="spotify", popularity="")
    features = build_track_features(track, [obs], fr)
    assert not any(k.startswith("num:spotify:popularity") for k in features)
    assert not any(k.startswith("popularity_band:") for k in features)


def test_popularity_feature_bands_low_mid_high():
    """3-band popularity covers the Spotify 0-100 range mapped to 0-1."""
    assert _popularity_band(0.10) == "lo"
    assert _popularity_band(0.45) == "mid"
    assert _popularity_band(0.85) == "hi"


def test_cache_key_resolution_uses_file_name_first_then_falls_back_to_path():
    fr_with_name = FileRecord(track_id="T1", file_name="abc.mp3", audio_duration_sec=200.0)
    key = _cache_key_from_file_record(fr_with_name)
    assert key is not None
    assert key[0] == "abc.mp3"
    # Duration: quick_duration on a non-existent path falls back to audio_duration_sec
    assert key[1] == pytest.approx(200.0)


def test_cache_key_resolution_returns_none_when_no_filename():
    fr_empty = FileRecord(track_id="T1")
    assert _cache_key_from_file_record(fr_empty) is None
    assert _cache_key_from_file_record(None) is None


def test_dsp_features_no_op_when_filename_unknown():
    """Even when ucache is provided, cache_key=None makes the helper a no-op."""
    track = LogicalTrack(track_id="T1", tagger_energy="E4")
    cache = _FakeCache({("track.mp3", "dsp"): {"onset_density": 2.4}})
    features = build_track_features(track, [], None, ucache=cache)
    assert not any(k.startswith("num:dsp:") for k in features)


# ── CLAP embedding wiring ──────────────────────────────────────────────────


def test_clap_features_emit_512_unit_norm_dims_when_cached():
    import numpy as np

    track, fr = _track_and_file()
    embedding = np.arange(512, dtype=np.float32)  # non-trivial vector
    cache = _FakeCache({("track.mp3", "clap"): embedding})

    features = build_track_features(track, [], fr, ucache=cache)

    clap_keys = [k for k in features if k.startswith("clap:d")]
    assert len(clap_keys) == 512
    # L2-normalized: sum of squares ~= 1
    sum_sq = sum(features[k] ** 2 for k in clap_keys)
    assert sum_sq == pytest.approx(1.0, abs=1e-4)


def test_clap_features_absent_on_cache_miss():
    track, fr = _track_and_file()
    empty_cache = _FakeCache({})
    features = build_track_features(track, [], fr, ucache=empty_cache)
    assert not any(k.startswith("clap:") for k in features)


def test_clap_features_skip_zero_norm_vectors():
    """A zero (or NaN) CLAP vector should not pollute the feature dict."""
    import numpy as np

    track, fr = _track_and_file()
    cache = _FakeCache({("track.mp3", "clap"): np.zeros(512, dtype=np.float32)})

    features = build_track_features(track, [], fr, ucache=cache)
    assert not any(k.startswith("clap:") for k in features)


# ── Demucs vocal_stem wiring ───────────────────────────────────────────────


def test_vocal_stem_features_emit_four_scalars_when_cached():
    track, fr = _track_and_file()
    cache = _FakeCache({
        ("track.mp3", "vocal_stem"): {
            "vocal_stem_rms_db": -18.4,
            "vocal_stem_mix_ratio_db": -4.2,
            "vocal_stem_activity_frac": 0.62,
            "vocal_stem_envelope_var": 2.31,
            "vocal_stem_n_slices": 5,  # not a feature, just metadata
        },
    })
    features = build_track_features(track, [], fr, ucache=cache)
    assert features.get("num:vocal_stem:rms_db") == pytest.approx(-18.4)
    assert features.get("num:vocal_stem:mix_ratio_db") == pytest.approx(-4.2)
    assert features.get("num:vocal_stem:activity_frac") == pytest.approx(0.62)
    assert features.get("num:vocal_stem:envelope_var") == pytest.approx(2.31)


def test_vocal_stem_features_absent_on_cache_miss():
    track, fr = _track_and_file()
    empty_cache = _FakeCache({})
    features = build_track_features(track, [], fr, ucache=empty_cache)
    assert not any(k.startswith("num:vocal_stem:") for k in features)


def test_vocal_stem_features_skipped_in_internal_feature_mode():
    track, fr = _track_and_file()
    cache = _FakeCache({
        ("track.mp3", "vocal_stem"): {"vocal_stem_rms_db": -18.4},
    })
    features = build_track_features(track, [], fr, feature_mode="internal", ucache=cache)
    assert not any(k.startswith("num:vocal_stem:") for k in features)

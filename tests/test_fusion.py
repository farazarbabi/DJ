"""Tests for PCA fusion module."""

import numpy as np
import pytest

from dj_grouper.features.builder import TrackFeatures, encode_tags, build_unified_vector
from dj_grouper.features.fusion import fit_unified_pca, apply_unified_pca
from dj_grouper.features.registry_bridge import RegistryEnrichment
from dj_grouper.scanner import TrackInfo


def _make_track_with_unified(
    energy=3, key="9A", bpm=128, vibe="HYPN", vocal="NV",
    danceability=0.7, valence=0.4, genre="Techno",
    path="test.aiff",
):
    info = TrackInfo(
        path=path, energy=energy, key=key, bpm=bpm,
        structure="64H", intro_bars=64, flow_type="H",
        vibe=vibe, vocal=vocal,
    )
    tag_vec = encode_tags(info)
    dsp_vec = np.random.RandomState(hash(path) % 2**31).rand(21).astype(np.float32)
    registry = RegistryEnrichment(
        danceability=danceability,
        valence=valence,
        primary_genre=genre,
        all_genres=[genre],
        has_songstats=True,
    )
    unified = build_unified_vector(tag_vec, dsp_vec, None, registry, clap_pca_dims=32)
    return TrackFeatures(
        path=path, info=info, tag_vector=tag_vec,
        dsp_vector=dsp_vec, unified_vector=unified, registry=registry,
    )


def test_fit_unified_pca_reduces_dims():
    tracks = [_make_track_with_unified(path=f"t{i}.aiff", bpm=120+i) for i in range(20)]
    reduced, params = fit_unified_pca(tracks, target_variance=0.90)
    assert reduced.shape[0] == 20
    assert reduced.shape[1] < tracks[0].unified_vector.shape[0]
    assert reduced.shape[1] >= 2


def test_pca_output_l2_normalized():
    tracks = [_make_track_with_unified(path=f"t{i}.aiff", bpm=120+i) for i in range(15)]
    reduced, _ = fit_unified_pca(tracks)
    norms = np.linalg.norm(reduced, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(15), decimal=5)


def test_apply_pca_consistent():
    tracks = [_make_track_with_unified(path=f"t{i}.aiff", bpm=120+i) for i in range(15)]
    reduced_fit, params = fit_unified_pca(tracks)
    reduced_apply = apply_unified_pca(tracks, params)
    np.testing.assert_array_almost_equal(reduced_fit, reduced_apply, decimal=4)


def test_unified_vector_shape():
    """Unified vector should be 80 dims: 19 tag + 21 dsp + 32 clap + 6 genre + 2 songstats."""
    info = TrackInfo(path="t.aiff", energy=3, key="9A", bpm=128,
                     structure="64H", intro_bars=64, flow_type="H",
                     vibe="HYPN", vocal="NV")
    tag_vec = encode_tags(info)
    dsp_vec = np.zeros(21, dtype=np.float32)
    registry = RegistryEnrichment(danceability=0.5, valence=0.5, has_songstats=True)
    unified = build_unified_vector(tag_vec, dsp_vec, None, registry, clap_pca_dims=32)
    # 19 + 21 + 32 + 6 + 2 = 80
    assert unified.shape == (80,)


def test_unified_vector_no_registry():
    """Without registry, genre and songstats should be neutral."""
    info = TrackInfo(path="t.aiff", energy=3, key="9A", bpm=128,
                     structure="64H", intro_bars=64, flow_type="H",
                     vibe="HYPN", vocal="NV")
    tag_vec = encode_tags(info)
    dsp_vec = np.zeros(21, dtype=np.float32)
    unified = build_unified_vector(tag_vec, dsp_vec, None, None, clap_pca_dims=32)
    assert unified.shape == (80,)
    # Genre dims should be 0.5 (neutral)
    genre_start = 19 + 21 + 32  # after tag + dsp + clap
    np.testing.assert_array_almost_equal(unified[genre_start:genre_start+6], [0.5]*6)
    # Songstats dims should be 0.5 (neutral)
    np.testing.assert_array_almost_equal(unified[-2:], [0.5, 0.5])


def test_unified_vector_with_clap():
    """When CLAP is provided, it should be included in the unified vector."""
    info = TrackInfo(path="t.aiff", energy=3, key="9A", bpm=128,
                     structure="64H", intro_bars=64, flow_type="H",
                     vibe="HYPN", vocal="NV")
    tag_vec = encode_tags(info)
    dsp_vec = np.zeros(21, dtype=np.float32)
    clap_vec = np.ones(32, dtype=np.float32) * 0.3
    unified = build_unified_vector(tag_vec, dsp_vec, clap_vec, None, clap_pca_dims=32)
    # CLAP section should be 0.3
    np.testing.assert_array_almost_equal(unified[40:72], [0.3]*32)

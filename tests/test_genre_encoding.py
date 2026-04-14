"""Tests for genre encoding."""

import numpy as np

from dj_grouper.features.genre_encoding import (
    encode_genres,
    GENRE_DIM,
    NEUTRAL_VECTOR,
    GENRE_VECTORS,
)


def test_known_genre_returns_correct_vector():
    vec = encode_genres(["Techno"])
    expected = np.array(GENRE_VECTORS["Techno"], dtype=np.float32)
    np.testing.assert_array_almost_equal(vec, expected)


def test_multi_genre_averages():
    vec = encode_genres(["Techno", "Deep House"])
    v1 = np.array(GENRE_VECTORS["Techno"], dtype=np.float32)
    v2 = np.array(GENRE_VECTORS["Deep House"], dtype=np.float32)
    expected = (v1 + v2) / 2
    np.testing.assert_array_almost_equal(vec, expected)


def test_unknown_genre_returns_neutral():
    vec = encode_genres(["Completely Unknown Genre"])
    expected = np.array(NEUTRAL_VECTOR, dtype=np.float32)
    np.testing.assert_array_almost_equal(vec, expected)


def test_empty_genres_returns_neutral():
    vec = encode_genres([])
    expected = np.array(NEUTRAL_VECTOR, dtype=np.float32)
    np.testing.assert_array_almost_equal(vec, expected)


def test_case_insensitive():
    vec_lower = encode_genres(["techno"])
    vec_title = encode_genres(["Techno"])
    np.testing.assert_array_almost_equal(vec_lower, vec_title)


def test_output_shape():
    vec = encode_genres(["Melodic Techno"])
    assert vec.shape == (GENRE_DIM,)
    assert vec.dtype == np.float32


def test_fuzzy_match():
    """Genre containing a known subgenre name should match."""
    vec = encode_genres(["melodic techno & progressive house"])
    # Should fuzzy-match to something, not return neutral
    neutral = np.array(NEUTRAL_VECTOR, dtype=np.float32)
    assert not np.allclose(vec, neutral)


def test_mixed_known_unknown():
    """Mix of known and unknown genres should average only the known ones."""
    vec = encode_genres(["Techno", "Some Unknown Genre"])
    expected = np.array(GENRE_VECTORS["Techno"], dtype=np.float32)
    np.testing.assert_array_almost_equal(vec, expected)


def test_all_genre_vectors_valid():
    """All genre vectors should be 6-dim with values in [0, 1]."""
    for name, vec in GENRE_VECTORS.items():
        assert len(vec) == GENRE_DIM, f"{name} has wrong dimension"
        for i, v in enumerate(vec):
            assert 0.0 <= v <= 1.0, f"{name}[{i}] = {v} out of [0, 1]"

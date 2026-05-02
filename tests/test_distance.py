"""Tests for distance computation."""

import numpy as np

from dj_grouper.config import GrouperConfig
from dj_grouper.features.builder import TrackFeatures, encode_tags
from dj_grouper.scanner import TrackInfo
from dj_grouper.grouping.distance import (
    blended_distance, tag_distance, dsp_distance, compute_distance_matrix,
)


def _make_track(energy=3, key="9A", bpm=128, structure="64H",
                vibe="HYPN", vocal="INST", path="test.aiff"):
    info = TrackInfo(
        path=path, energy=energy, key=key, bpm=bpm,
        structure=structure, intro_bars=int(structure[:-1]),
        flow_type=structure[-1], vibe=vibe, vocal=vocal,
    )
    tag_vec = encode_tags(info)
    dsp_vec = np.zeros(21, dtype=np.float32)
    return TrackFeatures(path=path, info=info, tag_vector=tag_vec, dsp_vector=dsp_vec)


def test_identical_tracks_zero_distance():
    config = GrouperConfig()
    a = _make_track()
    d = blended_distance(a, a, config)
    assert d == 0.0


def test_different_vibe_increases_distance():
    config = GrouperConfig()
    a = _make_track(vibe="HYPN")
    b = _make_track(vibe="RAW")
    d = blended_distance(a, b, config)
    assert d > 0.0


def test_vocal_mismatch_increases_tag_distance():
    config = GrouperConfig()
    a = _make_track(vocal="INST")
    b = _make_track(vocal="VOC")
    d_mismatch = tag_distance(a, b, config)
    c = _make_track(vocal="INST")
    d_match = tag_distance(a, c, config)
    assert d_mismatch > d_match


def test_distance_matrix_symmetric():
    config = GrouperConfig()
    tracks = [
        _make_track(energy=2, path="a.aiff"),
        _make_track(energy=4, path="b.aiff"),
        _make_track(energy=3, vibe="DRK", path="c.aiff"),
    ]
    matrix = compute_distance_matrix(tracks, config)
    assert matrix.shape == (3, 3)
    np.testing.assert_array_almost_equal(matrix, matrix.T)
    assert matrix[0, 0] == 0.0


def test_key_weight_higher_for_melodic():
    config = GrouperConfig()
    a_mel = _make_track(vibe="MEL", key="1A")
    b_mel = _make_track(vibe="MEL", key="7A")
    a_raw = _make_track(vibe="RAW", key="1A")
    b_raw = _make_track(vibe="RAW", key="7A")

    d_mel = tag_distance(a_mel, b_mel, config)
    d_raw = tag_distance(a_raw, b_raw, config)
    assert d_mel > d_raw

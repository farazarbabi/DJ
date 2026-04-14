"""Tests for constraint generation."""

import numpy as np

from dj_grouper.config import GrouperConfig
from dj_grouper.features.builder import TrackFeatures, encode_tags
from dj_grouper.scanner import TrackInfo
from dj_grouper.grouping.constraints import build_constraints, Constraints
from dj_grouper.feedback.store import FeedbackEntry


def _make_track(key="9A", bpm=128, path="test.aiff"):
    info = TrackInfo(
        path=path, energy=3, key=key, bpm=bpm,
        structure="64H", intro_bars=64, flow_type="H",
        vibe="HYPN", vocal="NV",
    )
    tag_vec = encode_tags(info)
    dsp_vec = np.zeros(21, dtype=np.float32)
    return TrackFeatures(path=path, info=info, tag_vector=tag_vec, dsp_vector=dsp_vec)


def test_compatible_keys_no_constraint():
    """Same key or Camelot distance 1 should not generate cannot-link."""
    config = GrouperConfig()
    tracks = [
        _make_track(key="9A", bpm=128, path="a.aiff"),
        _make_track(key="10A", bpm=128, path="b.aiff"),  # distance 1
    ]
    constraints = build_constraints(tracks, config)
    assert not constraints.is_cannot_link(0, 1)


def test_incompatible_keys_generate_cannot_link():
    """Camelot distance > 1 should generate cannot-link."""
    config = GrouperConfig()
    tracks = [
        _make_track(key="1A", bpm=128, path="a.aiff"),
        _make_track(key="5A", bpm=128, path="b.aiff"),  # distance 4
    ]
    constraints = build_constraints(tracks, config)
    assert constraints.is_cannot_link(0, 1)


def test_unknown_key_floats_free():
    """Tracks with unknown key should not generate key constraints."""
    config = GrouperConfig()
    tracks = [
        _make_track(key="??", bpm=128, path="a.aiff"),
        _make_track(key="5A", bpm=128, path="b.aiff"),
    ]
    constraints = build_constraints(tracks, config)
    assert not constraints.is_cannot_link(0, 1)


def test_none_key_floats_free():
    tracks = [
        _make_track(key="9A", bpm=128, path="a.aiff"),
        _make_track(key="9A", bpm=128, path="b.aiff"),
    ]
    tracks[1].info.key = None
    config = GrouperConfig()
    constraints = build_constraints(tracks, config)
    assert not constraints.is_cannot_link(0, 1)


def test_bpm_spread_generates_cannot_link():
    """BPM spread > 7% should generate cannot-link."""
    config = GrouperConfig(bpm_group_max_spread_pct=0.07)
    tracks = [
        _make_track(key="9A", bpm=120, path="a.aiff"),
        _make_track(key="9A", bpm=140, path="b.aiff"),  # ~15% spread
    ]
    constraints = build_constraints(tracks, config)
    assert constraints.is_cannot_link(0, 1)


def test_close_bpm_no_constraint():
    """BPM within threshold should not generate cannot-link."""
    config = GrouperConfig(bpm_group_max_spread_pct=0.07)
    tracks = [
        _make_track(key="9A", bpm=125, path="a.aiff"),
        _make_track(key="9A", bpm=128, path="b.aiff"),  # ~2.4% spread
    ]
    constraints = build_constraints(tracks, config)
    assert not constraints.is_cannot_link(0, 1)


def test_feedback_good_pair_creates_must_link():
    config = GrouperConfig()
    tracks = [
        _make_track(key="1A", bpm=128, path="a.aiff"),
        _make_track(key="5A", bpm=128, path="b.aiff"),  # normally cannot-link by key
    ]
    feedback = [FeedbackEntry(track_a="a.aiff", track_b="b.aiff", type="good_pair", strength="1.0")]
    constraints = build_constraints(tracks, config, feedback)
    # good_pair overrides cannot-link
    assert constraints.is_must_link(0, 1)
    assert not constraints.is_cannot_link(0, 1)


def test_feedback_bad_pair_creates_cannot_link():
    config = GrouperConfig()
    tracks = [
        _make_track(key="9A", bpm=128, path="a.aiff"),
        _make_track(key="9A", bpm=128, path="b.aiff"),  # normally compatible
    ]
    feedback = [FeedbackEntry(track_a="a.aiff", track_b="b.aiff", type="bad_pair", strength="1.0")]
    constraints = build_constraints(tracks, config, feedback)
    assert constraints.is_cannot_link(0, 1)


def test_constraints_symmetric():
    c = Constraints()
    c.cannot_link.add((0, 1))
    c.must_link.add((2, 3))
    assert c.is_cannot_link(0, 1)
    assert c.is_cannot_link(1, 0)
    assert c.is_must_link(2, 3)
    assert c.is_must_link(3, 2)

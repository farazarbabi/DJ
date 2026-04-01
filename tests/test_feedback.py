"""Tests for feedback store and apply."""

import numpy as np

from dj_grouper.feedback.store import (
    FeedbackEntry, load_feedback, save_feedback, add_feedback,
)
from dj_grouper.feedback.apply import apply_feedback_to_distances
from dj_grouper.features.builder import TrackFeatures, encode_tags
from dj_grouper.scanner import TrackInfo
from dj_grouper.config import GrouperConfig


def test_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "feedback.csv")
    entries = [
        FeedbackEntry("a.aiff", "b.aiff", "good_pair", "1.0"),
        FeedbackEntry("c.aiff", "d.aiff", "bad_pair", "0.5"),
    ]
    save_feedback(entries, path)
    loaded = load_feedback(path)
    assert len(loaded) == 2
    assert loaded[0].type == "good_pair"
    assert loaded[1].strength == "0.5"


def test_add_feedback(tmp_path):
    path = str(tmp_path / "feedback.csv")
    add_feedback(path, "a.aiff", "b.aiff", "good_pair")
    entries = load_feedback(path)
    assert len(entries) == 1


def test_apply_good_pair_reduces_distance():
    config = GrouperConfig()
    info_a = TrackInfo(path="a.aiff")
    info_b = TrackInfo(path="b.aiff")
    tracks = [
        TrackFeatures("a.aiff", info_a, np.zeros(19, dtype=np.float32), np.zeros(21, dtype=np.float32)),
        TrackFeatures("b.aiff", info_b, np.zeros(19, dtype=np.float32), np.zeros(21, dtype=np.float32)),
    ]
    dist = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    feedback = [FeedbackEntry("a.aiff", "b.aiff", "good_pair", "1.0")]
    modified = apply_feedback_to_distances(dist, tracks, feedback, config)
    assert modified[0, 1] < 1.0


def test_apply_bad_pair_increases_distance():
    config = GrouperConfig()
    info_a = TrackInfo(path="a.aiff")
    info_b = TrackInfo(path="b.aiff")
    tracks = [
        TrackFeatures("a.aiff", info_a, np.zeros(19, dtype=np.float32), np.zeros(21, dtype=np.float32)),
        TrackFeatures("b.aiff", info_b, np.zeros(19, dtype=np.float32), np.zeros(21, dtype=np.float32)),
    ]
    dist = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    feedback = [FeedbackEntry("a.aiff", "b.aiff", "bad_pair", "1.0")]
    modified = apply_feedback_to_distances(dist, tracks, feedback, config)
    assert modified[0, 1] > 1.0

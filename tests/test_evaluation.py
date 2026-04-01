"""Tests for evaluation framework."""

import numpy as np

from dj_grouper.config import GrouperConfig
from dj_grouper.features.builder import TrackFeatures, encode_tags
from dj_grouper.scanner import TrackInfo
from dj_grouper.evaluation import EvalPair, evaluate


def _make_track(energy=3, vibe="HYPN", bpm=128, path="t.aiff"):
    info = TrackInfo(path=path, energy=energy, vibe=vibe, bpm=bpm,
                     structure="64H", intro_bars=64, flow_type="H", vocal="NV")
    return TrackFeatures(
        path=path, info=info,
        tag_vector=encode_tags(info),
        dsp_vector=np.zeros(21, dtype=np.float32),
    )


def test_evaluate_good_pair_scores_higher():
    config = GrouperConfig()
    tracks = [
        _make_track(energy=3, vibe="HYPN", bpm=126, path="a.aiff"),
        _make_track(energy=3, vibe="HYPN", bpm=128, path="b.aiff"),
        _make_track(energy=5, vibe="RAW", bpm=140, path="c.aiff"),
    ]
    pairs = [
        EvalPair("a.aiff", "b.aiff", "good"),
        EvalPair("a.aiff", "c.aiff", "bad"),
    ]
    metrics = evaluate(tracks, pairs, config)
    assert metrics["n_good"] == 1
    assert metrics["n_bad"] == 1
    assert metrics["good_avg_score"] > metrics["bad_avg_score"]

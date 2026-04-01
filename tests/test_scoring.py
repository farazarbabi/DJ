"""Tests for recommendation scoring."""

import numpy as np

from dj_grouper.config import GrouperConfig
from dj_grouper.features.builder import TrackFeatures, encode_tags
from dj_grouper.scanner import TrackInfo
from dj_grouper.recommend.scoring import recommend_score


def _make_track(energy=3, key="9A", bpm=128, structure="64H",
                vibe="HYPN", vocal="NV", path="test.aiff"):
    info = TrackInfo(
        path=path, energy=energy, key=key, bpm=bpm,
        structure=structure, intro_bars=int(structure[:-1]),
        flow_type=structure[-1], vibe=vibe, vocal=vocal,
    )
    tag_vec = encode_tags(info)
    dsp_vec = np.zeros(10, dtype=np.float32)
    return TrackFeatures(path=path, info=info, tag_vector=tag_vec, dsp_vector=dsp_vec)


def test_hard_bpm_cutoff():
    """BPM diff > 8% should be rejected."""
    config = GrouperConfig()
    a = _make_track(bpm=128)
    b = _make_track(bpm=150)  # ~17% diff
    assert recommend_score(a, b, config) == -float("inf")


def test_soft_bpm_penalty():
    """BPM 4-8% should be penalized but not rejected."""
    config = GrouperConfig()
    a = _make_track(bpm=128, path="a.aiff")
    b_close = _make_track(bpm=130, path="b.aiff")   # ~1.5%
    b_far = _make_track(bpm=137, path="c.aiff")      # ~7%
    s_close = recommend_score(a, b_close, config)
    s_far = recommend_score(a, b_far, config)
    assert s_close > s_far
    assert s_far > -float("inf")  # not rejected, just penalized


def test_similar_tracks_score_higher():
    config = GrouperConfig()
    a = _make_track(energy=3, vibe="HYPN", bpm=128, path="a.aiff")
    b_similar = _make_track(energy=3, vibe="HYPN", bpm=126, path="b.aiff")
    b_different = _make_track(energy=3, vibe="RAW", bpm=126, path="c.aiff")
    assert recommend_score(a, b_similar, config) > recommend_score(a, b_different, config)


def test_directional_scoring():
    """Score should differ based on direction (A->B vs B->A)."""
    config = GrouperConfig()
    a = _make_track(energy=3, structure="64H", bpm=128, path="a.aiff")
    b = _make_track(energy=4, structure="16D", bpm=128, path="b.aiff")
    s_ab = recommend_score(a, b, config)
    s_ba = recommend_score(b, a, config)
    # Different because intro usability, energy direction, breakdown risk differ
    assert s_ab != s_ba


def test_good_intro_scores_better():
    """Track with long hypnotic intro should score better as destination."""
    config = GrouperConfig()
    src = _make_track(energy=3, bpm=128, path="src.aiff")
    dest_long = _make_track(energy=3, bpm=128, structure="64H", path="long.aiff")
    dest_short = _make_track(energy=3, bpm=128, structure="16D", path="short.aiff")
    assert recommend_score(src, dest_long, config) > recommend_score(src, dest_short, config)

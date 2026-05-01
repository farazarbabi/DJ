"""Tests for taxonomy-derived mood vocabulary."""

from dj_tagger.moods import MOOD_LABELS, MOOD_NAME_BY_CODE, normalize_mood_code, normalize_mood_scores


def test_moods_loaded_from_taxonomy():
    assert len(MOOD_LABELS) >= 22
    assert "WARM" in MOOD_LABELS
    assert "WHSE" in MOOD_LABELS
    assert MOOD_NAME_BY_CODE["DRK"] == "dark"


def test_legacy_vibe_aliases_normalize_to_moods():
    assert normalize_mood_code("HYP") == "HYPN"
    assert normalize_mood_code("melodic") == "MEL"
    assert normalize_mood_code("dark") == "DRK"


def test_legacy_score_maps_expand_to_full_mood_space():
    scores = normalize_mood_scores({"HYP": 0.7, "DRK": 0.2})
    assert set(scores) == set(MOOD_LABELS)
    assert scores["HYPN"] == 0.7
    assert scores["DRK"] == 0.2

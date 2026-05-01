"""Tests for taxonomy-derived vocal profile vocabulary."""

from dj_tagger.vocals import (
    VOCAL_PROFILE_LABELS,
    VOCAL_PROFILE_NAME_BY_CODE,
    has_vocal_content,
    normalize_vocal_profile,
    normalize_vocal_profile_scores,
    score_vocal_profile,
)


def test_vocal_profiles_loaded_from_taxonomy():
    assert set(VOCAL_PROFILE_LABELS) == {"CHANT", "DUB", "FVOC", "INST", "SPK", "TOOL", "VOC"}
    assert VOCAL_PROFILE_NAME_BY_CODE["FVOC"] == "featured_vocal"


def test_legacy_vocal_aliases_normalize_to_profiles():
    assert normalize_vocal_profile("NV") == "INST"
    assert normalize_vocal_profile("V") == "VOC"
    assert normalize_vocal_profile("featured vocal") == "FVOC"


def test_legacy_score_maps_expand_to_full_profile_space():
    scores = normalize_vocal_profile_scores({"NV": 0.7, "V": 0.2})
    assert set(scores) == set(VOCAL_PROFILE_LABELS)
    assert scores["INST"] == 0.7
    assert scores["VOC"] == 0.2


def test_has_vocal_content_by_profile():
    assert has_vocal_content("VOC") is True
    assert has_vocal_content("CHANT") is True
    assert has_vocal_content("INST") is False
    assert has_vocal_content("DUB") is False


def test_score_vocal_profile_uses_speechiness():
    label, scores, has_vocals, confidence = score_vocal_profile(
        vocal_ratio=0.35,
        temporal_bonus=0.2,
        threshold=0.2,
        audio_features={"speechiness": 0.9, "instrumentalness": 0.1},
    )
    assert label == "SPK"
    assert scores["SPK"] > scores["VOC"]
    assert has_vocals is True
    assert 0.0 <= confidence <= 1.0

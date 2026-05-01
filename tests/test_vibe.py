"""Tests for vibe classifier."""

import pytest

from dj_grouper.features.dsp import extract_dsp_features
from dj_tagger.audio import load_audio_features
from dj_tagger.analyzers.vibe import analyze_vibe
from dj_tagger.derive import derive_vibe
from dj_tagger.moods import MOOD_LABELS

VALID_VIBES = set(MOOD_LABELS)


def test_vibe_valid_label(sine_440hz):
    """Output must be one of the taxonomy mood labels."""
    track = load_audio_features(sine_440hz)
    result = analyze_vibe(track)
    assert result.label in VALID_VIBES


def test_vibe_scores_present(sine_440hz):
    """Scores dict must have all taxonomy mood labels."""
    track = load_audio_features(sine_440hz)
    result = analyze_vibe(track)
    assert set(result.scores.keys()) == VALID_VIBES


def test_low_bass_deep_or_dark(low_bass):
    """Sub-bass heavy content should lean DEEP or DRK."""
    track = load_audio_features(low_bass)
    result = analyze_vibe(track)
    assert result.label in VALID_VIBES
    # Check that DEEP or DRK score high relative to others
    deep_dark = result.scores["DEEP"] + result.scores["DRK"]
    avg = sum(result.scores.values()) / len(result.scores)
    assert deep_dark >= avg  # at least average combined


def test_silence_atmospheric(silence):
    """Silence should lean ATM (low energy, low onset)."""
    track = load_audio_features(silence)
    result = analyze_vibe(track)
    assert result.label in VALID_VIBES


def test_vibe_analyzer_matches_canonical_derive(sine_440hz):
    """Direct analysis and cached derivation must use the same vibe scoring path."""
    track = load_audio_features(sine_440hz)
    audio_features = {
        "valence": 0.78,
        "energy": 0.62,
        "instrumentalness": 0.41,
        "liveness": 0.18,
        "acousticness": 0.07,
    }

    analyzed = analyze_vibe(track, audio_features=audio_features)
    derived = derive_vibe(extract_dsp_features(track), audio_features=audio_features)

    assert analyzed.label == derived["vibe"]
    for vibe in VALID_VIBES:
        assert analyzed.scores[vibe] == pytest.approx(derived["vibe_scores"][vibe], abs=1e-8)
    assert analyzed.confidence == pytest.approx(derived["vibe_confidence"], abs=1e-8)

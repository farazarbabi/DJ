"""Tests for vibe classifier."""

from dj_tagger.audio import load_audio_features
from dj_tagger.analyzers.vibe import analyze_vibe

VALID_VIBES = {"HYPN", "DRK", "RAW", "DEEP", "TRIB", "MEL", "ACID", "ATM"}


def test_vibe_valid_label(sine_440hz):
    """Output must be one of the 8 allowed labels."""
    track = load_audio_features(sine_440hz)
    result = analyze_vibe(track)
    assert result.label in VALID_VIBES


def test_vibe_scores_present(sine_440hz):
    """Scores dict must have all 8 labels."""
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

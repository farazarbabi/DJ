"""Tests for vocal detection."""

from dj_tagger.audio import load_audio_features
from dj_tagger.analyzers.vocal import analyze_vocal


def test_sub_bass_no_vocals(low_bass):
    """Pure sub-bass should have no vocals."""
    track = load_audio_features(low_bass)
    result = analyze_vocal(track)
    assert result.has_vocals is False


def test_silence_no_vocals(silence):
    """Silence should have no vocals."""
    track = load_audio_features(silence)
    result = analyze_vocal(track)
    assert result.has_vocals is False


def test_vocal_range_tonal(vocal_range_tone):
    """Tonal content in vocal range may or may not trigger.

    This tests that the function runs without error and returns a valid result.
    """
    track = load_audio_features(vocal_range_tone)
    result = analyze_vocal(track)
    assert isinstance(result.has_vocals, bool)
    assert 0.0 <= result.vocal_ratio <= 1.0

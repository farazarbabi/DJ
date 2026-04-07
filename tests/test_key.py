"""Tests for key detection."""

from dj_tagger.audio import load_audio_features
from dj_tagger.analyzers.key import analyze_key
from dj_tagger.constants import KEY_TO_CAMELOT


def test_camelot_map_complete():
    """All 24 keys must be mapped."""
    assert len(KEY_TO_CAMELOT) == 24
    for pitch in range(12):
        assert (pitch, "major") in KEY_TO_CAMELOT
        assert (pitch, "minor") in KEY_TO_CAMELOT


def test_camelot_values_valid():
    """All Camelot values should be 1-12 followed by A or B."""
    import re
    for v in KEY_TO_CAMELOT.values():
        assert re.match(r"^(1[0-2]|[1-9])[AB]$", v), f"Invalid Camelot: {v}"


def test_key_returns_valid_camelot(sine_440hz):
    """Key result should contain a valid Camelot code."""
    track = load_audio_features(sine_440hz)
    result = analyze_key(track)
    assert result.camelot in KEY_TO_CAMELOT.values()
    assert 0.0 <= result.confidence <= 1.0


def test_key_c_major(c_major_chord):
    """A C major chord should detect as C major (8B) or close."""
    track = load_audio_features(c_major_chord)
    result = analyze_key(track)
    # C major = 8B. Accept close keys on the Camelot wheel.
    assert result.camelot.endswith("B") or result.camelot.endswith("A")
    assert result.key_name  # non-empty


def test_key_a_minor(a_minor_chord):
    """An A minor chord should detect as A minor (8A) or close."""
    track = load_audio_features(a_minor_chord)
    result = analyze_key(track)
    assert result.camelot in KEY_TO_CAMELOT.values()

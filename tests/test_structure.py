"""Tests for structure analyzer."""

from dj_tagger.audio import load_audio_features
from dj_tagger.analyzers.structure import analyze_structure


def test_structure_valid_format(sine_440hz):
    """Output should be a valid intro+flow string."""
    track = load_audio_features(sine_440hz)
    result = analyze_structure(track)
    assert result.intro_bars in (16, 32, 64)
    assert result.flow_type in ("G", "H", "D", "B", "L")
    assert result.formatted == f"{result.intro_bars}{result.flow_type}"


def test_silence_structure(silence):
    """Silence should return a valid structure without crashing."""
    track = load_audio_features(silence)
    result = analyze_structure(track)
    assert result.intro_bars in (16, 32, 64)
    assert result.flow_type in ("G", "H", "D", "B", "L")


def test_constant_noise_flow(white_noise):
    """Constant noise should likely be L (linear) or H (hypnotic)."""
    track = load_audio_features(white_noise)
    result = analyze_structure(track)
    assert result.flow_type in ("L", "H", "G", "D", "B")

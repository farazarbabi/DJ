"""Tests for section detection."""

import numpy as np
import soundfile as sf

from dj_tagger.audio import load_audio_features
from dj_tagger.analyzers.sections import analyze_sections


def test_sections_on_sine(sine_440hz):
    """Section detection should not crash on a simple sine wave."""
    track = load_audio_features(sine_440hz)
    result = analyze_sections(track)
    assert result.n_bars >= 0


def test_sections_on_silence(silence):
    """Section detection should handle silence gracefully."""
    track = load_audio_features(silence)
    result = analyze_sections(track)
    assert result.n_bars >= 0


def test_sections_labels_valid(sine_440hz):
    """All section labels should be from the allowed set."""
    track = load_audio_features(sine_440hz)
    result = analyze_sections(track)
    valid = {"intro", "groove", "peak", "breakdown", "outro"}
    for s in result.sections:
        assert s.label in valid

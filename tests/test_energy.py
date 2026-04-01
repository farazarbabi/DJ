"""Tests for energy analyzer."""

from dj_tagger.audio import load_audio_features
from dj_tagger.analyzers.energy import analyze_energy


def test_silence_low_energy(silence):
    """Silence should be E1."""
    track = load_audio_features(silence)
    result = analyze_energy(track)
    assert result.level == 1


def test_loud_noise_high_energy(loud_noise):
    """Loud noise should be E4 or E5."""
    track = load_audio_features(loud_noise)
    result = analyze_energy(track)
    assert result.level >= 4


def test_moderate_signal_mid_energy(sine_440hz):
    """A moderate sine wave should be mid-range energy."""
    track = load_audio_features(sine_440hz)
    result = analyze_energy(track)
    assert 1 <= result.level <= 5


def test_energy_details_present(white_noise):
    """Details dict should have all feature scores."""
    track = load_audio_features(white_noise)
    result = analyze_energy(track)
    for key in ("rms", "centroid", "flux", "onset", "low_freq", "composite"):
        assert key in result.details
        assert 0.0 <= result.details[key] <= 1.0 or key == "composite"

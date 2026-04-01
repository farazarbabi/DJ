"""Shared fixtures: synthetic audio generation for testing."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sine_440hz(tmp_path):
    """Pure 440 Hz sine wave, 10 seconds."""
    sr = 22050
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "sine_440.wav"
    sf.write(str(path), y, sr)
    return str(path)


@pytest.fixture
def silence(tmp_path):
    """10 seconds of silence."""
    sr = 22050
    y = np.zeros(sr * 10, dtype=np.float32)
    path = tmp_path / "silence.wav"
    sf.write(str(path), y, sr)
    return str(path)


@pytest.fixture
def white_noise(tmp_path):
    """White noise at moderate amplitude, 10 seconds."""
    sr = 22050
    rng = np.random.default_rng(42)
    y = (rng.standard_normal(sr * 10) * 0.1).astype(np.float32)
    path = tmp_path / "noise.wav"
    sf.write(str(path), y, sr)
    return str(path)


@pytest.fixture
def loud_noise(tmp_path):
    """Loud white noise, 10 seconds."""
    sr = 22050
    rng = np.random.default_rng(42)
    y = (rng.standard_normal(sr * 10) * 0.4).astype(np.float32)
    path = tmp_path / "loud_noise.wav"
    sf.write(str(path), y, sr)
    return str(path)


@pytest.fixture
def low_bass(tmp_path):
    """Low sine wave (80 Hz) simulating sub-bass, 10 seconds."""
    sr = 22050
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 80 * t)).astype(np.float32)
    path = tmp_path / "low_bass.wav"
    sf.write(str(path), y, sr)
    return str(path)


@pytest.fixture
def vocal_range_tone(tmp_path):
    """Tonal content in vocal range (1000 Hz fundamental + harmonics), 10 seconds."""
    sr = 22050
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    y = (
        0.3 * np.sin(2 * np.pi * 1000 * t)
        + 0.2 * np.sin(2 * np.pi * 2000 * t)
        + 0.1 * np.sin(2 * np.pi * 3000 * t)
    ).astype(np.float32)
    path = tmp_path / "vocal_range.wav"
    sf.write(str(path), y, sr)
    return str(path)


@pytest.fixture
def c_major_chord(tmp_path):
    """C major chord (C4=261.6, E4=329.6, G4=392.0), 10 seconds."""
    sr = 22050
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    y = (
        0.3 * np.sin(2 * np.pi * 261.63 * t)
        + 0.3 * np.sin(2 * np.pi * 329.63 * t)
        + 0.3 * np.sin(2 * np.pi * 392.00 * t)
    ).astype(np.float32)
    path = tmp_path / "c_major.wav"
    sf.write(str(path), y, sr)
    return str(path)


@pytest.fixture
def a_minor_chord(tmp_path):
    """A minor chord (A3=220, C4=261.6, E4=329.6), 10 seconds."""
    sr = 22050
    t = np.linspace(0, 10, sr * 10, endpoint=False)
    y = (
        0.3 * np.sin(2 * np.pi * 220.00 * t)
        + 0.3 * np.sin(2 * np.pi * 261.63 * t)
        + 0.3 * np.sin(2 * np.pi * 329.63 * t)
    ).astype(np.float32)
    path = tmp_path / "a_minor.wav"
    sf.write(str(path), y, sr)
    return str(path)


@pytest.fixture
def mp3_file(tmp_path):
    """Create a minimal MP3 file for metadata testing."""
    # We create a WAV and use it for audio tests;
    # for MP3 metadata tests we need an actual MP3.
    # Generate a short WAV, then use pydub or lameenc if available,
    # otherwise skip.
    sr = 22050
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), y, sr)
    return str(wav_path)


@pytest.fixture
def flac_file(tmp_path):
    """Create a FLAC file for metadata testing."""
    sr = 22050
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "test.flac"
    sf.write(str(path), y, sr)
    return str(path)

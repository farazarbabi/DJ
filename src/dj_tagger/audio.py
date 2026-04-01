"""Audio loading and shared feature precomputation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray

from .constants import SAMPLE_RATE

logger = logging.getLogger(__name__)


@dataclass
class TrackAudio:
    """Precomputed audio features shared across all analyzers."""

    path: str
    y: NDArray[np.floating]
    sr: int
    y_harmonic: NDArray[np.floating]
    y_percussive: NDArray[np.floating]
    tempo: float
    beat_frames: NDArray[np.intp]
    duration: float


def load_audio_features(
    path: str | Path,
    max_duration: float | None = None,
) -> TrackAudio:
    """Load an audio file and precompute features used by all analyzers."""
    path = str(path)
    logger.debug("Loading %s (max_duration=%s)", path, max_duration)

    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=max_duration)
    duration = len(y) / sr

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    # librosa may return tempo as an array in some versions
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)

    logger.debug(
        "Loaded %s: %.1fs, %.1f BPM, %d beats",
        path, duration, tempo, len(beat_frames),
    )
    return TrackAudio(
        path=path,
        y=y,
        sr=sr,
        y_harmonic=y_harmonic,
        y_percussive=y_percussive,
        tempo=tempo,
        beat_frames=beat_frames,
        duration=duration,
    )

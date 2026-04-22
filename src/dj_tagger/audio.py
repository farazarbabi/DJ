"""Audio loading and shared feature precomputation."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray

# Suppress librosa n_fft warnings for short segments (zero-padded, harmless)
warnings.filterwarnings("ignore", message="n_fft=.*is too large", category=UserWarning)

from .constants import SAMPLE_RATE  # noqa: E402 — must follow warnings filter

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


def _read_native_bpm(path: str) -> float | None:
    """Read BPM from the file's native metadata (Rekordbox, etc.).

    Checks TBPM (ID3), BPM tags before falling back to None.
    """
    try:
        from mutagen import File
        audio = File(path)
        if audio is None or audio.tags is None:
            return None

        # ID3 (MP3, AIFF, WAV): TBPM frame
        for key in ("TBPM", "TBPM:"):
            if key in audio.tags:
                val = str(audio.tags[key])
                try:
                    bpm = float(val)
                    if 60 < bpm < 200:
                        logger.debug("Native BPM from %s: %.1f", key, bpm)
                        return bpm
                except (ValueError, TypeError):
                    pass

        # Vorbis (FLAC): BPM tag
        for key in ("bpm", "BPM", "TEMPO"):
            if key in audio:
                try:
                    bpm = float(audio[key][0])
                    if 60 < bpm < 200:
                        logger.debug("Native BPM from %s: %.1f", key, bpm)
                        return bpm
                except (ValueError, TypeError, IndexError):
                    pass

        # MP4 (M4A): tmpo atom
        if "\xa9tmpo" in audio:
            try:
                bpm = float(audio["\xa9tmpo"][0])
                if 60 < bpm < 200:
                    return bpm
            except (ValueError, TypeError, IndexError):
                pass
        if "tmpo" in audio:
            try:
                bpm = float(audio["tmpo"][0])
                if 60 < bpm < 200:
                    return bpm
            except (ValueError, TypeError, IndexError):
                pass

    except Exception:
        logger.debug("Could not read native BPM from %s", path, exc_info=True)

    return None


def _read_native_key(path: str) -> str | None:
    """Read key from the file's native TKEY/InitialKey tag (Camelot notation).

    Returns the Camelot key string (e.g. '9A') or None if not found.
    """
    try:
        from mutagen import File
        audio = File(path)
        if audio is None or audio.tags is None:
            return None

        # ID3 (MP3, AIFF, WAV): TKEY frame
        for key in ("TKEY", "TKEY:"):
            if key in audio.tags:
                val = str(audio.tags[key]).strip()
                if val:
                    logger.debug("Native key from %s: %s", key, val)
                    return val

        # Vorbis (FLAC): INITIALKEY tag
        for key in ("INITIALKEY", "initialkey", "KEY", "key"):
            if key in audio:
                try:
                    val = str(audio[key][0]).strip()
                    if val:
                        logger.debug("Native key from %s: %s", key, val)
                        return val
                except (IndexError, TypeError):
                    pass

    except Exception:
        logger.debug("Could not read native key from %s", path, exc_info=True)

    return None


def load_audio_features(
    path: str | Path,
    max_duration: float | None = None,
) -> TrackAudio:
    """Load an audio file and precompute features used by all analyzers."""
    path = str(path)
    logger.debug("Loading %s (max_duration=%s)", path, max_duration)

    # Try native BPM from file metadata first (Rekordbox, etc.)
    native_bpm = _read_native_bpm(path)

    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=max_duration)
    duration = len(y) / sr

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)

    # Use native BPM if available (much more accurate than librosa)
    if native_bpm is not None:
        logger.debug("Using native BPM %.1f (librosa detected %.1f)", native_bpm, tempo)
        tempo = native_bpm

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

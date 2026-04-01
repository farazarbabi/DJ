"""Musical key detection with Camelot notation output."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np

from ..audio import TrackAudio
from ..constants import KEY_TO_CAMELOT, MAJOR_PROFILE, MINOR_PROFILE, NOTE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class KeyResult:
    camelot: str
    key_name: str
    confidence: float


def analyze_key(track_audio: TrackAudio, use_essentia: bool = False) -> KeyResult:
    """Detect the musical key and return Camelot notation."""
    if use_essentia:
        try:
            return _detect_key_essentia(track_audio)
        except ImportError:
            logger.warning("Essentia not installed, falling back to librosa")
        except Exception:
            logger.warning("Essentia key detection failed, falling back to librosa", exc_info=True)

    return _detect_key_librosa(track_audio)


def _detect_key_librosa(track_audio: TrackAudio) -> KeyResult:
    """Key detection via chroma + Krumhansl-Schmuckler correlation."""
    chroma = librosa.feature.chroma_cqt(
        y=track_audio.y_harmonic, sr=track_audio.sr, n_chroma=12,
    )
    chroma_avg = np.mean(chroma, axis=1)
    chroma_max = np.max(chroma_avg)
    if chroma_max > 0:
        chroma_avg = chroma_avg / chroma_max

    major = np.array(MAJOR_PROFILE)
    minor = np.array(MINOR_PROFILE)

    best_corr = -2.0
    best_key = 0
    best_mode = "major"

    for shift in range(12):
        rotated = np.roll(chroma_avg, -shift)
        corr_maj = np.corrcoef(rotated, major)[0, 1]
        corr_min = np.corrcoef(rotated, minor)[0, 1]
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = shift
            best_mode = "major"
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = shift
            best_mode = "minor"

    camelot = KEY_TO_CAMELOT[(best_key, best_mode)]
    suffix = "m" if best_mode == "minor" else ""
    key_name = f"{NOTE_NAMES[best_key]}{suffix}"

    logger.debug("Key: %s (%s) confidence=%.3f", key_name, camelot, best_corr)
    return KeyResult(camelot=camelot, key_name=key_name, confidence=float(best_corr))


def _detect_key_essentia(track_audio: TrackAudio) -> KeyResult:
    """Key detection via Essentia KeyExtractor (EDMA profile)."""
    from essentia.standard import KeyExtractor
    from ..constants import ESSENTIA_KEY_TO_CAMELOT

    extractor = KeyExtractor(profileType="edma")
    key, scale, confidence = extractor(track_audio.y.astype(np.float32))

    camelot = ESSENTIA_KEY_TO_CAMELOT.get((key, scale), "??")
    suffix = "m" if scale == "minor" else ""
    key_name = f"{key}{suffix}"

    logger.debug("Key (essentia): %s (%s) confidence=%.3f", key_name, camelot, confidence)
    return KeyResult(camelot=camelot, key_name=key_name, confidence=float(confidence))

"""Energy level estimation (E1-E5)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa
import numpy as np

from ..audio import TrackAudio
from ..constants import ENERGY_BPM_NORM, ENERGY_NORM, ENERGY_THRESHOLDS, ENERGY_WEIGHTS

logger = logging.getLogger(__name__)


@dataclass
class EnergyResult:
    level: int | None
    details: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0  # distance from nearest threshold boundary


def _normalize(value: float, key: str) -> float:
    """Normalize a raw feature value to [0, 1] using constants."""
    lo, rng = ENERGY_NORM[key]
    return float(np.clip((value - lo) / rng, 0.0, 1.0))


def analyze_energy(track_audio: TrackAudio) -> EnergyResult:
    """Compute weighted energy composite and map to E1-E5."""
    y, sr = track_audio.y, track_audio.sr

    # Feature 1: RMS energy
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_norm = _normalize(rms_mean, "rms")

    # Feature 2: Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_norm = _normalize(centroid_mean, "centroid")

    # Feature 3: Spectral flux (rate of spectral change)
    S = np.abs(librosa.stft(y))
    diff = np.diff(S, axis=1)
    flux = np.sqrt(np.mean(diff ** 2, axis=0))
    flux_mean = float(np.mean(flux))
    flux_norm = _normalize(flux_mean, "flux")

    # Feature 4: Onset density (transients per second)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_env)
    duration_sec = len(y) / sr
    onset_rate = len(onsets) / duration_sec if duration_sec > 0 else 0.0
    onset_norm = _normalize(onset_rate, "onset")

    # Feature 5: Low-frequency energy ratio
    S_full = np.abs(librosa.stft(y, n_fft=2048))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = freqs < 150
    low_energy = float(np.mean(S_full[low_mask, :] ** 2))
    total_energy = float(np.mean(S_full ** 2))
    low_ratio = low_energy / (total_energy + 1e-8)
    low_norm = _normalize(low_ratio, "low_freq")

    # BPM normalization
    bpm = track_audio.tempo
    bpm_lo, bpm_rng = ENERGY_BPM_NORM
    bpm_norm = float(np.clip((bpm - bpm_lo) / bpm_rng, 0.0, 1.0))

    # Weighted composite
    features = {
        "rms": rms_norm,
        "centroid": centroid_norm,
        "flux": flux_norm,
        "onset": onset_norm,
        "low_freq": low_norm,
        "bpm": bpm_norm,
    }
    composite = sum(ENERGY_WEIGHTS[k] * v for k, v in features.items())

    # Map to E1-E5
    level = 1
    for i, threshold in enumerate(ENERGY_THRESHOLDS):
        if composite >= threshold:
            level = i + 2

    # Confidence: distance from nearest threshold boundary
    boundaries = [0.0] + ENERGY_THRESHOLDS + [1.0]
    min_dist = min(abs(composite - b) for b in boundaries)
    confidence = min(1.0, min_dist / 0.10)  # full confidence at 0.10 from boundary

    features["composite"] = composite
    logger.debug("Energy: E%d (composite=%.3f, conf=%.2f) %s", level, composite, confidence, features)
    return EnergyResult(level=level, details=features, confidence=confidence)

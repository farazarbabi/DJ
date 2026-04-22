"""Vibe classification into one of 8 categories."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa
import numpy as np

from ..audio import TrackAudio
from ..vibe_scoring import score_vibe_metrics

logger = logging.getLogger(__name__)


@dataclass
class VibeResult:
    label: str
    scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0  # margin between top two scores


def analyze_vibe(
    track_audio: TrackAudio,
    audio_features: dict[str, float] | None = None,
) -> VibeResult:
    """Classify the track's vibe using spectral heuristics + optional API features.

    Labels: HYPN, DRK, RAW, DEEP, TRIB, MEL, ACID, ATM

    audio_features: optional dict with keys like 'valence', 'instrumentalness',
    'liveness', 'energy', 'acousticness' (0-1 scale). When provided, these
    contribute additional scoring signals per vibe via [vibe.songstats] settings.
    """
    y, sr = track_audio.y, track_audio.sr
    y_h = track_audio.y_harmonic
    y_p = track_audio.y_percussive

    S = np.abs(librosa.stft(y))

    # --- Precompute features ---
    rms = float(np.mean(librosa.feature.rms(y=y)[0]))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_var = float(np.var(centroid)) / (centroid_mean ** 2 + 1e-8)

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)[0]))

    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr)
    chroma_var = float(np.mean(np.var(chroma, axis=1)))

    spectral_stability = 1.0 - min(1.0, centroid_var * 10)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_density = float(np.mean(onset_env))
    onset_var = float(np.var(onset_env)) / (float(np.mean(onset_env)) ** 2 + 1e-8)

    harmonic_energy = float(np.mean(y_h ** 2))
    percussive_energy = float(np.mean(y_p ** 2))
    perc_ratio = percussive_energy / (harmonic_energy + percussive_energy + 1e-8)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = freqs < 150
    S_power = S ** 2
    low_ratio = float(np.mean(S_power[low_mask, :])) / (float(np.mean(S_power)) + 1e-8)

    flux_raw = np.diff(S, axis=1)
    flux = float(np.mean(np.sqrt(np.mean(flux_raw ** 2, axis=0))))

    # --- Score each vibe (all params from settings.toml) ---
    metrics = {
        "rms": rms,
        "centroid_mean": centroid_mean,
        "centroid_cv": centroid_var,
        "flatness": flatness,
        "onset_density": onset_density,
        "onset_var": onset_var,
        "perc_ratio": perc_ratio,
        "low_ratio": low_ratio,
        "flux": flux,
        "chroma_var": chroma_var,
        "spectral_stability": spectral_stability,
    }
    best_label, scores, confidence = score_vibe_metrics(metrics, audio_features=audio_features)

    logger.debug("Vibe: %s (conf=%.2f) scores=%s", best_label, confidence, {k: f"{v:.3f}" for k, v in scores.items()})
    return VibeResult(label=best_label, scores=scores, confidence=confidence)

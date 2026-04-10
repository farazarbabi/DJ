"""Vibe classification into one of 8 categories."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import librosa
import numpy as np

from ..audio import TrackAudio

logger = logging.getLogger(__name__)


@dataclass
class VibeResult:
    label: str
    scores: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0  # margin between top two scores


def analyze_vibe(track_audio: TrackAudio) -> VibeResult:
    """Classify the track's vibe using spectral heuristics.

    Labels: HYPN, DRK, RAW, DEEP, TRIB, MEL, ACID, ATM
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
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]))

    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr)
    chroma_strength = float(np.mean(np.max(chroma, axis=0)))

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

    spec_avg = np.mean(S, axis=1)
    peakiness = float(np.max(spec_avg)) / (float(np.mean(spec_avg)) + 1e-8)

    # --- Helper ---
    def _clip01(v: float) -> float:
        return max(0.0, min(1.0, v))

    # --- Score each vibe ---
    # Evidence-based approach: MEL is the baseline for electronic music.
    # Other vibes require strong, specific spectral evidence to override.
    # Absolute thresholds calibrated on real electronic music features:
    #   onset_var: 1.2-3.5   flatness: 0.006-0.049   low_ratio: 34-67
    #   flux: 1.6-4.4        perc_ratio: 0.12-0.51

    chroma_var = float(np.mean(np.var(chroma, axis=1)))

    scores: dict[str, float] = {}

    # MEL: baseline — chroma_var drives melodic evidence, low flatness = clean
    mel_chroma = min(1.0, chroma_var / 0.065)
    mel_lowflat = _clip01((0.025 - flatness) / 0.015)
    scores["MEL"] = 0.38 + 0.25 * mel_chroma + 0.12 * mel_lowflat

    # DRK: aggressive rhythmic character + noisiness + bass-heavy
    drk_onset = _clip01((onset_var - 1.5) / 2.0)
    drk_flat = _clip01((flatness - 0.010) / 0.030)
    drk_low = _clip01((low_ratio - 45) / 25)
    drk_flux = _clip01((flux - 2.0) / 2.5)
    scores["DRK"] = (
        0.25 * drk_onset
        + 0.30 * drk_flat
        + 0.25 * drk_low
        + 0.20 * drk_flux
    )

    # TRB: very high percussion ratio — must be clearly percussive, not noisy
    trib_perc = _clip01((perc_ratio - 0.40) / 0.15)
    trib_clean = _clip01((0.03 - flatness) / 0.02)
    scores["TRIB"] = 0.70 * trib_perc + 0.30 * trib_clean

    # RAW: non-electronic character (very low bass ratio + sparse)
    raw_nobass = _clip01((42 - low_ratio) / 12)
    raw_sparse = _clip01((1.5 - onset_density) / 0.8)
    raw_lowvar = _clip01((1.5 - onset_var) / 1.0)
    scores["RAW"] = 0.45 * raw_nobass + 0.30 * raw_sparse + 0.25 * raw_lowvar

    # HYPN: high stability, low onset variance
    scores["HYPN"] = 0.50 * spectral_stability + 0.50 * _clip01((2.0 - onset_var) / 1.5)

    # DEEP: low centroid + low loudness + bass-heavy
    scores["DEEP"] = (
        0.35 * _clip01((2000 - centroid_mean) / 1000)
        + 0.35 * _clip01((0.22 - rms) / 0.10)
        + 0.30 * _clip01((low_ratio - 50) / 20)
    )

    # ACID: extreme spectral movement (filter sweeps) — very rare
    scores["ACID"] = _clip01((centroid_var - 0.75) / 0.40)

    # ATM: sparse + quiet
    scores["ATM"] = (
        0.50 * _clip01((1.5 - onset_density) / 1.0)
        + 0.50 * _clip01((0.20 - rms) / 0.08)
    )

    # Clamp all scores to [0, 1]
    scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

    best_label = max(scores, key=scores.get)  # type: ignore[arg-type]

    # Confidence: margin between top two scores
    sorted_scores = sorted(scores.values(), reverse=True)
    confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else 1.0

    logger.debug("Vibe: %s (conf=%.2f) scores=%s", best_label, confidence, {k: f"{v:.3f}" for k, v in scores.items()})
    return VibeResult(label=best_label, scores=scores, confidence=confidence)
